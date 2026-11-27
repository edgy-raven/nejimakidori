"""Expert records restore policy and belief structures without RL targets."""

import tensorflow as tf

from training import restoration


def test_recorded_structures_restore_policy_and_beliefs():
    import json
    import pathlib

    import numpy

    from log_dataset import replay, scoring, training_data
    from model import actor as actor_layers
    from model import beliefs
    from model import network as joint_network

    events = json.loads(
        pathlib.Path(__file__)
        .parents[1]
        .joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    targets = scoring.round_labels(events + [{"type": "end_game"}])
    observations = list(replay.Replay("restoration", events).observations())[:2]
    raw = tf.constant(
        [
            training_data.serialize(row, targets[row.kyoku_id])
            for row in observations
        ]
    )
    features, labels, _ = training_data.parse_batch(raw)
    head, chosen, legal = restoration.policy_choices(features, labels)
    numpy.testing.assert_array_equal(
        tf.gather(legal, chosen, batch_dims=1), True
    )
    network = joint_network.JointPolicyPayment()
    with tf.GradientTape(persistent=True) as tape:
        outputs = network.actor(features, training=False)
        policy = {
            "policy_loss": -tf.reduce_mean(
                tf.gather(
                    actor_layers.distribution(outputs, head, legal),
                    chosen,
                    batch_dims=1,
                )
            ),
            **beliefs.completion_loss_terms(labels, outputs),
        }
        belief, _ = beliefs.belief_loss_terms(features, labels, outputs)
        policy_loss = tf.add_n(list(policy.values()))
        belief_loss = tf.add_n(list(belief.values()))
    # Restoration is ordinary supervised learning of the recorded choice;
    # critic values and magnetic references are not needed.
    assert numpy.isfinite(float(policy_loss))
    assert numpy.isfinite(float(belief_loss))
    refinement_gradient = tape.gradient(
        belief_loss, network.beliefs.refinement.layers[-1].kernel
    )
    assert numpy.isfinite(refinement_gradient).all()
    assert tf.linalg.global_norm([refinement_gradient]) > 0
    for loss in (policy_loss, belief_loss):
        gradients = tape.gradient(loss, network.actor.trainable_variables)
        assert (
            tf.linalg.global_norm(
                [gradient for gradient in gradients if gradient is not None]
            )
            > 0
        )
    assert not network.critic.built


def test_three_expert_passes_fit_critic_and_release_main_loop(
    tmp_path, monkeypatch
):
    import itertools
    import json
    import pathlib
    import types
    import zipfile

    import numpy

    from integration import reference_policy
    from log_dataset import (
        _native_collection,
        records,
        replay,
        root_states,
        training_data,
    )
    from model import actor as actor_layers
    from model import beliefs, candidate_predictions, export, payment_scoring
    from training import expert, imitation

    events = json.loads(
        pathlib.Path(__file__)
        .parents[1]
        .joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    events[0]["player_ranks"]["retained_seats"] = [0, 1, 2, 3]
    raw = []
    for row in itertools.islice(
        replay.Replay("expert-0", events).observations(), 3
    ):
        example = tf.train.Example.FromString(
            records.serialize_observation(row)
        )
        legal = numpy.flatnonzero(row.features["discard_action_legal"])
        reference = (
            legal[legal != row.discard_action][0]
            if not raw
            else row.discard_action
        )
        example.features.feature[
            "expert/mortal_action"
        ].bytes_list.value.append(numpy.int16(reference).tobytes())
        raw.append(example.SerializeToString())
    batch = expert.decode_records(raw)
    valid = tf.constant([True, False, True])
    labels = {
        k: tf.concat([v[:1], -tf.ones_like(v[1:2]), v[2:]], axis=0)
        for k, v in batch["labels"].items()
    }
    network = export.build_network(batch["features"])
    outputs = network.actor(batch["features"])
    expected, _ = beliefs.belief_loss_terms(batch["features"], labels, outputs)
    expected.update(beliefs.completion_loss_terms(labels, outputs))
    expected.update(
        candidate_predictions.loss_terms(batch["features"], labels, outputs)
    )
    learner = imitation.ImitationTrainer(
        network=network,
        features=batch["features"],
        learning_rate=1e-5,
    )
    assert batch["chosen"][0] != batch["reference_action"][0]
    expected_expert = -tf.reduce_mean(
        tf.boolean_mask(
            tf.gather(
                actor_layers.distribution(
                    outputs, batch["head"], batch["legal"]
                ),
                batch["expert"],
                batch_dims=1,
            ),
            valid,
        )
    )
    critic_before = network.critic.event.kernel.numpy().copy()
    encoder_before = (
        network.actor.policy.get_layer("board_block_3_contract")
        .kernel.numpy()
        .copy()
    )
    metrics = learner.update.python_function(batch, valid)
    numpy.testing.assert_allclose(metrics["unweighted_ce"], expected_expert)
    assert not numpy.array_equal(critic_before, network.critic.event.kernel)
    assert not numpy.array_equal(
        encoder_before,
        network.actor.policy.get_layer("board_block_3_contract").kernel,
    )
    assert float(metrics["payment_nll"]) > 0
    numpy.testing.assert_allclose(
        metrics["critic_loss"],
        metrics["payment_nll"] + metrics["critic_constraint_loss"],
    )

    assert len(expected) == 7
    for name, loss in expected.items():
        numpy.testing.assert_allclose(metrics[name], loss, atol=2e-5)
    numpy.testing.assert_allclose(
        metrics["joint_loss"],
        metrics["weighted_ce"]
        + metrics["critic_loss"]
        + sum(expected.values()),
        rtol=1e-5,
    )

    # Tracing uses different BF16 fusion boundaries from eager execution.
    # Check that the real compiled update also fits the critic successfully.
    critic_before = network.critic.event.kernel.numpy().copy()
    compiled = learner.update(batch, valid)
    assert numpy.isfinite(float(compiled["joint_loss"]))
    assert float(compiled["payment_nll"]) > 0
    assert not numpy.array_equal(critic_before, network.critic.event.kernel)

    # Mortal labels supply Q baselines; human actions supply policy labels.
    # Observed human-action settlements train Q in all three passes; no
    # counterfactual settlement is assigned to Mortal's unplayed action.
    data = tmp_path / "data"
    data.mkdir()
    archives = tmp_path / "archives"
    archives.mkdir()
    with zipfile.ZipFile(archives / "human.zip", "w") as archive:
        archive.writestr(
            "expert-0.mjson",
            "\n".join(json.dumps(event) for event in events),
        )
    (data / "games.jsonl").write_text(
        json.dumps(
            {
                "game_id": "expert-0",
                "archive": str(archives / "human.zip"),
                "member": "expert-0.mjson",
            }
        )
        + "\n"
    )
    with tf.io.TFRecordWriter(
        str(data / "train-00000.tfrecord.gz"), options="GZIP"
    ) as stream:
        for value in raw * 16:
            example = tf.train.Example.FromString(value)
            example.features.feature[
                "critic/partition"
            ].int64_list.value.append(0)
            for name in training_data.SAMPLING_FIELDS:
                example.features.feature[name].int64_list.value.append(1)
            stream.write(example.SerializeToString())
    (data / "dataset_summary.json").write_text(
        json.dumps(
            {
                "input_schema": _native_collection.INPUTS,
                "archives": str(archives),
                "player_ranks": {},
                "reference_schema": records.REFERENCE_INPUTS,
                "label_schema": records.LABELS,
                "critic_auxiliary": training_data.AUXILIARY,
                "critic_bonus_han": list(range(14)),
                "critic_scoring_states": [
                    list(state) for state in payment_scoring.SCORES
                ],
                "validation_sample": training_data.VALIDATION_SAMPLE,
                "sampling_fields": list(training_data.SAMPLING_FIELDS),
                "partition_counts": [48] + [0] * 9,
                "preparation": {"mortal_policy": {"path": "recorded-mortal"}},
                "shard_manifest": [
                    {
                        "path": "train-00000.tfrecord.gz",
                        "counts": [48] + [0] * 9,
                        "sampling_counts": [[48, 48]] + [[0, 0]] * 9,
                    }
                ],
            }
        )
    )
    root_states.run(
        types.SimpleNamespace(
            data=data,
            output=data / "root-states",
            workers=1,
            limit_games=None,
        )
    )
    initial = tmp_path / "initial.weights.h5"
    network.save_weights(initial)
    command = tmp_path / "match.json"
    command.write_text(
        json.dumps(
            [
                "match-only-after-init",
                "--games",
                "10000",
                "{weights}",
                "{output}",
            ]
        )
    )
    learners = []
    rollout_learners = []
    rollout_trainer_class = expert.mmd_trainer.MMDTrainer
    critic_boundaries = []
    trainer_class = imitation.ImitationTrainer

    def track_optimizer(**kwargs):
        learner = trainer_class(**kwargs)
        assert int(learner.optimizer.iterations) == 0
        learners.append(learner)
        critic_boundaries.append(
            learner.network.critic.event.kernel.numpy().copy()
        )
        return learner

    def track_rollout_optimizer(**kwargs):
        learner = rollout_trainer_class(**kwargs)
        rollout_learners.append(learner)
        assert int(learner.optimizer.iterations) == 0
        return learner

    monkeypatch.setattr(
        expert.mmd_trainer, "MMDTrainer", track_rollout_optimizer
    )
    monkeypatch.setattr(
        expert.fixed_reference, "Policy", lambda path: reference_policy.Policy()
    )
    monkeypatch.setattr(imitation, "ImitationTrainer", track_optimizer)
    monkeypatch.setattr(
        expert.self_play,
        "predictor",
        lambda actor: reference_policy.uniform_logits,
    )
    monkeypatch.setattr(
        expert.collection,
        "root_scorer",
        lambda network: lambda features: numpy.where(
            features["discard_action_legal"], numpy.arange(37), -numpy.inf
        ),
    )
    eager = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(True)
    try:
        expert.run(
            types.SimpleNamespace(
                data=str(data),
                initial=str(initial),
                reference="fixture",
                output=str(tmp_path / "candidate"),
                evaluation_command=str(command),
                batch_size=48,
                rollout_batch_size=64,
                seed=13,
            )
        )
    finally:
        tf.config.run_functions_eagerly(eager)
    assert len(learners) == 3
    critic_boundaries.append(
        learners[-1].network.critic.event.kernel.numpy().copy()
    )
    assert all(
        not numpy.array_equal(before, after)
        for before, after in zip(critic_boundaries, critic_boundaries[1:])
    )
    assert sum(int(value.optimizer.iterations) for value in learners) > len(
        learners
    )
    assert len({id(value.optimizer) for value in learners}) == 3
    root = tmp_path / "candidate"
    plan = json.loads((root / "plan.json").read_text())
    assert plan["method"] == "q-weighted-expert-imitation"
    assert plan["passes"] == 3 and plan["expected_exposures"] == 144
    assert plan["learning_rates"] == {"imitation": 1e-5}
    assert plan["beta_target"] == 0.75
    assert plan["sampling"] == {
        "natural": 0.5,
        "focused": 0.5,
    }
    assert plan["pass_exposures"] == 48
    status = json.loads((root / "status.json").read_text())
    assert status["stage"] == "main-loop-ready"
    assert status["pass"] == 3 and status["steps"] > 3
    assert status["steps"] == sum(
        int(value.optimizer.iterations) for value in learners + rollout_learners
    )
    assert status["exposures"] == 144
    assert status["actor_exposures"] == status["critic_exposures"]
    assert status["mc_actor_exposures"] > 0
    assert 144 < status["critic_exposures"] <= 288
    assert plan["imitation_rollouts"]["collection_interval_batches"] == 16
    assert plan["imitation_rollouts"]["continuations_per_action"] == 1
    assert status["mc_human_positions"] == status["exposures"]
    assert status["mc_simulated_decisions"] > 0
    assert "mc_decision_balance" not in status
    assert plan["imitation_rollouts"]["mmd"]["magnet"] == (
        reference_policy.Policy.provenance
    )
    assert all(
        int(learner.optimizer.iterations) > 0 for learner in rollout_learners
    )
    completed = json.loads((root / "training-complete.json").read_text())
    assert completed["weights"] == str(root / "final.weights.h5")
    assert completed["exposures"] == 144
    assert all(
        (root / name).exists()
        for name in (
            "pass-1.weights.h5",
            "pass-2.weights.h5",
            "pass-3.weights.h5",
            "final.weights.h5",
        )
    )
    progress = [
        json.loads(line)
        for line in (root / "progress.jsonl").read_text().splitlines()
    ]
    updates = [row for row in progress if "weighted_ce" in row]
    for expert_pass in range(1, 4):
        stages = [
            row["stage"]
            for row in progress
            if row["pass"] == expert_pass and "trained_positions" in row
        ]
        assert stages[1:] == ["imitation-rollout", "imitation-mmd"]
        assert stages[0] == "imitation"
    assert [row["pass"] for row in updates] == [1, 2, 3]
    assert [row["stage"] for row in updates] == ["imitation"] * 3
    assert all("ess_fraction" in row for row in updates)
    for row in updates:
        numpy.testing.assert_allclose(
            row["joint_loss"],
            row["weighted_ce"]
            + row["critic_loss"]
            + sum(row[name] for name in expected),
            rtol=1e-5,
        )


def test_expert_decoding_maps_recorded_mortal_call_followups():
    import json
    import pathlib

    import numpy

    from log_dataset import records, replay
    from training import expert

    events = json.loads(
        pathlib.Path(__file__)
        .parents[1]
        .joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    raw = []
    expected = []
    covered = set()
    for row in replay.Replay("mortal-baseline", events).observations():
        if row.features["decision_phase"] != 1:
            continue
        for kind, options, offset in (
            ("chi", slice(0, 3), 1),
            ("pon", slice(3, 4), 112),
        ):
            choices = numpy.flatnonzero(
                row.features["call_discard_legal"][options]
            )
            if kind in covered or not len(choices):
                continue
            mortal = offset + int(choices[-1])
            example = tf.train.Example.FromString(
                records.serialize_observation(row)
            )
            example.features.feature[
                "expert/mortal_action"
            ].bytes_list.value.append(numpy.int16(mortal).tobytes())
            raw.append(example.SerializeToString())
            expected.append(111 + mortal)
            covered.add(kind)
        if len(covered) == 2:
            break
    assert covered == {"chi", "pon"}
    batch = expert.decode_records(raw)
    numpy.testing.assert_array_equal(batch["reference_action"], expected)
    numpy.testing.assert_array_equal(batch["head"], [2, 2])
