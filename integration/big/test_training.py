"""Phasic training preserves behavior probabilities and exact alternating budgets."""

import json
import pathlib
import types

import numpy
import pytest
import tensorflow as tf

from log_dataset import focus_sampling, training_data
from model import actions
from model import actor as actor_layers
from model import losses
from model import network as joint_network
from model import self_play, training


class CpuStaging:
    """Exercise real input staging with the scheduling tests' toy updates."""

    strategy = tf.distribute.MirroredStrategy(devices=["/CPU:0"])
    distribute = training.DistributedTrainer.distribute

    def update(self, batch, inputs, count, awr, anchored=False):
        return self.auxiliary_update(
            tf.nest.map_structure(
                lambda value: self.strategy.experimental_local_results(value)[
                    0
                ],
                inputs,
            ),
            awr,
            anchored=anchored,
        )


def test_prefetch_preserves_order_and_partial_budget():
    seen = []

    class Trainer(CpuStaging):
        def auxiliary_update(self, inputs, awr, anchored=False):
            seen.extend(inputs.numpy().tolist())
            return {"sum": tf.reduce_sum(inputs)}

    stream = iter([numpy.arange(start, start + 3) for start in (0, 3, 6)])
    # A shorter fixed interval on resume can leave its target behind progress.
    for exposures in (-1, 0, 5):
        updates = list(
            training.auxiliary_updates(
                trainer=Trainer(),
                stream=stream,
                exposures=exposures,
                batch_size=3,
                awr=True,
            )
        )
        assert sum(count for count, _ in updates) == max(exposures, 0)
    assert seen == [0, 1, 2, 3, 4]
    numpy.testing.assert_array_equal(next(stream), [6, 7, 8])


def test_validation_covers_partial_batch_and_saves_best_weights(
    tmp_path, monkeypatch
):
    from model import training

    rows = tf.constant([str(n) for n in range(130)])
    (tmp_path / "dataset_summary.json").write_text(
        json.dumps({"validation_examples": 130})
    )
    monkeypatch.setattr(
        training_data,
        "dataset",
        lambda **kwargs: tf.data.Dataset.from_tensor_slices(rows).batch(8),
    )
    monkeypatch.setattr(
        training_data,
        "parse_batch",
        lambda rows: (rows, {}, {"points": tf.zeros(tf.shape(rows))}),
    )
    monkeypatch.setattr(training, "policy_rates", lambda metrics: metrics)
    seen = []

    def evaluate(batch, statistics=None):
        if statistics is not None:
            return {
                "policy_loss": tf.constant(0.5),
                "payment_nll": tf.constant(1.2),
                "belief_loss": tf.constant(0.1),
                "completion_yaku_loss": tf.constant(0.01),
            }
        seen.extend(batch[0].numpy().tolist())
        return {
            "statistics": tf.constant(0.0),
            "counts": {"example_count": len(batch[0])},
        }

    saved = []
    writer = tf.summary.create_file_writer(str(tmp_path / "events"))
    for step in (1, 2):
        seen.clear()
        result = training.validate(
            args=types.SimpleNamespace(
                output=str(tmp_path),
                data=str(tmp_path),
                batch_size=8,
            ),
            network=types.SimpleNamespace(save_weights=saved.append),
            strategy=tf.distribute.get_strategy(),
            evaluate=evaluate,
            writer=writer,
            step=step,
        )
        assert seen == rows.numpy().tolist()
        assert result["examples"] == len(seen)
        assert (tmp_path / f"validation-{step}.json").exists()
    assert saved == [tmp_path / "best-policy.weights.h5"]
    writer.close()
    import tensorboard.backend.event_processing.event_accumulator as events

    log = events.EventAccumulator(str(tmp_path / "events"))
    log.Reload()
    assert set(log.Tags()["tensors"]) == {
        "validation/policy_loss",
        "validation/payment_nll",
        "validation/belief_loss",
    }
    assert result["completion_yaku_loss"] > 0
    assert (
        json.loads((tmp_path / "validation-2.json").read_text())[
            "completion_yaku_loss"
        ]
        == result["completion_yaku_loss"]
    )


@pytest.mark.parametrize(
    "head, gap, tail",
    [("response_policy_logits", 0, 147), ("riichi_policy_logits", 36, 35)],
)
def test_entropy_explores_followups_without_biasing_the_parent(head, gap, tail):
    parent = tf.Variable(
        [numpy.log(0.25), 0.0, numpy.log(4.0), -1e9], dtype=tf.float32
    )
    followup = tf.Variable(
        [[numpy.log(0.8), numpy.log(0.2)]] * 4, dtype=tf.float32
    )
    with tf.GradientTape(persistent=True) as tape:
        logits = tf.concat(
            [
                tf.zeros((4, 1)),
                tf.fill((4, gap), -1e9),
                parent[:, None] + tf.nn.log_softmax(followup),
                tf.fill((4, tail), -1e9),
            ],
            axis=1,
        )
        log_probability = tf.nn.log_softmax(logits)
        entropy = actor_layers.action_entropy(log_probability, head)
        total = tf.reduce_sum(entropy)
        offline = actor_layers.policy_entropy(
            {
                name: tf.fill((4,), 0 if name == head else -1)
                for name in actions.TRAIN_POLICY_HEADS
            },
            {
                name: logits if name == head else tf.zeros((4, 150))
                for name in actions.TRAIN_POLICY_HEADS
            },
        )
    # Parent exploration is exactly binary entropy, independent of the
    # continuation's entropy. Impossible call branches have finite gradients.
    expected = [0.16 * numpy.log(4), 0, -0.16 * numpy.log(4), 0]
    numpy.testing.assert_allclose(
        tape.gradient(total, parent), expected, atol=1e-6
    )
    numpy.testing.assert_allclose(
        tape.gradient(offline, parent), numpy.array(expected) / 4, atol=1e-6
    )
    gradient = tape.gradient(total, followup).numpy()
    assert (gradient[:3, 0] < 0).all()
    assert (gradient[:3, 1] > 0).all()
    numpy.testing.assert_array_equal(gradient[3], 0)
    numpy.testing.assert_allclose(
        entropy,
        -tf.reduce_sum(tf.exp(log_probability) * log_probability, axis=-1),
        atol=1e-6,
    )


def test_clipped_ppo_and_forward_kl_have_the_expected_gradients():
    behavior = tf.math.log([[0.5, 0.5]] * 2 + [[0.1, 0.9]] * 3 + [[0.5, 0.5]])
    logits = tf.Variable(tf.math.log([[0.8, 0.2], [0.2, 0.8]] * 3))
    # Ratios: 1.6, 0.4, 8, 2, 8, 0.4. Cover ordinary clipping,
    # a capped negative tail, and uncapped negative/positive gradients.
    with tf.GradientTape() as tape:
        loss = actor_layers.ppo_loss(
            current=tf.nn.log_softmax(logits),
            behavior=behavior,
            chosen=tf.zeros(6, tf.int32),
            advantages=tf.constant([1.0, -1.0, -2.0, -2.0, 2.0, 2.0]),
            clip=0.2,
            dual_clip=3.0,
            mean=lambda loss, mask: loss,
        )
    numpy.testing.assert_allclose(
        loss, [-1.2, 0.8, 6.0, 4.0, -2.4, -0.8], atol=1e-6
    )
    numpy.testing.assert_allclose(
        tape.gradient(loss, logits),
        [[0, 0], [0, 0], [0, 0], [3.2, -3.2], [0, 0], [-0.64, 0.64]],
        atol=1e-6,
    )
    reference = tf.Variable(behavior)
    with tf.GradientTape(persistent=True) as tape:
        kl = actor_layers.forward_kl(reference, tf.nn.log_softmax(logits))
    assert tape.gradient(kl, reference) is None
    assert float(kl) > 0
    assert numpy.linalg.norm(tape.gradient(kl, logits)) > 0
    numpy.testing.assert_allclose(
        actor_layers.forward_kl(behavior, behavior), 0
    )

    # Multiple active heads sum within a decision; unlabeled rows add no mass.
    reference = tf.Variable([[3.0, 0.0]] * 4)
    current = tf.Variable(tf.zeros((4, 2)))
    labels = {
        name: tf.constant(
            [0, 0, -1, -1]
            if head == 0
            else ([0, -1, 0, -1] if head == 1 else [-1] * 4)
        )
        for head, name in enumerate(actions.TRAIN_POLICY_HEADS)
    }
    with tf.GradientTape(persistent=True) as tape:
        kl = actor_layers.decision_kl(
            {name: reference for name in labels},
            {name: current for name in labels},
            labels,
        )
    numpy.testing.assert_allclose(
        kl,
        4
        / 3
        * actor_layers.forward_kl(
            tf.nn.log_softmax(reference), tf.nn.log_softmax(current)
        ),
        atol=1e-7,
    )
    assert tape.gradient(kl, reference) is None
    assert numpy.linalg.norm(tape.gradient(kl, current)) > 0


def test_alternating_updates_use_awr_for_every_offline_segment(monkeypatch):
    updates = []
    monkeypatch.setattr(
        training, "save_checkpoint", lambda manager: manager.save()
    )

    class Trainer(CpuStaging):
        packing = None
        auxiliary_optimizer = types.SimpleNamespace(iterations=tf.Variable(0))

        def auxiliary_update(self, inputs, awr, anchored=False):
            updates.append(("awr" if awr else "foundation", len(inputs)))
            self.auxiliary_optimizer.iterations.assign_add(1)
            return {}

    def ppo_block(**kwargs):
        updates.append(("ppo", kwargs["remaining"]))
        return kwargs["remaining"], {}

    monkeypatch.setattr(
        training,
        "offline_stream",
        lambda *args: iter([numpy.ones((2, 1))] * 10),
    )
    monkeypatch.setattr(training, "ppo_block", ppo_block)
    monkeypatch.setattr(
        training,
        "collect_cohort",
        lambda **kwargs: ({"chosen": numpy.zeros(2)}, 1, {}),
    )
    progress = {
        name: tf.Variable(0, dtype=tf.int64)
        for name in (
            "foundation",
            "awr",
            "ppo",
            "cycle",
            "cohort",
            "collected",
            "games",
        )
    }
    reports = []
    training.train_cycles(
        args=types.SimpleNamespace(
            awr_exposures=8,
            awr_only_exposures=0,
            final_awr_exposures=0,
            final_awr_kl=0.1,
            prior_weight=1.0,
            ppo_exposure_ratio=0.5,
            block_exposures=4,
            cycle_blocks=1,
            batch_size=2,
            seed=1,
            ppo_epochs=1,
            cohort_exposures=2,
        ),
        trainer=Trainer(),
        collector=None,
        progress=progress,
        manager=types.SimpleNamespace(save=lambda: None),
        ppo_manager=types.SimpleNamespace(save=lambda: None),
        report=lambda stage, metrics: reports.append(stage),
    )
    assert updates == [
        ("awr", 2),
        ("awr", 2),
        ("ppo", 2),
        ("awr", 2),
        ("awr", 2),
        ("ppo", 2),
    ]
    assert int(progress["awr"]) == 8
    assert int(progress["ppo"]) == 4
    assert "foundation_repair" not in reports


@pytest.mark.parametrize("needed", [4, 8])
def test_collection_report_uses_cohort_deltas_and_snapshot_provenance(needed):
    class Collector:
        metrics = dict(
            **self_play.collection_counts(),
            inference_rows=100,
            inference_calls=10,
        )

        def collect(self, games, seed):
            for name, value in dict(
                games=games,
                hands=2,
                wins=2,
                tsumo_wins=1,
                dealin_hands=1,
                win_points=12000,
                dealin_points=8000,
                inference_rows=4,
                inference_calls=2,
            ).items():
                self.metrics[name] += value
            self.batch = {"chosen": numpy.zeros(4)}
            return self.batch

    collector = Collector()
    batch, games, metrics = training.collect_cohort(
        collector=collector,
        args=types.SimpleNamespace(games=1),
        needed=needed,
        seed=12,
        prior_games=0,
        prior_rows=0,
        snapshot_exposures=123,
    )
    assert games == needed // 4
    assert len(batch["chosen"]) == needed
    if games == 1:
        assert batch is collector.batch
    assert metrics["collection_seed_start"] == 12
    assert metrics["collection_seed_end"] == 12 + games
    assert metrics["collection_snapshot_exposures"] == 123
    assert metrics["collection_win_pct"] == 25
    assert metrics["collection_dealin_pct"] == 12.5
    assert metrics["collection_tsumo_pct"] == 50
    assert metrics["collection_mean_win_points"] == 6000
    assert metrics["collection_mean_inference_batch"] == 2
    assert "collection_mean_riichi_turn" not in metrics


def test_native_self_play_records_legal_sampled_actions(monkeypatch):
    game_logs = []
    continuation = self_play.outcomes.hand_continuation_states

    def record_log(events):
        game_logs.append(events)
        return continuation(events)

    monkeypatch.setattr(
        self_play.outcomes, "hand_continuation_states", record_log
    )

    class Actor:
        def __init__(self):
            self.scale = tf.Variable(0.0)
            self.traces = 0

        def __call__(self, inputs, training=False):
            self.traces += 1
            count = tf.shape(inputs["decision_phase"])[0]
            return {
                **{
                    name: tf.zeros((count, actions.ACTOR_OUTPUTS[name][0]))
                    + self.scale
                    * tf.range(
                        actions.ACTOR_OUTPUTS[name][0],
                        dtype=tf.float32,
                    )
                    for name in actions.TRAIN_POLICY_HEADS
                },
                "board": tf.zeros((count, 1)),
            }

    tf.keras.utils.set_random_seed(12)
    actor = Actor()
    predict = self_play.predictor(actor)
    collector = self_play.Collector(predict, 1)
    batch, statistics = self_play.collect(collector.predict, 1, 12)
    tf.keras.utils.set_random_seed(12)
    tf.nest.map_structure(
        numpy.testing.assert_array_equal,
        (batch, statistics),
        self_play.collect(collector.predict, 1, 12),
    )
    assert statistics["games"] == 1
    assert statistics["hands"] >= 1
    for name, event_type in (
        ("hands", "start_kyoku"),
        ("wins", "hora"),
        ("draws", "ryukyoku"),
        ("games", "end_game"),
        ("riichi_hands", "reach_accepted"),
    ):
        assert statistics[name] == sum(
            event["type"] == event_type for event in game_logs[0]
        )
    assert statistics["win_points"] == sum(
        event["deltas"][event["actor"]]
        for event in game_logs[0]
        if event["type"] == "hora"
    )
    assert len(batch["chosen"]) > 100
    probability = numpy.exp(batch["behavior"])
    numpy.testing.assert_allclose(probability.sum(-1), 1, atol=1e-6)
    assert numpy.all(probability[~batch["legal"]] == 0)
    assert numpy.all(
        batch["legal"][numpy.arange(len(batch["chosen"])), batch["chosen"]]
    )
    assert len(numpy.unique(batch["settlement"], axis=0)) > 1
    assert "advantages" not in batch
    assert "value" not in batch
    current = actor_layers.distribution(
        Actor()(batch["features"]), batch["head"], batch["legal"]
    )
    numpy.testing.assert_allclose(current, batch["behavior"], atol=1e-6)
    actor.scale.assign(1.0)
    for count in (1, 2):
        outputs = predict(
            {name: value[:count] for name, value in batch["features"].items()}
        )
        numpy.testing.assert_array_equal(
            outputs["discard_policy_logits"][:, 1], numpy.ones(count)
        )
    assert actor.traces == 1


def test_fresh_critic_targets_preserve_deposits_and_censor_preempted_calls(
    monkeypatch,
):
    from log_dataset import _native_collection
    from model import outcomes

    events = [
        {
            "type": "start_kyoku",
            "scores": [25000] * 4,
            "oya": 0,
            "kyoku": 1,
            "bakaze": "E",
            "honba": 1,
            "kyotaku": 1,
        },
        {"type": "reach", "actor": 0},
        {"type": "reach_accepted", "actor": 0},
        {"type": "dahai", "actor": 0},
        {"type": "pon", "actor": 2},
        {"type": "dahai", "actor": 2},
        {"type": "reach", "actor": 3},
        {"type": "reach_accepted", "actor": 3},
        {"type": "dahai", "actor": 3},
        {
            "type": "hora",
            "actor": 1,
            "target": 3,
            "deltas": [0, 11300, 0, -8300],
        },
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    features = numpy.zeros(4, self_play.FEATURE_DTYPE)
    features["riichi_state"][1, 3] = 2  # Absolute seat 0 already paid.
    features["riichi_state"][2, 2] = 2
    features["riichi_state"][3, 1] = 2
    features["riichi_state"][3, 0] = 1  # Declaration is not acceptance.
    legal = numpy.zeros((4, 150), bool)
    legal[numpy.arange(4), [37, 112, 112, 0]] = True

    class Collector:
        submitted = False

        def __init__(self, games, seed):
            pass

        def next(self):
            if self.submitted:
                return None
            return {
                "features": features.tobytes(),
                "head": [1, 2, 2, 0],
                "game": [0] * 4,
                "hand": [0] * 4,
                "seat": [0, 1, 2, 3],
                "event": [1, 4, 4, 7],
                "legal": legal.tobytes(),
            }

        def submit(self, chosen):
            assert chosen == [37, 112, 112, 0]
            self.submitted = True

        def finish(self):
            return [json.dumps(events)], {"games": 1}

    monkeypatch.setattr(_native_collection, "NativeCollector", Collector)
    batch, _ = self_play.collect(
        lambda features: {
            name: numpy.zeros((4, actions.ACTOR_OUTPUTS[name][0]))
            for name in actions.TRAIN_POLICY_HEADS
        },
        1,
        12,
    )
    numpy.testing.assert_array_equal(batch["critic_valid"], [1, 0, 1, 1])
    # Honba is removed; carried pots and only future deposits remain.
    expected = numpy.zeros((4, 4, 4), numpy.int32)
    expected[0, 3, 1] = 8000
    expected[3] = numpy.diag([-1000, 3000, 0, -1000])
    numpy.testing.assert_array_equal(
        batch["payment_points"][0], expected[outcomes.PAYMENT_CELLS]
    )
    expected[3, 0, 0] = 0
    for seat in (1, 2, 3):
        order = (seat + numpy.arange(4)) % 4
        numpy.testing.assert_array_equal(
            batch["payment_points"][seat],
            expected[:, order][:, :, order][outcomes.PAYMENT_CELLS],
        )
    numpy.testing.assert_array_equal(
        batch["settlement"][0, :4], [24000, 36300, 25000, 15700]
    )


@pytest.mark.parametrize("awr", [False, True])
def test_phases_update_actor_and_preserve_stored_behavior(
    monkeypatch, tmp_path, awr
):
    class Payment(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.value = tf.keras.layers.Dense(1)

        def supervised_statistics(self, inputs, labels, valid):
            prediction = self.value(inputs["state"])[:, 0]
            return tf.stack(
                [
                    losses.masked_totals(tf.square(prediction - 1), valid),
                    tf.zeros(2),
                    tf.zeros(2),
                    losses.masked_totals(tf.zeros_like(prediction), valid),
                    losses.masked_totals(tf.ones_like(prediction), valid),
                ]
            )

    class Network(tf.keras.Model):
        def __init__(self):
            super().__init__()
            inputs = tf.keras.Input((1,), name="x")
            board = tf.keras.layers.Dense(2)(inputs)
            logits = tf.keras.layers.Dense(150)(board)
            self.critic = Payment()
            self.actor = tf.keras.Model(
                {"x": inputs},
                {
                    "board": board,
                    **{
                        name + "_prior_loss": logits[:, 0] * 0
                        for name in actions.TRAIN_POLICY_HEADS
                    },
                    **{name: logits for name in actions.TRAIN_POLICY_HEADS},
                },
            )

        def candidate_context(self, features, outputs):
            return outputs, outputs["board"]

        def chosen_inputs(self, features, outputs, state, chosen):
            return {"state": state, "pot": tf.zeros(tf.shape(state)[0])}

        def call(self, inputs):
            outputs = self.actor(inputs[0])
            self.critic.value(outputs["board"])
            return outputs

    monkeypatch.setattr(
        training.critic,
        "chosen_actions",
        lambda features, labels: labels["discard_policy_logits"],
    )
    monkeypatch.setattr(
        training.critic,
        "advantages",
        lambda *args: (
            tf.zeros(2) if awr else pytest.fail("CE-only phase scored AWR")
        ),
    )
    monkeypatch.setattr(
        training,
        "objective_metrics",
        lambda inputs, outputs, advantages: (
            -tf.reduce_mean(outputs["discard_policy_logits"][:, 0]),
            {
                "policy_entropy": tf.constant(0.5),
                "policy_prior_loss": tf.constant(0.0),
                **{
                    name.removesuffix("_logits")
                    + "_"
                    + suffix: tf.constant(0.0)
                    for name in actions.TRAIN_POLICY_HEADS
                    for suffix in ("count", "correct", "top3_correct")
                },
            },
        ),
    )
    network = Network()
    inputs = ({"x": tf.ones((2, 1))}, {}, {})
    network(inputs)
    monkeypatch.setattr(
        training.beliefs,
        "belief_loss_terms",
        lambda features, labels, outputs, mean=losses.mean_valid: (
            {
                "belief": mean(
                    tf.square(outputs["board"]), tf.ones_like(outputs["board"])
                )
            },
            {},
        ),
    )
    trainer = training.Trainer(
        network=network,
        first=inputs,
        args=types.SimpleNamespace(
            ppo_learning_rate=0.01,
            supervised_learning_rate=2e-4,
            awr_only_exposures=0,
            final_awr_exposures=0,
            final_awr_kl=0.1,
            prior_weight=1.0,
            foundation_exposures=400,
            awr_exposures=600,
            block_exposures=100,
            batch_size=2,
            clip=0.2,
            dual_clip=3.0,
            entropy=0.0,
            awr_entropy=0.01,
            awr_temperature=3.0,
        ),
    )
    assert type(trainer.auxiliary_optimizer) is tf.keras.optimizers.AdamW
    assert type(trainer.policy_optimizer) is tf.keras.optimizers.AdamW
    if awr:
        before = trainer.auxiliary_update.python_function(
            trainer.packing.pack(inputs), True, train=False
        )["auxiliary"]
        trainer.args.awr_entropy = 0.03
        after = trainer.auxiliary_update.python_function(
            trainer.packing.pack(inputs), True, train=False
        )["auxiliary"]
        numpy.testing.assert_allclose(after - before, -0.01, atol=1e-6)
        trainer.args.awr_entropy = 0.01
    for optimizer in (trainer.auxiliary_optimizer, trainer.policy_optimizer):
        assert optimizer.weight_decay == 1e-4
        assert optimizer.clipnorm == 1.0
    schedule = training.supervised_schedule(trainer.args)
    numpy.testing.assert_allclose(
        [schedule(0), schedule(10), schedule(375), schedule(500)],
        [2e-5, 2e-4, 2e-4, 2e-5],
        rtol=1e-6,
    )
    batch = dict(
        features=trainer.feature_packing.pack(inputs[0]),
        head=tf.constant([0, 0]),
        legal=tf.ones((2, 150), tf.bool),
        chosen=tf.constant([0, 1]),
        payment_points=tf.zeros((2, 40), tf.int32),
        critic_valid=tf.ones(2, tf.bool),
        returns=tf.constant([1.0, -1.0]),
        advantages=tf.zeros(2),
    )
    batch["behavior"] = trainer.predict(batch)
    before_belief = network.actor.get_weights()
    metrics = trainer.policy_update(
        batch, trainer.packing.pack(inputs), tf.constant(2)
    )
    assert all(numpy.isfinite(float(v)) for v in metrics.values())
    assert any(
        numpy.any(old != new)
        for old, new in zip(before_belief, network.actor.get_weights())
    )  # Beliefs and fresh settlement train even with zero PPO advantages.
    batch["reference"] = trainer.predict(batch)
    snapshot = batch["reference"].numpy().copy()
    before = [v.numpy().copy() for v in network.trainable_variables]
    trainer.auxiliary_update(trainer.packing.pack(inputs), awr)
    assert any(
        numpy.any(old != variable.numpy())
        for old, variable in zip(before, network.trainable_variables)
    )
    numpy.testing.assert_array_equal(batch["reference"], snapshot)
    assert numpy.any(
        batch["reference"].numpy() != trainer.predict(batch).numpy()
    )

    checkpoint = tf.train.Checkpoint(
        network=network,
        policy_optimizer=trainer.policy_optimizer,
        auxiliary_optimizer=trainer.auxiliary_optimizer,
    )
    saved = checkpoint.save(str(tmp_path / "cycle"))
    expected = trainer.predict(batch).numpy()
    trainer.policy_update(batch, trainer.packing.pack(inputs), tf.constant(2))
    trainer.auxiliary_update(trainer.packing.pack(inputs), awr)
    checkpoint.restore(saved).assert_consumed()
    numpy.testing.assert_allclose(trainer.predict(batch), expected)
    assert int(trainer.policy_optimizer.iterations) == 1
    assert int(trainer.auxiliary_optimizer.iterations) == 1

    metrics = trainer.policy_update(
        batch, trainer.packing.pack(inputs), tf.constant(1)
    )
    assert all(numpy.isfinite(float(v)) for v in metrics.values())
    expected = network.actor.get_weights()
    traces = trainer.policy_update.experimental_get_tracing_count()
    checkpoint.restore(saved).assert_consumed()
    padded = tf.nest.map_structure(
        lambda value: (
            tf.concat([value[:1], value[1:] * 0], axis=0)
            if value.dtype != tf.bool
            else value
        ),
        batch,
    )
    padded_inputs = ({"x": tf.constant([[1.0], [1000.0]])}, {}, {})
    trainer.policy_update(
        padded, trainer.packing.pack(padded_inputs), tf.constant(1)
    )
    for left, right in zip(expected, network.actor.get_weights()):
        numpy.testing.assert_allclose(left, right, atol=1e-7)
    assert trainer.policy_update.experimental_get_tracing_count() == traces


def test_real_joint_network_supports_both_training_phases(
    monkeypatch, tmp_path
):
    import json
    import pathlib

    from log_dataset import afk, replay, scoring, training_data

    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    targets = scoring.round_labels(events)
    observations = list(
        replay.Replay(
            "training-probe",
            afk.retained_hands(events)[0],
        ).observations()
    )[:2]
    inputs = training_data.parse_batch(
        tf.constant(
            [
                training_data.serialize(row, targets[row.kyoku_id])
                for row in observations
            ]
        )
    )
    network = joint_network.JointPolicyPayment()
    outputs = network(inputs)
    assert [
        layer.name
        for layer in network.actor.layers
        if layer.name.startswith("board_block_") and layer.name.endswith("_add")
    ] == [f"board_block_{block}_add" for block in range(4)]
    network.compile_training()
    trainer = training.Trainer(
        network=network,
        first=inputs,
        args=types.SimpleNamespace(
            ppo_learning_rate=1e-5,
            supervised_learning_rate=2e-4,
            awr_only_exposures=0,
            final_awr_exposures=8,
            final_awr_kl=0.1,
            prior_weight=1.0,
            foundation_exposures=400,
            awr_exposures=600,
            block_exposures=100,
            batch_size=2,
            clip=0.2,
            dual_clip=3.0,
            entropy=0.001,
            awr_entropy=0.01,
            awr_temperature=3.0,
        ),
    )
    heads = tf.stack(
        [inputs[1][name] >= 0 for name in actions.TRAIN_POLICY_HEADS],
        axis=1,
    )
    head = tf.argmax(tf.cast(heads, tf.int32), axis=1, output_type=tf.int32)
    chosen = tf.reduce_max(
        tf.stack(
            [
                tf.cast(inputs[1][name], tf.int32)
                for name in actions.TRAIN_POLICY_HEADS
            ],
            axis=1,
        ),
        axis=1,
    )
    legal = tf.gather(
        tf.stack(
            [
                tf.pad(
                    outputs[name] > -1e3,
                    [[0, 0], [0, 150 - outputs[name].shape[-1]]],
                )
                for name in actions.TRAIN_POLICY_HEADS
            ],
            axis=1,
        ),
        head,
        batch_dims=1,
    )
    batch = dict(
        features=trainer.feature_packing.pack(inputs[0]),
        head=head,
        legal=legal,
        chosen=chosen,
        payment_points=tf.gather(
            training.joint_settlement.payment_values(inputs[2]),
            inputs[2]["payment"],
            axis=1,
            batch_dims=1,
        ),
        critic_valid=tf.ones(2, tf.bool),
        returns=tf.constant([1.0, -1.0]),
        advantages=tf.constant([1.0, -1.0]),
    )
    batch["behavior"] = trainer.predict(batch)
    shared_before = (
        network.actor.get_layer("board_block_3_contract").kernel.numpy().copy()
    )
    before = network.actor.get_layer("final_pass").kernel.numpy().copy()
    critic_before = network.critic.event.kernel.numpy().copy()
    policy = trainer.policy_update(
        batch, trainer.packing.pack(inputs), tf.constant(2)
    )
    assert all(numpy.isfinite(float(v)) for v in policy.values())
    assert numpy.any(
        before != network.actor.get_layer("final_pass").kernel.numpy()
    )
    assert numpy.any(
        shared_before
        != network.actor.get_layer("board_block_3_contract").kernel.numpy()
    )
    assert float(policy["payment_nll"]) > 0
    assert float(policy["settlement_supported_fraction"]) == 1
    assert numpy.any(critic_before != network.critic.event.kernel.numpy())
    shared_before = (
        network.actor.get_layer("board_block_3_contract").kernel.numpy().copy()
    )
    batch["reference"] = trainer.predict(batch)
    auxiliary = trainer.auxiliary_update(trainer.packing.pack(inputs), True)
    assert numpy.any(
        shared_before
        != network.actor.get_layer("board_block_3_contract").kernel.numpy()
    )
    assert all(
        numpy.isfinite(float(v)) for v in {**policy, **auxiliary}.values()
    )

    (tmp_path / "dataset_summary.json").write_text(
        json.dumps({"validation_examples": 3})
    )
    (tmp_path / "status.json").write_text('{"stage": "validation", "awr": 0}')

    def validation_data(**kwargs):
        assert kwargs["sampling"] == focus_sampling.Sampling.NATURAL
        assert kwargs["partition"] == "validation"
        assert kwargs["repeat"] is False
        return tf.data.Dataset.from_tensor_slices(
            tf.constant(
                [
                    training_data.serialize(row, targets[row.kyoku_id])
                    for row in [*observations, observations[0]]
                ]
            )
        ).batch(kwargs["batch_size"])

    monkeypatch.setattr(training_data, "dataset", validation_data)
    distributed = training.DistributedTrainer(
        trainer, tf.distribute.get_strategy()
    )
    writer = tf.summary.create_file_writer(str(tmp_path / "tensorboard"))
    training.validate(
        args=types.SimpleNamespace(
            output=str(tmp_path), data=str(tmp_path), batch_size=2
        ),
        network=network,
        strategy=tf.distribute.get_strategy(),
        evaluate=distributed.evaluate,
        writer=writer,
        step=400,
    )
    writer.close()
    validation = json.loads((tmp_path / "validation-400.json").read_text())
    assert validation["examples"] == 3
    assert 0 <= validation["imitation_accuracy"] <= 1
    assert (
        validation["imitation_top3_accuracy"]
        >= validation["imitation_accuracy"]
    )
    assert json.loads((tmp_path / "status.json").read_text())["awr"] == 0

    baseline = trainer.baseline(trainer.feature_packing.pack(inputs[0]))
    assert baseline.shape == (2,)
    assert numpy.isfinite(baseline).all()
    # PPO keeps the state baseline; AWR cannot rank conditional call values.
    from model import critic

    @tf.function
    def shared_values():
        outputs = network(inputs)
        selected, values, expected = critic.action_values(
            network, inputs[0], outputs, outputs["critic_candidate_state"]
        )
        chosen = critic.chosen_actions(inputs[0], inputs[1])
        recorded = tf.gather(chosen, selected[:, 0]) == selected[:, 1]
        return (
            expected,
            (tf.boolean_mask(values, recorded) - expected) / 3.0,
            critic.advantages(network, inputs, outputs, 3.0),
        )

    expected, advantage, awr_advantage = shared_values()
    numpy.testing.assert_allclose(baseline * 100, expected, atol=1e-4)
    numpy.testing.assert_allclose(
        awr_advantage,
        tf.where(inputs[0]["decision_phase"] == 1, 0.0, advantage),
        atol=1e-4,
    )

    encoded = network.actor(inputs[0])
    context = network.context(
        [inputs[0], encoded["selected"], encoded["structure"]]
    )
    assert context.shape[1] == 4 + encoded["structure"].shape[1] - 1
    numpy.testing.assert_array_equal(
        context,
        network.context(
            [
                inputs[0],
                encoded["selected"],
                tf.concat(
                    [
                        encoded["structure"][:, :-1],
                        tf.zeros_like(encoded["structure"][:, -1:]),
                    ],
                    axis=1,
                ),
            ]
        ),
    )
    distributed.anchor.set_weights(network.actor.get_weights())
    distributed.anchor_ready.assign(True)
    anchor_weights = distributed.anchor.get_weights()
    for _ in range(2):
        anchored = distributed.auxiliary_update(
            trainer.packing.pack(inputs), True, anchored=True
        )
        assert numpy.isfinite(float(anchored["auxiliary"]))
        numpy.testing.assert_allclose(
            anchored["weighted_anchor_loss"],
            0.1 * anchored["anchor_kl"],
            atol=1e-7,
        )
    for before, after in zip(anchor_weights, distributed.anchor.get_weights()):
        numpy.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("rows, remaining", [(20, 5), (3, 7)])
def test_ppo_scores_only_uniformly_selected_eligible_decisions(
    rows, remaining, monkeypatch
):
    monkeypatch.setattr(
        training.joint_settlement,
        "post_hand_forecast",
        lambda **kwargs: (kwargs["scores"][:, 0], None),
    )
    scored, trained = [], []
    selected_count = min(rows, remaining)

    class Trainer:
        feature_packing = training_data.PackedBatch({"row": tf.zeros(2)})

        def baseline(self, features):
            assert features["float32"].shape == (2, 1)
            scored.extend(features["float32"][:, 0].tolist())
            return tf.zeros(2)

        def policy_update(self, batch, inputs, count):
            assert len(scored) == 2 * ((selected_count + 1) // 2)
            numpy.testing.assert_array_equal(
                batch["returns"], batch["features"]["float32"][:, 0] * 2
            )
            numpy.testing.assert_allclose(
                batch["advantages"],
                (batch["returns"] - numpy.mean(scored[:selected_count]) * 2)
                / (numpy.std(scored[:selected_count]) * 2),
                atol=1e-6,
            )
            assert len(inputs) == len(batch["chosen"])
            trained.extend(batch["features"]["float32"][: int(count), 0])
            return {
                "belief_loss": tf.constant(1.0),
            }

        def predict(self, batch):
            # Diagnose before training and after each (possibly partial) epoch.
            assert len(trained) in (0, remaining) or (
                len(trained) % selected_count == 0
            )
            return tf.math.log(tf.tile([[0.99, 0.01]], [2, 1]))

    tf.keras.utils.set_random_seed(1941)
    batch = {
        "features": {"row": numpy.arange(rows, dtype=numpy.float32)},
        "chosen": numpy.zeros(rows, numpy.int32),
        "settlement": numpy.pad(
            (numpy.arange(rows, dtype=numpy.int32) * 200)[:, None],
            ((0, 0), (0, 11)),
        ),
        "behavior": numpy.full((rows, 2), -numpy.log(2), numpy.float32),
    }
    used, metrics = training.ppo_block(
        trainer=Trainer(),
        args=types.SimpleNamespace(batch_size=2, ppo_epochs=3, seed=941),
        stream=iter([numpy.zeros(2)] * remaining),
        batch=batch,
        remaining=remaining,
    )
    assert metrics["update_positions_per_second"] > 0
    assert metrics["scoring_seconds"] >= 0
    assert metrics["diagnostic_seconds"] >= 0
    assert metrics["behavior_kl"] > 0.02
    assert metrics["behavior_kl_before"] == metrics["behavior_kl"]
    assert metrics[f"behavior_kl_epoch_{metrics['ppo_epochs']}"] == (
        metrics["behavior_kl"]
    )
    assert used == metrics["trained_exposures"] == remaining
    assert metrics["scored_decisions"] == selected_count
    assert metrics["collected_decisions"] == len(batch["chosen"]) == rows
    assert len(set(scored)) == selected_count
    assert sorted(scored[:selected_count]) == sorted(set(trained))


@pytest.mark.parametrize("collection_size", [2, 5])
@pytest.mark.parametrize(
    "awr_budget, block, ratio, foundation",
    [(28, 16, 0.25, 0), (31, 17, 0.25, 5), (7, 4, 1.0, 5)],
)
def test_schedule_preserves_phase_budgets_anchor_and_explicit_resume(
    monkeypatch, tmp_path, collection_size, awr_budget, block, ratio, foundation
):
    import json

    class Network(tf.keras.Model):
        def __init__(self):
            super().__init__()
            inputs = tf.keras.Input((1,), name="x")
            self.actor = tf.keras.Model(
                {"x": inputs}, tf.keras.layers.Dense(1)(inputs)
            )

        def call(self, inputs):
            return self.actor(inputs[0])

        def compile_training(self):
            pass

    class Trainer(CpuStaging):
        def __init__(self, *, network, args, first):
            self.packing = training_data.PackedBatch(first)
            self.feature_packing = training_data.PackedBatch(first[0])
            self.network = network
            self.anchor = tf.keras.models.clone_model(network.actor)
            self.anchor.trainable = False
            self.anchor_ready = tf.Variable(False, trainable=False)
            # MirroredVariable truthiness is not its scalar value.
            monkeypatch.setattr(
                type(self.anchor_ready), "__bool__", lambda value: True
            )
            self.policy_optimizer = tf.keras.optimizers.Adam(0.01)
            self.auxiliary_optimizer = tf.keras.optimizers.Adam(0.01)
            self.policy_optimizer.build(network.trainable_variables)
            self.auxiliary_optimizer.build(network.trainable_variables)

        def evaluate(self, batch, statistics=None):
            raise AssertionError("Validation stub owns the toy evaluator")

        def predict(self, batch):
            # High drift remains diagnostic and does not stop the cohort.
            return tf.nn.log_softmax(tf.one_hot(batch["chosen"], 150) * 3)

        def baseline(self, features):
            scored.append(len(features["float32"]))
            return tf.zeros(len(features["float32"]))

        def policy_update(self, batch, inputs, count):
            calls.append(("ppo", int(count)))
            assert len(inputs["float32"]) == len(batch["chosen"]) == 3
            self.policy_optimizer.apply_gradients(
                (tf.ones_like(v), v) for v in self.network.trainable_variables
            )
            return {
                "belief_loss": tf.constant(1.0),
            }

        def auxiliary_update(self, inputs, awr, train=True, anchored=False):
            if not train:
                assert not awr
                assert int(self.auxiliary_optimizer.iterations) == 0
                return {"belief_loss": tf.constant(2.0)}
            if anchored:
                assert awr and bool(self.anchor_ready.numpy())
                middle = Network()
                middle(({"x": tf.ones((1, 1))}, {}, {}))
                middle.load_weights(
                    pathlib.Path(args.output) / "middle.weights.h5"
                )
                for expected, actual in zip(
                    middle.actor.get_weights(), self.anchor.get_weights()
                ):
                    numpy.testing.assert_array_equal(actual, expected)
            calls.append(
                (
                    {
                        (False, False): "foundation",
                        (True, False): "awr",
                        (True, True): "final_awr",
                    }[awr, anchored],
                    len(inputs["float32"]),
                )
            )
            self.auxiliary_optimizer.apply_gradients(
                (tf.ones_like(v), v) for v in self.network.trainable_variables
            )
            return {
                "belief_loss": tf.constant(1.0),
                "policy_loss": tf.constant(0.5),
                **(
                    {"utility_effective_sample_fraction": tf.constant(0.8)}
                    if awr
                    else {}
                ),
            }

    def collect(actor, games, seed):
        if seed == 100944 and not interrupted:
            interrupted.append(True)
            raise RuntimeError("interrupted between fresh cohorts")
        return {
            "features": {"x": numpy.ones((collection_size, 1), numpy.float32)},
            "chosen": numpy.zeros(collection_size, numpy.int32),
            "head": numpy.zeros(collection_size, numpy.int32),
            "legal": numpy.ones((collection_size, 150), bool),
            "behavior": numpy.full(
                (collection_size, 150), -numpy.log(150), numpy.float32
            ),
            "settlement": numpy.pad(
                numpy.arange(collection_size, dtype=numpy.int32)[:, None],
                ((0, 0), (0, 11)),
            ),
        }, {**self_play.collection_counts(), "games": games}

    def validate(**kwargs):
        assert (
            sum(n for stage, n in calls if stage == "foundation") == foundation
        )
        assert sum(n for stage, n in calls if stage == "awr") == (
            args.awr_exposures if kwargs["step"] > foundation else 0
        )
        validations.append(kwargs["step"])

    calls, interrupted, scored, validations = [], [], [], []
    monkeypatch.setattr(training, "validate", validate)
    # Exercise scheduling with the toy CPU network while production fails
    # immediately if TensorFlow cannot see the selected training GPU.
    monkeypatch.setattr(
        training.tf.config, "list_physical_devices", lambda kind: ["test-gpu"]
    )
    monkeypatch.setattr(
        training.tf.config.experimental, "set_memory_growth", lambda *args: None
    )
    monkeypatch.setattr(training.joint_network, "JointPolicyPayment", Network)
    monkeypatch.setattr(training, "Trainer", Trainer)
    monkeypatch.setattr(
        training, "DistributedTrainer", lambda trainer, strategy: trainer
    )
    monkeypatch.setattr(
        training.tf.distribute,
        "MirroredStrategy",
        training.tf.distribute.get_strategy,
    )
    monkeypatch.setattr(training.self_play, "collect", collect)
    monkeypatch.setattr(
        training.training_data,
        "dataset",
        lambda **kwargs: tf.data.Dataset.range(30)
        .repeat()
        .batch(kwargs["batch_size"]),
    )
    monkeypatch.setattr(
        training.training_data,
        "parse_batch",
        lambda raw: ({"x": tf.cast(raw[:, None], tf.float32)}, {}, {}),
    )
    assert training.argument_parser().get_default("ppo_exposure_ratio") == 1.0
    args = types.SimpleNamespace(
        data="fixture",
        initial=None,
        resume=False,
        cohort_exposures=1572864,
        output=str(tmp_path),
        seed=941,
        foundation_exposures=foundation,
        awr_exposures=awr_budget,
        block_exposures=block,
        cycle_blocks=1,
        games=3,
        collection_workers=1,
        batch_size=3,
        ppo_epochs=1,
        ppo_exposure_ratio=ratio,
        clip=0.2,
        dual_clip=3.0,
        awr_temperature=3.0,
        ppo_learning_rate=1e-5,
        supervised_learning_rate=2e-4,
        awr_only_exposures=0,
        final_awr_exposures=block + 1,
        final_awr_kl=0.1,
        prior_weight=1.0,
        entropy=0.0,
        awr_entropy=0.01,
        validation_data=None,
    )
    with monkeypatch.context() as ctx:
        ctx.setattr(
            training.tf.config, "list_physical_devices", lambda kind: []
        )
        with pytest.raises(ValueError, match="requires visible GPUs"):
            training.run(args)
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="interrupted"):
        training.run(args)
    before = list(calls)
    with pytest.raises(ValueError, match="Existing run requires --resume"):
        training.run(args)
    assert calls == before
    args.resume = True
    training.run(args)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["stage"] == "complete"
    assert status["final_awr"] == args.final_awr_exposures
    assert (status["foundation"], status["awr"], status["ppo"]) == (
        foundation,
        awr_budget,
        7,
    )
    import tensorboard.backend.event_processing.event_accumulator as events

    log = events.EventAccumulator(str(tmp_path / "tensorboard"))
    log.Reload()
    tags = log.Tags()["tensors"]
    assert "custom_scalars__config__" in tags
    assert "progress/awr_fraction" in tags
    assert "runtime/window_positions_per_second" in tags
    assert not any(tag.endswith("/belief_loss") for tag in tags)
    assert "final_awr/policy_loss" in tags
    assert "awr/utility_effective_sample_fraction" in tags
    assert "final_awr/utility_effective_sample_fraction" in tags
    assert all(
        float(tf.make_ndarray(event.tensor_proto)) > 0
        for event in log.Tensors("runtime/window_positions_per_second")
    )
    import re

    from tensorboard.plugins.custom_scalar import layout_pb2

    layout = layout_pb2.Layout.FromString(
        tf.make_ndarray(
            log.Tensors("custom_scalars__config__")[0].tensor_proto
        ).item()
    )
    patterns = [
        pattern
        for category in layout.category
        for chart in category.chart
        for pattern in chart.multiline.tag
    ]
    for stage in ("awr_only", "awr", "final_awr"):
        assert any(
            re.fullmatch(pattern, stage + "/utility_effective_sample_fraction")
            for pattern in patterns
        )
    assert calls[0][0] == ("foundation" if foundation else "awr")
    assert validations == (
        ([foundation] if foundation else [])
        + [foundation + awr_budget + 7 + args.final_awr_exposures]
    )
    assert (tmp_path / "foundation.weights.h5").exists() == bool(foundation)
    assert status["elapsed_seconds"] > 0
    assert status["total_exposures"] == status["budget_exposures"]
    history = [
        json.loads(row)
        for row in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    initial = [row for row in history if row["stage"] == "initial"]
    assert len(initial) == 1
    assert initial[0]["total_exposures"] == 0
    assert initial[0]["belief_loss"] == 2.0
    assert any(row["stage"] == "cycle_complete" for row in history)
    assert status["cycle"] == 2
    assert status["cohort"] == 2
    assert status["collected"] == {2: 8, 5: 10}[collection_size]
    assert status["games"] == {2: 11, 5: 5}[collection_size]
    assert sum(scored) == 9  # Seven eligible rows plus two padding rows.
    for stage, expected in (
        ("foundation", foundation),
        ("awr", awr_budget),
        ("ppo", 7),
        ("final_awr", args.final_awr_exposures),
    ):
        assert sum(count for name, count in calls if name == stage) == expected
    awr_count = ppo_count = 0
    for stage, count in calls:
        if stage == "awr":
            awr_count += count
        elif stage == "ppo":
            ppo_count += count
            assert awr_count in (block, awr_budget)
            assert ppo_count <= int(awr_count * ratio)
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert (plan["awr_exposures"], plan["ppo_exposures"]) == (awr_budget, 7)
    assert plan["ppo_exposure_ratio"] == ratio
    assert plan["final_awr_kl"] == 0.1
    assert "target_kl" not in plan
    # Restart weights are the same model as the completed run, without slots.
    latest = Network()
    final = Network()
    inputs = ({"x": tf.ones((1, 1))}, {}, {})
    latest(inputs)
    final(inputs)
    latest.load_weights(tmp_path / "latest.weights.h5")
    final.load_weights(tmp_path / "final.weights.h5")
    numpy.testing.assert_array_equal(latest(inputs), final(inputs))
    assert (tmp_path / "middle.weights.h5").exists()
    ppo_checkpoint = tf.train.latest_checkpoint(str(tmp_path / "ppo-budget"))
    assert ppo_checkpoint
    saved = tf.train.load_checkpoint(ppo_checkpoint)
    assert saved.get_tensor("ppo/.ATTRIBUTES/VARIABLE_VALUE") == 7

    before = list(calls)
    training.run(args)
    assert calls == before
    args.games = 128
    args.cycle_blocks = 2
    training.run(args)
    assert calls == before
    changes = [
        json.loads(row)
        for row in (tmp_path / "plan-changes.jsonl").read_text().splitlines()
    ]
    assert len(changes) == 1
    assert changes[0]["previous"] == plan
    assert changes[0]["current"]["games"] == 128
    assert changes[0]["current"]["cycle_blocks"] == 2
    assert changes[0]["current"]["batch_size"] == 3
    resumed = json.loads((tmp_path / "status.json").read_text())
    for name in (
        "awr",
        "final_awr",
        "ppo",
        "foundation",
        "cycle",
        "games",
        "cohort",
    ):
        assert resumed[name] == status[name]
    args.batch_size *= 2
    with pytest.raises(ValueError, match="Changed settings"):
        training.run(args)

    # An offline-only pass finishes without constructing a game collector.
    args.batch_size = 3
    args.output = str(tmp_path / "awr-only")
    args.resume = False
    args.foundation_exposures = 0
    args.awr_only_exposures = awr_budget
    args.ppo_exposure_ratio = 0
    calls.clear()
    with monkeypatch.context() as ctx:

        def no_collector(*args, **kwargs):
            pytest.fail("AWR-only training must not start a collector")

        ctx.setattr(training.self_play, "Collector", no_collector)
        ctx.setattr(training, "validate", lambda **kwargs: {})
        training.run(args)
    status = json.loads((pathlib.Path(args.output) / "status.json").read_text())
    assert status["stage"] == "complete"
    assert status["awr"] == awr_budget
    assert all(status[name] == 0 for name in ("foundation", "ppo", "games"))
    assert sum(count for name, count in calls if name == "awr") == awr_budget
    assert (pathlib.Path(args.output) / "final.weights.h5").exists()
