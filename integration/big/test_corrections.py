"""One MMD objective learns across complete matches and root descendants."""

import json
import types

import numpy
import pytest
import tensorflow as tf

from log_dataset import training_data
from model import actions, critic
from model import network as joint_network
from training import self_play
from training import trainer as base_trainer
from training.corrections import (
    collection,
    distributed,
    matches,
    rollouts,
    run,
    trainer,
)


def test_mmd_has_the_closed_form_optimum_and_detached_value_targets():
    features = {
        "decision_phase": tf.constant([0, 0, 0]),
        "trigger_tile_id": tf.constant([0, 0, 0]),
    }
    values = tf.Variable([[0.0, 0.2], [0.2, 0.0], [0.0, 1000.0]])
    mirror = tf.Variable(tf.math.log([[0.4, 0.6]] * 3))
    magnet = tf.Variable(tf.math.log([[0.8, 0.2]] * 3))
    # Exact statewise optimum of -E[Q] + .3 KL(p||old) + .3 KL(p||magnet).
    logits = tf.Variable(
        tf.concat(
            [
                (values[:2] + 0.3 * mirror[:2] + 0.3 * magnet[:2]) / 0.6,
                tf.zeros([1, 2]),
            ],
            axis=0,
        )
    )
    with tf.GradientTape(persistent=True) as tape:
        metrics = trainer.mmd_terms(
            features=features,
            current=tf.pad(
                tf.nn.log_softmax(logits),
                [[0, 0], [0, 148]],
                constant_values=-1e9,
            ),
            mirror=tf.pad(mirror, [[0, 0], [0, 148]], constant_values=-1e9),
            magnet=tf.pad(magnet, [[0, 0], [0, 148]], constant_values=-1e9),
            values=tf.pad(values, [[0, 0], [0, 293]]),
            count=tf.constant(2),
        )
    numpy.testing.assert_allclose(
        tape.gradient(metrics["mmd_loss"], logits), 0, atol=1e-7
    )
    for reference in (values, mirror, magnet):
        assert tape.gradient(metrics["mmd_loss"], reference) is None


def test_mmd_learns_from_descendants_and_root_scoring_uses_live_weights():
    def predict(features):
        return {
            name: numpy.zeros(
                (len(features), actions.ACTOR_OUTPUTS[name][0]), numpy.float32
            )
            for name in actions.TRAIN_POLICY_HEADS
        }

    rows, _ = self_play.collect(predict, 1, 941)
    # These are ordinary decisions, with no forced-root flag or paired outcome.
    assert not rows["paired_root"].any()
    batch = tf.nest.map_structure(
        lambda value: value[:3],
        {
            name: rows[name]
            for name in (
                "features",
                "head",
                "legal",
                "chosen",
                "payment_points",
            )
        },
    )
    batch["chosen"] = critic.chosen_actions(
        batch["features"],
        {
            name: numpy.where(batch["head"] == head, batch["chosen"], -1)
            for head, name in enumerate(actions.TRAIN_POLICY_HEADS)
        },
    )
    network = joint_network.JointPolicyPayment()
    outputs, state = network.candidate_context(
        batch["features"], network.actor(batch["features"], training=False)
    )
    network.critic(
        network.chosen_inputs(
            batch["features"], outputs, state, batch["chosen"]
        )
    )
    fixed = {name: tf.identity(value) for name, value in outputs.items()}
    learner = trainer.MMDTrainer(
        network=network,
        mirror=lambda features, training: fixed,
        magnet=lambda **features: fixed,
        learning_rate=1e-5,
    )
    critic_learner = base_trainer.CriticTrainer(
        network, batch["features"], 1e-5
    )
    before = network.critic.get_weights()
    metrics = critic_learner.critic_update(
        {
            **batch,
            "features": critic_learner.feature_packing.pack_numpy(
                batch["features"]
            ),
        },
        tf.constant(3),
    )
    assert metrics["settlement_supported_fraction"] == 1
    assert metrics["gradient/global_norm"] > 0
    assert any(
        not numpy.array_equal(a, b)
        for a, b in zip(before, network.critic.get_weights())
    )
    before_actor = network.actor.get_weights()
    before_critic = network.critic.get_weights()
    metrics = learner.update(batch, tf.constant(3))
    assert all(numpy.isfinite(value).all() for value in metrics.values())
    assert metrics["gradient/global_norm"] > 0
    assert any(
        not numpy.array_equal(a, b)
        for a, b in zip(before_actor, network.actor.get_weights())
    )
    for before, after in zip(before_critic, network.critic.get_weights()):
        numpy.testing.assert_array_equal(before, after)
    numpy.testing.assert_allclose(
        metrics["joint_loss"],
        metrics["mmd_loss"] + 0.01 * metrics["constraint_loss"],
        rtol=1e-6,
    )

    # Root scoring keeps legal actions and reads updated critic weights.
    discard_features = {
        name: value[batch["head"] == 0]
        for name, value in batch["features"].items()
    }
    score = collection.root_scorer(network)
    values = score(discard_features)
    legal = discard_features["discard_action_legal"].astype(bool)
    assert numpy.isfinite(values[legal]).all()
    assert numpy.isneginf(values[~legal]).all()
    network.critic.event.kernel.assign(
        tf.zeros_like(network.critic.event.kernel)
    )
    updated = score(discard_features)
    assert not numpy.allclose(values[legal], updated[legal])


def test_rollout_shards_keep_every_packed_row_and_partial_batch(tmp_path):
    features = {
        "decision_phase": numpy.arange(5, dtype=numpy.int8),
        "public_value": numpy.arange(10, dtype=numpy.float32).reshape(5, 2),
    }
    packing = training_data.PackedBatch(features)
    paths = [tmp_path / "first.npz", tmp_path / "second.npz"]
    for path, rows in zip(paths, (slice(0, 3), slice(3, 5))):
        numpy.savez_compressed(
            path,
            **{
                "features/" + name: value
                for name, value in packing.pack_numpy(
                    {name: value[rows] for name, value in features.items()}
                ).items()
            },
            chosen=numpy.arange(5)[rows],
            paired_root=numpy.array([True, False, False, True, False])[rows],
        )
    recovered = list(rollouts.batches(paths, packing, 2, 941))
    assert sorted(len(batch["chosen"]) for batch in recovered) == [1, 2, 2]
    assert sorted(
        row for batch in recovered for row in batch["chosen"].numpy()
    ) == list(range(5))
    for batch in recovered:
        for name, values in features.items():
            numpy.testing.assert_array_equal(
                batch["features"][name], values[batch["chosen"]]
            )
        numpy.testing.assert_array_equal(
            batch["paired_root"],
            numpy.isin(batch["chosen"], [0, 3]),
        )


def test_launch_requires_complete_mortal_matches(tmp_path, monkeypatch):
    command = tmp_path / "evaluation.json"
    args = types.SimpleNamespace(
        rounds=2,
        root_games=10000,
        batch_size=64,
        workers=2,
        evaluation_command=str(command),
        sync_command=None,
        seed=941,
    )
    argv = [
        "eval",
        "--seed",
        "1",
        "--games",
        "10000",
        "--mortal-source",
        "mortal",
        "{weights}",
        "{output}",
    ]
    command.write_text(json.dumps(argv))
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind: [])
    assert run.validate_launch(args) == argv
    calls = []
    monkeypatch.setattr(
        run.subprocess, "run", lambda command, check: calls.append(command)
    )
    run.evaluate(argv, tmp_path / "policy.h5", tmp_path / "evaluation", 2)
    assert calls[0][2] == "5001"
    assert calls[0][-2:] == [
        str(tmp_path / "policy.h5"),
        str(tmp_path / "evaluation"),
    ]
    assert argv[2] == "1"

    for invalid, message in (
        ([*argv[:4], "4", *argv[5:]], "10,000"),
        ([*argv[:5], *argv[7:]], "Mortal source"),
        (argv[:-1], "needs {weights} and {output}"),
    ):
        command.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match=message):
            run.validate_launch(args)
    command.write_text(json.dumps(argv))
    args.root_games = 10001
    with pytest.raises(ValueError, match="cannot exceed"):
        run.validate_launch(args)
    args.root_games = 0
    with pytest.raises(ValueError, match="must be positive"):
        run.validate_launch(args)
    args.root_games = 10000
    monkeypatch.setattr(
        tf.config, "list_physical_devices", lambda kind: ["GPU-a", "GPU-b"]
    )
    with pytest.raises(ValueError, match="one learner GPU"):
        run.validate_launch(args)


def test_match_sources_preserve_all_seats_and_reject_incomplete_replays(
    tmp_path,
):
    manifest = {
        "format": "riichienv_standard_south",
        "mortal": True,
        "games": 10000,
    }
    (tmp_path / "evaluation.json").write_text(json.dumps(manifest))
    games = [
        {
            "replay": f"{game:05d}.jsonl.gz",
            "seed": game // 4,
            "new_seats": [game % 4],
        }
        for game in range(10000)
    ]
    records = tmp_path / "games.jsonl"
    records.write_text("\n".join(map(json.dumps, reversed(games))))
    sources = matches.sources(tmp_path)
    assert len(sources) == 10000
    assert [source["new_seats"] for source in sources[:4]] == [
        [0],
        [1],
        [2],
        [3],
    ]
    assert sources[0] == {
        "path": str(tmp_path / games[0]["replay"]),
        "seed": 0,
        "new_seats": [0],
    }
    assert sources[-1]["seed"] == 2499

    records.write_text("\n".join(map(json.dumps, games[:-1])))
    with pytest.raises(ValueError, match="all 10,000"):
        matches.sources(tmp_path)
    records.write_text("\n".join(map(json.dumps, [games[0], *games[:-1]])))
    with pytest.raises(ValueError, match="duplicate replay paths"):
        matches.sources(tmp_path)
    manifest["mortal"] = False
    (tmp_path / "evaluation.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="standard Mortal match"):
        matches.sources(tmp_path)


def test_each_phase_fits_all_outcomes_before_all_actor_updates(tmp_path):
    observed = []
    progress = {"actor_exposures": 0, "critic_exposures": 0}
    learner = types.SimpleNamespace(
        network=types.SimpleNamespace(
            actor=types.SimpleNamespace(
                get_weights=lambda: [1],
            )
        ),
        mirror=types.SimpleNamespace(
            set_weights=lambda weights: observed.append(("snapshot", weights)),
        ),
    )
    group = types.SimpleNamespace(
        chief=True,
        stopped=lambda root: (root / "STOP").exists(),
        update=lambda update, batch: observed.append(
            (update, batch["chosen"].numpy().tolist())
        )
        or {},
    )
    reports = []
    for phase, choices in (("match", [0, 1, 2]), ("roots", [3, 4, 5, 6, 7])):

        def batches():
            for begin in range(0, len(choices), 2):
                yield {
                    "features": {},
                    "chosen": tf.constant(choices[begin : begin + 2]),
                    "payment_points": tf.zeros(
                        (len(choices[begin : begin + 2]), 1)
                    ),
                }

        train_begin = len(observed)
        run.train_phase(
            phase=phase,
            batches=batches,
            learner=learner,
            group=group,
            root=tmp_path,
            progress=progress,
            report=lambda stage, metrics: reports.append(stage),
        )
        assert observed[train_begin] == ("snapshot", [1])
        steps = observed[train_begin + 1 :]
        assert [action for action, _ in steps] == (
            ["critic"] * ((len(choices) + 1) // 2)
            + ["actor"] * ((len(choices) + 1) // 2)
        )
        for action in ("critic", "actor"):
            assert [
                choice
                for name, rows in steps
                if name == action
                for choice in rows
            ] == choices
    assert progress == {"actor_exposures": 8, "critic_exposures": 8}
    assert reports.index("roots-critic") > max(
        index for index, stage in enumerate(reports) if stage == "match-actor"
    )

    # A stop after a completed critic batch must not begin the actor pass.
    observed.clear()
    run.train_phase(
        phase="roots",
        batches=batches,
        learner=learner,
        group=group,
        root=tmp_path,
        progress=progress,
        report=lambda stage, metrics: (tmp_path / "STOP").touch(),
    )
    assert observed == [("snapshot", [1]), ("critic", [3, 4])]
    assert progress == {"actor_exposures": 8, "critic_exposures": 10}


def test_stopping_collection_records_only_mined_games_and_keeps_checkpoint(
    tmp_path, monkeypatch
):
    initial = tmp_path / "initial.weights.h5"
    initial.write_bytes(b"initial")
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "saved_model.pb").write_bytes(b"fixed magnet")
    command = tmp_path / "evaluation.json"
    command.write_text(
        json.dumps(
            [
                "evaluate",
                "--games",
                "10000",
                "--seed",
                "1",
                "--mortal-source",
                "mortal",
                "{weights}",
                "{output}",
            ]
        )
    )
    args = types.SimpleNamespace(
        initial=str(initial),
        reference=str(reference),
        output=str(tmp_path / "run"),
        evaluation_command=str(command),
        sync_command=None,
        rounds=2,
        root_games=64,
        batch_size=2,
        workers=1,
        seed=941,
    )
    network = types.SimpleNamespace(
        actor=types.SimpleNamespace(),
        save_weights=lambda path: path.write_bytes(b"updated weights"),
    )
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind: [])
    monkeypatch.setattr(run.self_play, "initial_features", lambda seed: {})
    monkeypatch.setattr(
        run.export, "load_network", lambda path, features: network
    )
    monkeypatch.setattr(
        run.tf.saved_model,
        "load",
        lambda path: types.SimpleNamespace(signatures={"policy": None}),
    )
    monkeypatch.setattr(
        run.distributed.LearnerGroup, "bind", lambda *args: None
    )
    monkeypatch.setattr(run.trainer, "MMDTrainer", lambda **kwargs: None)
    monkeypatch.setattr(
        run.base_trainer,
        "CriticTrainer",
        lambda *args: types.SimpleNamespace(feature_packing=None),
    )
    monkeypatch.setattr(run.self_play, "predictor", lambda actor: None)
    monkeypatch.setattr(run.collection, "root_scorer", lambda model: None)
    commands = []
    monkeypatch.setattr(
        run.subprocess, "run", lambda command, check: commands.append(command)
    )
    phases = []
    monkeypatch.setattr(
        run, "train_phase", lambda **kwargs: phases.append(kwargs["phase"])
    )
    games = [{"path": f"game-{game}"} for game in range(64)]
    monkeypatch.setattr(matches, "sources", lambda path: games)
    mined = []

    def discover(*, predict, games, capture):
        mined.extend(games)
        return {"logged_games": len(games)}

    def collect(**kwargs):
        statistics = kwargs["discover"](None)
        (tmp_path / "run" / "STOP").touch()
        return {
            "features": {},
            "source": numpy.empty((0, 4), numpy.int64),
        }, statistics

    monkeypatch.setattr(run.matches, "discover", discover)
    monkeypatch.setattr(run.collection, "collect", collect)
    run.run(args)
    root = tmp_path / "run"
    assert json.loads((root / "plan.json").read_text())["replicas"] == 1
    receipt = json.loads((root / "round-1-rollouts.json").read_text())
    assert len(mined) == 32
    assert receipt["games"] == mined
    assert receipt["files"] == []
    assert receipt["positions"] == {"targeted": 0, "normal": 0}
    assert phases == ["match"]
    assert len(commands) == 2
    assert commands[1][2] == "training.match_records"
    assert "--learner-only" not in commands[1]
    assert (
        root / "latest.weights.h5"
    ).resolve().name == "round-1-match.weights.h5"
    status = (root / "status.json").read_text()
    assert json.loads(status)["stage"] == "stopped"
    with pytest.raises(FileExistsError):
        run.run(args)
    assert (root / "status.json").read_text() == status


def test_stage_publication_releases_workers_after_sync(tmp_path, monkeypatch):
    command = tmp_path / "sync.json"
    command.write_text(json.dumps(["sync", "{root}", "{stage}"]))
    args = types.SimpleNamespace(sync_command=str(command))
    chief_root = tmp_path / "chief"
    remote_root = tmp_path / "remote"
    chief_root.mkdir()
    remote_root.mkdir()
    events = []
    chief = types.SimpleNamespace(
        chief=True, barrier=lambda: events.append("chief-barrier")
    )
    worker = types.SimpleNamespace(
        chief=False, barrier=lambda: events.append("worker-barrier")
    )

    def sync(argv, check):
        assert check and argv == ["sync", str(chief_root), "round-1-match"]
        assert not (chief_root / "round-1-match-ready.json").exists()
        pending = chief_root / "round-1-match-ready.pending.json"
        (remote_root / "round-1-match-ready.json").write_bytes(
            pending.read_bytes()
        )
        events.append("synced")

    monkeypatch.setattr(run.subprocess, "run", sync)
    (chief_root / "STOP").touch()
    assert distributed.publish_stage(args, chief, chief_root, "round-1-match")
    assert distributed.publish_stage(args, worker, remote_root, "round-1-match")
    assert events == ["synced", "chief-barrier", "worker-barrier"]
    assert not (chief_root / "round-1-match-ready.pending.json").exists()
