"""Replay respects holdouts, wall-time credit and paired win evidence."""

import io
import json
import pathlib
import sqlite3

import numpy
import pytest
import tensorflow as tf

from log_dataset import records, replay, rollout_store
from model import critic, feature_vector, rollout_training


def test_replay_samples_training_games_without_counting_confirmation(tmp_path):
    path = tmp_path / "rollouts.sqlite"
    store = sqlite3.connect(path)
    store.executescript("""
        CREATE TABLE roots (
            id TEXT PRIMARY KEY, state TEXT NOT NULL, features BLOB NOT NULL,
            policy TEXT NOT NULL, position TEXT NOT NULL);
        CREATE TABLE samples (
            root TEXT NOT NULL REFERENCES roots(id), actions TEXT NOT NULL,
            seed INTEGER NOT NULL, sample INTEGER NOT NULL,
            stage TEXT NOT NULL, outcomes TEXT NOT NULL,
            PRIMARY KEY(root, actions, seed, sample));
    """)
    games = [{"type": "start_kyoku", "game": game} for game in range(100)]
    train_game = next(
        row
        for row in games
        if int(rollout_store.fingerprint(row)[:8], 16) % 10 < 7
    )
    heldout_game = next(
        row
        for row in games
        if int(rollout_store.fingerprint(row)[:8], 16) % 10 >= 7
    )
    pair = [
        {
            "payments": numpy.zeros((4, 4, 4), int).tolist(),
            "scores": [25000] * 4,
            "events": [{"type": "ryukyoku"}],
        }
    ] * 2
    for game, feature, distance, wall in (
        (train_game, 1, 1, 40),
        (heldout_game, 2, 1, 40),
        (train_game, 3, 2, 16),
        (train_game, 4, 3, 10),
    ):
        packed = io.BytesIO()
        values = feature_vector.FeatureVector().tensors
        for name, value in {
            "honba": feature,
            "normal_self_shanten": distance,
            "chiitoitsu_self_shanten": 5,
            "kokushi_self_shanten": 5,
            "live_wall_count": wall,
        }.items():
            values[name][...] = value
        numpy.savez_compressed(file=packed, **values)
        store.execute(
            "INSERT INTO roots VALUES (?, ?, ?, ?, ?)",
            (
                str(feature),
                json.dumps({"events": [game], "actor": 1}),
                packed.getvalue(),
                '{"policy":"old"}',
                str(feature),
            ),
        )
        store.executemany(
            "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)",
            [
                (str(feature), "[3,4]", 10, 0, "screen", json.dumps(pair)),
                (
                    str(feature),
                    "[8,9]",
                    11,
                    0,
                    "confirmation",
                    json.dumps(pair),
                ),
            ],
        )
    store.commit()
    store.close()
    replay = rollout_training.Replay(path)
    assert len(replay.positions) == 1
    for features, actions, labels in replay.batch(0, 5):
        numpy.testing.assert_array_equal(features["honba"], [1])
        numpy.testing.assert_array_equal(actions, [3, 4])
        numpy.testing.assert_array_equal(
            labels["payment_points"], numpy.zeros((2, 40))
        )
        assert labels["stale"].all()
        assert labels["win_advantage"] == 0
    replay.close()
    # Raw source IDs must follow the main corpus split. JSON-string hashing
    # wrongly admits both "4" (validation) and "1" (held-out test).
    assert [records.game_partition(game) for game in ("2", "4", "1")] == [
        4,
        7,
        9,
    ]
    with sqlite3.connect(path) as store:
        for game in ("2", "4", "1"):
            identity = "recorded-" + game
            store.execute(
                "INSERT INTO roots SELECT ?,?,features,policy,? "
                "FROM roots WHERE id='1'",
                (
                    identity,
                    json.dumps(
                        {"events": [train_game], "actor": 1, "game_id": game}
                    ),
                    identity,
                ),
            )
            store.execute(
                "INSERT INTO samples SELECT ?,actions,seed,sample,stage,"
                "outcomes FROM samples WHERE root='1'",
                (identity,),
            )
    replay = rollout_training.Replay(path)
    assert len(replay.positions) == 2
    replay.close()
    # The root actor is seat 1; each winner of a multiple ron counts as a win.
    pair[0] = {
        **pair[0],
        "events": [
            {"type": "hora", "actor": 1},
            {"type": "hora", "actor": 3},
        ],
    }
    pair[1] = {**pair[1], "events": [{"type": "hora", "actor": 0}]}
    with sqlite3.connect(path) as store:
        store.executemany(
            "INSERT INTO samples VALUES ('1','[3,4]',?,0,'screen',?)",
            [(seed, json.dumps(pair)) for seed in range(100, 131)],
        )
    replay = rollout_training.Replay(path)
    gaps = [labels["win_advantage"] for _, _, labels in replay.batch(0, 64)]
    assert min(gaps) == 0 < max(gaps) < 31 / 32
    replay.close()
    # Retained feature caches must fail before allocating a training model.
    del values["last_tedashi_tile"]
    packed = io.BytesIO()
    numpy.savez_compressed(file=packed, **values)
    with sqlite3.connect(path) as store:
        store.execute(
            "UPDATE roots SET features=? WHERE id='1'", (packed.getvalue(),)
        )
    store = rollout_store.RolloutStore(path)
    with pytest.raises(ValueError, match="regenerate the root feature cache"):
        store.root("1")
    store.close()


def test_budget_charges_setup_and_whole_replay_batches_and_waits_after_spike():
    budget = rollout_training.Budget(0.1)
    budget.replay_seconds = 2
    assert not budget.ready()
    for _ in range(320):
        budget.record(1, False)
    assert budget.ready()
    budget.record(5, True)
    assert not budget.ready()
    for _ in range(50):
        budget.record(1, False)
    assert budget.ready()
    budget.record(40, True)
    assert not budget.ready()
    assert budget.reserve_seconds == 80
    assert budget.metrics()["rollout_updates"] == 2
    with pytest.raises(ValueError, match="overhead"):
        rollout_training.Budget(0.11)


def test_replay_separates_settlement_gradients_from_paired_policy_learning():
    events = json.loads(
        pathlib.Path(__file__)
        .parents[1]
        .joinpath("fixtures/model/pass_speed_round.json")
        .read_text()
    )
    observation = next(replay.Replay("rollout-gradient", events).observations())
    features, _, _ = records.parse_batch(
        [records.serialize_observation(observation, True)]
    )
    network = critic.JointPolicyPayment()
    actions = network.encoder(features)["selected"][:2, 1]
    labels = {
        "payment_points": tf.zeros((2, 40), tf.int32),
        "stale": tf.ones(2, tf.bool),
        "win_advantage": tf.constant(0.25),
    }
    with tf.GradientTape(persistent=True) as tape:
        inputs = rollout_training.candidate_inputs(
            network, (features, actions, labels)
        )
        full, _ = rollout_training.loss(network.critic, inputs, 1.0)
        half, metrics = rollout_training.loss(network.critic, inputs, 0.5)
        regret, _ = rollout_training.policy_loss(
            network.actor, (features, actions, labels), 0.5
        )
    numpy.testing.assert_allclose(half, full / 2)
    assert all(
        gradient is None
        for gradient in tape.gradient(half, network.encoder.trainable_variables)
    )
    gradients = tape.gradient(half, network.critic.trainable_variables)
    assert all(gradient is not None for gradient in gradients)
    assert float(tf.linalg.global_norm(gradients)) > 0
    assert float(metrics["rollout_settlement_supported_fraction"]) == 1
    assert 0 < float(regret) < 0.125
    assert all(
        gradient is None
        for gradient in tape.gradient(
            regret, network.critic.trainable_variables
        )
    )
    assert (
        float(
            tf.linalg.global_norm(
                tape.gradient(regret, network.encoder.trainable_variables)
            )
        )
        > 0
    )
    assert (
        float(
            tf.linalg.global_norm(
                tape.gradient(
                    regret,
                    network.actor.get_layer("final_scores").trainable_variables,
                )
            )
        )
        > 0
    )


def test_paired_win_regret_ignores_ties_and_shrinks_uncertain_evidence():
    assert rollout_training.win_advantage([0] * 32) == 0
    assert rollout_training.win_advantage([1]) == 0
    assert rollout_training.win_advantage([1, -1] * 16) == 0
    advantage = rollout_training.win_advantage([1] * 16 + [0] * 16)
    assert 0 < advantage < 0.5
    numpy.testing.assert_allclose(
        rollout_training.win_advantage([-1] * 16 + [0] * 16), -advantage
    )
    logits = tf.Variable([[0.0] * 37])
    features = {
        "decision_phase": tf.constant([0]),
        "trigger_tile_id": tf.constant([0]),
    }

    def actor(features):
        return {
            "discard_policy_logits": logits,
            "riichi_policy_logits": tf.zeros((1, 74)),
            "response_policy_logits": tf.zeros((1, 150)),
            "kan_action_logits": tf.zeros((1, 35)),
        }

    with tf.GradientTape(persistent=True) as tape:
        labels = {
            "win_advantage": tf.constant(advantage, tf.float32),
            "stale": tf.ones(2, tf.bool),
        }
        regret, _ = rollout_training.policy_loss(
            actor, (features, tf.constant([3, 4]), labels), 0.5
        )
        tied, _ = rollout_training.policy_loss(
            actor,
            (features, tf.constant([3, 4]), {**labels, "win_advantage": 0.0}),
            0.5,
        )
    gradient = tape.gradient(regret, logits).numpy()[0]
    assert gradient[3] < 0 < gradient[4]
    numpy.testing.assert_allclose(gradient.sum(), 0, atol=1e-7)
    numpy.testing.assert_array_equal(tape.gradient(tied, logits), 0)
