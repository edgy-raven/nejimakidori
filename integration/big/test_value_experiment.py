"""Value pilots warm-start, isolate policy weights and hold out whole roots."""

import json
import types

import numpy
import pytest
import tensorflow as tf

from model import actions
from training import value_experiment


def test_value_fit_warm_starts_and_policy_probe_keeps_encoder_and_value_fixed(
    tmp_path,
):
    tf.keras.utils.set_random_seed(941)
    inputs = {"features": tf.keras.Input(shape=(3,), name="features")}
    board = tf.keras.layers.Dense(4, activation="tanh")(inputs["features"])
    hidden = tf.keras.layers.Dense(8, activation="gelu", name="value_hidden")(
        board
    )
    value = tf.keras.layers.Dense(1, name="state_value")(hidden)
    policy = tf.keras.Model(
        inputs,
        {
            "board": board,
            "state_value": value[:, 0],
            **{
                name: tf.keras.layers.Dense(actions.ACTOR_OUTPUTS[name][0])(
                    board
                )
                for name in actions.TRAIN_POLICY_HEADS
            },
        },
    )
    network = types.SimpleNamespace(
        actor=policy,
        encoder=tf.keras.Model(inputs, board),
        weights=policy.weights,
    )
    generator = numpy.random.default_rng(941)
    features = generator.normal(size=(128, 3)).astype(numpy.float32)
    encoded = network.encoder({"features": features}).numpy()
    original = value_experiment.weight_hash(network.weights)
    head = value_experiment.value_head(network)
    numpy.testing.assert_array_equal(
        head(encoded).numpy()[:, 0],
        policy({"features": features})["state_value"].numpy(),
    )
    assert numpy.any(head.get_weights()[0] != 0)
    widened = value_experiment.value_head(network, width=16)
    numpy.testing.assert_allclose(
        widened(encoded).numpy(), head(encoded).numpy(), atol=1e-6
    )
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((widened(encoded)[:, 0] - encoded[:, 0]) ** 2)
    gradients = tape.gradient(loss, widened.trainable_variables)
    assert numpy.any(gradients[2].numpy()[8:] != 0)
    deep = value_experiment.value_head(network, width=8, depth=3, dropout=0.1)
    numpy.testing.assert_allclose(
        deep(encoded, training=True).numpy(), head(encoded).numpy(), atol=1e-6
    )
    deep.get_layer("residual_value").kernel.assign(
        numpy.ones((128, 1), numpy.float32)
    )
    numpy.testing.assert_array_equal(
        deep(encoded, training=False).numpy(),
        deep(encoded, training=False).numpy(),
    )
    assert not numpy.array_equal(
        deep(encoded, training=True).numpy(),
        deep(encoded, training=True).numpy(),
    )
    scoring = numpy.zeros(
        len(encoded),
        dtype=[("public_fu", "i2", (4,)), ("public_bonus", "i1", (4,))],
    )
    scoring["public_fu"][:, 0] = 8
    scoring["public_bonus"][:, 1] = 3
    augmented = value_experiment.value_board(
        {"board": encoded, "features": scoring}, scoring_features=True
    )
    feature_head = value_experiment.value_head(
        network, linear=True, extra_features=8
    )
    numpy.testing.assert_allclose(
        feature_head(augmented).numpy(), head(encoded).numpy(), atol=1e-6
    )
    feature_head.get_layer("residual_value").kernel.assign(
        numpy.concatenate([numpy.zeros((4, 1)), numpy.ones((8, 1))])
    )
    feature_head.save(tmp_path / "scoring.keras")
    restored = tf.keras.models.load_model(tmp_path / "scoring.keras")
    numpy.testing.assert_allclose(
        restored(augmented).numpy() - head(encoded).numpy(),
        numpy.full((len(encoded), 1), 8 / 128 + 3 / 13),
        atol=1e-6,
    )
    targets = encoded[:, 0] * 0.5
    validation = numpy.arange(128) >= 96
    fit_args = types.SimpleNamespace(
        value_learning_rate=0.01, seed=941, epochs=30, patience=5
    )
    result = value_experiment.fit_head(
        head=head,
        board=encoded,
        targets=targets,
        validation=validation,
        args=fit_args,
    )
    assert result["selected_epoch"] > 0
    assert (
        min(row["validation_mse"] for row in result["history"])
        < result["history"][0]["validation_mse"]
    )
    assert value_experiment.weight_hash(network.weights) == original
    correction_norms = []
    for penalty in (0.0, 10.0):
        regularized = value_experiment.value_head(
            network, linear=True, penalty=penalty
        )
        numpy.testing.assert_allclose(
            regularized(encoded).numpy()[:, 0],
            policy({"features": features})["state_value"].numpy(),
            atol=1e-6,
        )
        value_experiment.fit_head(
            head=regularized,
            board=encoded,
            targets=targets,
            validation=validation,
            args=fit_args,
        )
        correction_norms.append(
            numpy.linalg.norm(
                regularized.get_layer("residual_value").kernel.numpy()
            )
        )
    assert correction_norms[1] < correction_norms[0] / 2
    assert value_experiment.weight_hash(network.weights) == original
    structured = numpy.empty(128, dtype=[("features", numpy.float32, (3,))])
    structured["features"] = features
    value_experiment.policy_probe(
        network,
        {
            "features": structured,
            "returns": numpy.stack((targets, targets), axis=1),
            "head": numpy.zeros(128, numpy.int32),
            "legal": numpy.tile(numpy.arange(150) < 37, (128, 1)),
            "chosen": numpy.zeros((128, 2), numpy.int32),
        },
        policy({"features": features})["state_value"].numpy(),
        types.SimpleNamespace(seed=941, policy_passes=1),
    )
    assert value_experiment.weight_hash(network.weights) != original
    value_model = tf.keras.Model(
        policy.input, policy.output["state_value"][:, None]
    )
    selected = {id(variable) for variable in value_model.weights}
    frozen = [
        variable for variable in policy.weights if id(variable) not in selected
    ]
    frozen_hash = value_experiment.weight_hash(frozen)
    encoder_hash = value_experiment.weight_hash(network.encoder.weights)
    result = value_experiment.fit_head(
        head=value_model,
        board=structured,
        targets=targets,
        validation=validation,
        args=types.SimpleNamespace(
            value_learning_rate=0.01, seed=941, epochs=20, patience=5
        ),
        batch_size=32,
        prediction_batch_size=32,
    )
    assert result["selected_epoch"] > 0
    assert value_experiment.weight_hash(frozen) == frozen_hash
    assert value_experiment.weight_hash(network.encoder.weights) != encoder_hash


def test_continuation_selects_the_root_seat_and_reports_return_noise(
    monkeypatch,
):
    def steps(*, games, seed, retain_features, collector, root_callback):
        root_callback(collector, {"head": numpy.zeros(games * 3)})
        yield numpy.zeros(games * 3)
        return {
            name: numpy.arange(games * 6)
            for name in ("settlement", "chosen", "head", "legal")
        }, {}

    monkeypatch.setattr(value_experiment.self_play, "collection_steps", steps)
    stream = value_experiment.continuation(
        types.SimpleNamespace(decision=2, fork=lambda copies, seed: None),
        4,
        941,
    )
    next(stream)
    with pytest.raises(StopIteration) as finished:
        stream.send(None)
    result = finished.value.value
    for name in ("chosen", "settlement"):
        numpy.testing.assert_array_equal(result[name], [2, 5, 8, 11])
    target = numpy.arange(20, dtype=numpy.float32)
    returns = numpy.stack((target - 1, target + 1), axis=1)
    result = value_experiment.metrics(target, returns)
    assert result["mean_return_rmse_st3"] == 0
    assert result["single_return_rmse_st3"] == 100
    assert result["explained_variance"] == 1
    assert result["mean_target_sampling_variance_st3_squared"] == 10000


def test_confirmation_requires_paired_improvement_on_fresh_returns(tmp_path):
    inputs = tf.keras.Input(shape=(3,), name="features")
    hidden = tf.keras.layers.Dense(3, name="value_hidden")(inputs)
    output = tf.keras.layers.Dense(1, name="state_value")(hidden)
    model = tf.keras.Model(inputs, output)
    model.get_layer("value_hidden").set_weights(
        [numpy.eye(3, dtype=numpy.float32), numpy.zeros(3, numpy.float32)]
    )
    model.get_layer("state_value").set_weights(
        [numpy.zeros((3, 1), numpy.float32), numpy.zeros(1, numpy.float32)]
    )
    model.save_weights(tmp_path / "original.weights.h5")
    model.get_layer("state_value").set_weights(
        [
            numpy.asarray([[1], [0], [0]], numpy.float32),
            numpy.zeros(1, numpy.float32),
        ]
    )
    model.save_weights(tmp_path / "candidate.weights.h5")
    data = tmp_path / "data"
    data.mkdir()
    board = (
        numpy.random.default_rng(941).normal(size=(64, 3)).astype(numpy.float32)
    )
    features = numpy.empty(64, dtype=[("features", numpy.float32, (3,))])
    features["features"] = board
    numpy.savez_compressed(
        file=data / "roots-000000.npz",
        board=board,
        features=features,
        returns=numpy.stack((board[:, 0] - 0.1, board[:, 0] + 0.1), axis=1),
    )
    network = types.SimpleNamespace(
        actor=model, weights=model.weights, load_weights=model.load_weights
    )
    args = types.SimpleNamespace(
        data=data,
        output=tmp_path,
        compare=tmp_path / "original.weights.h5",
        seed=941,
        alpha=0.025,
        scoring_features=False,
    )
    model.load_weights(args.compare)
    value_experiment.confirm(args, network)
    assert not json.loads((tmp_path / "summary.json").read_text())[
        "confirmed_improvement"
    ]
    args.compare = tmp_path / "candidate.weights.h5"
    value_experiment.confirm(args, network)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["confirmed_improvement"]
    assert summary["paired_interval"][1] < 0
    assert summary["candidate"]["mean_return_rmse_st3"] < 0.001
    for seed in (941, 942, 943):
        value_experiment.value_head(network).save(
            tmp_path / f"seed-{seed}.keras"
        )
    model.load_weights(tmp_path / "original.weights.h5")
    args.compare = tmp_path
    value_experiment.confirm_heads(args, network)
    assert json.loads((tmp_path / "summary.json").read_text())[
        "sustained_panel_improvement"
    ]
    value_experiment.value_head(network).save(tmp_path / "seed-943.keras")
    value_experiment.confirm_heads(args, network)
    assert not json.loads((tmp_path / "summary.json").read_text())[
        "sustained_panel_improvement"
    ]
    model.load_weights(tmp_path / "candidate.weights.h5")
    for seed in (941, 942, 943):
        model.save_weights(tmp_path / f"seed-{seed}.value.weights.h5")
    model.load_weights(tmp_path / "original.weights.h5")
    network.actor = tf.keras.Model(
        model.input, {"state_value": model.output[:, 0]}
    )
    value_experiment.confirm_encoder(args, network)
    assert json.loads((tmp_path / "summary.json").read_text())[
        "sustained_panel_improvement"
    ]


def test_learning_curve_keeps_nested_roots_and_excludes_test_panel(
    tmp_path, monkeypatch
):
    board = tf.keras.Input(shape=(4,))
    hidden = tf.keras.layers.Dense(8, name="value_hidden")(board)
    output = tf.keras.layers.Dense(1, name="state_value")(hidden)
    policy = tf.keras.Model(board, output)
    network = types.SimpleNamespace(actor=policy, weights=policy.weights)
    generator = numpy.random.default_rng(17)
    data = {
        "board": generator.normal(size=(32, 4)).astype(numpy.float32),
        "returns": generator.normal(size=(32, 2)).astype(numpy.float32),
    }
    split = numpy.repeat([0, 1, 2], [24, 4, 4])
    data["returns"][split == 2] = numpy.nan
    numpy.save(tmp_path / "split.npy", split)
    monkeypatch.setattr(value_experiment, "load_data", lambda path: data)
    original_hash = value_experiment.weight_hash(network.weights)
    value_experiment.learning_curve(
        types.SimpleNamespace(
            data=tmp_path,
            split=tmp_path / "split.npy",
            output=tmp_path,
            training_roots=[8, 16],
            epochs=1,
            patience=1,
        ),
        network,
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["trials"]) == 12
    for seed in (941, 942, 943):
        trials = [row for row in summary["trials"] if row["seed"] == seed]
        assert trials[0]["training_rows"] == trials[1]["training_rows"]
        assert trials[2]["training_rows"] == trials[3]["training_rows"]
        assert trials[0]["training_rows"] == trials[2]["training_rows"][:8]
        for trial in trials:
            assert trial["validation_rows"] == [24, 25, 26, 27]
            assert all(row < 24 for row in trial["training_rows"])
            assert numpy.isfinite(trial["validation"]["mean_return_rmse_st3"])
            assert (
                trial["history"][0]["validation_mse"]
                == trials[0]["history"][0]["validation_mse"]
            )
    assert value_experiment.weight_hash(network.weights) == original_hash
