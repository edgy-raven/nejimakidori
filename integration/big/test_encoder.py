"""Encoder inputs and shared-suit gradients preserve their contracts."""

import json
import pathlib

import numpy
import pytest
import tensorflow as tf

from log_dataset import _native_collection, features, replay, shanten
from model import actions, encoder, export, initialize
from training import self_play


def test_candidate_river_attention_learns_masked_action_specific_history():
    tf.keras.utils.set_random_seed(941)
    inputs = {
        name: tf.zeros((4, *spec["shape"]), tf.as_dtype(spec["dtype"]))
        for name, spec in _native_collection.INPUTS.items()
        if name.startswith("river_")
    }
    inputs["river_pre_valid"] = tf.tensor_scatter_nd_update(
        inputs["river_pre_valid"], [[0, 0, 0], [0, 0, 1], [2, 1, 3]], [1, 1, 1]
    )
    inputs["river_pre_tile_id"] = tf.tensor_scatter_nd_update(
        inputs["river_pre_tile_id"],
        [[0, 0, 0], [0, 0, 1], [2, 1, 3]],
        [4, 13, 22],
    )
    river = encoder.RiverEncoding(dtype="float32")
    _, tokens, valid = river(inputs)
    candidates = tf.random.normal((4, 384))
    selected = tf.constant([[0, 1], [0, 9], [1, 3], [2, 7]])
    attention = encoder.CandidateRiverAttention(dtype="float32")
    values = [candidates, selected, tokens, valid]
    numpy.testing.assert_array_equal(attention(values), candidates)

    # A neutral insertion must start learning without changing old outputs.
    target = tf.random.normal((4, 384))
    optimizer = tf.keras.optimizers.SGD(0.01)
    for _ in range(2):
        with tf.GradientTape() as tape:
            output = attention(values)
            loss = tf.reduce_sum(output * target)
        gradients = tape.gradient(loss, attention.trainable_variables)
        assert all(
            numpy.isfinite(tf.convert_to_tensor(g)).all() for g in gradients
        )
        optimizer.apply_gradients(zip(gradients, attention.trainable_variables))
    assert all(numpy.any(tf.convert_to_tensor(g) != 0) for g in gradients)
    output = attention(values).numpy()
    assert not numpy.allclose(
        output[0] - candidates[0], output[1] - candidates[1]
    )
    numpy.testing.assert_array_equal(output[2], candidates[2])

    # Padding, other boards and the unoccupied board cannot contaminate a query.
    padded = tf.where(valid[..., None], tokens, tf.ones_like(tokens) * 100)
    numpy.testing.assert_array_equal(
        attention([candidates, selected, padded, valid]), output
    )
    changed = tf.tensor_scatter_nd_add(tokens, [[2, 67]], [tf.ones(64) * 10])
    actual = attention([candidates, selected, changed, valid]).numpy()
    numpy.testing.assert_array_equal(actual[:3], output[:3])
    assert not numpy.allclose(actual[3], output[3])
    shifted = tf.roll(tokens, 1, axis=1)
    shifted_valid = tf.roll(valid, 1, axis=1)
    assert not numpy.allclose(
        attention([candidates, selected, shifted, shifted_valid])[0], output[0]
    )
    compiled = tf.function(attention)
    numpy.testing.assert_allclose(
        compiled(values), output, rtol=1e-5, atol=1e-5
    )
    assert attention([candidates[:0], selected[:0], tokens, valid]).shape == (
        0,
        384,
    )


def test_river_self_attention_connects_streams_without_padding_leakage():
    tf.keras.utils.set_random_seed(941)
    valid = tf.constant(
        [[True] * 64 + [False] * 64 + [True] * 64, [False] * 192]
    )
    tokens = tf.where(valid[..., None], tf.random.normal((2, 192, 64)), 0)
    attention = encoder.RiverSelfAttention(dtype="float32")
    numpy.testing.assert_array_equal(attention([tokens, valid]), tokens)
    optimizer = tf.keras.optimizers.SGD(0.01)
    target = tf.random.normal((2, 192, 64))
    for _ in range(2):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(attention([tokens, valid]) * target)
        gradients = tape.gradient(loss, attention.trainable_variables)
        assert all(
            numpy.isfinite(tf.convert_to_tensor(g)).all() for g in gradients
        )
        optimizer.apply_gradients(zip(gradients, attention.trainable_variables))
    assert all(numpy.any(tf.convert_to_tensor(g) != 0) for g in gradients)
    expected = attention([tokens, valid])
    changed = tf.tensor_scatter_nd_add(
        tokens, [[0, 191]], [tf.cast(tf.range(64), tf.float32)]
    )
    assert numpy.any(attention([changed, valid])[0, 0] != expected[0, 0])
    padded = tf.where(valid[..., None], tokens, 100)
    numpy.testing.assert_array_equal(attention([padded, valid]), expected)
    numpy.testing.assert_array_equal(expected[1], numpy.zeros((192, 64)))
    numpy.testing.assert_allclose(
        tf.function(attention)([tokens, valid]), expected, atol=1e-6
    )


@pytest.mark.parametrize("candidate_attention", [False, True])
def test_river_attention_weight_conversion_preserves_policy_and_critic(
    tmp_path, monkeypatch, candidate_attention
):
    inputs = self_play.initial_features(941)
    # Convert both earlier architectures without changing existing heads.
    with monkeypatch.context() as context:
        context.setattr(
            encoder,
            "RiverSelfAttention",
            lambda **kwargs: tf.keras.layers.Lambda(lambda values: values[0]),
        )
        if not candidate_attention:
            context.setattr(
                encoder,
                "CandidateRiverAttention",
                lambda **kwargs: lambda values: values[0],
            )
        previous = export.build_network(inputs)
    original = tmp_path / "original.weights.h5"
    previous.save_weights(original)
    previous_module = export.ServingModel(previous)
    expected = previous_module.serve(**inputs)
    output = tmp_path / "attention.weights.h5"
    converted = initialize.initialize(original, output, inputs)
    module = export.ServingModel(converted)
    actual = module.serve(**inputs)
    for name in expected:
        numpy.testing.assert_array_equal(actual[name], expected[name])
    # Persist an active branch too, not just its initially zero correction.
    correction = converted.encoder.get_layer(
        "candidate_river_attention"
    ).correction
    correction.kernel.assign(tf.random.normal(correction.kernel.shape) * 0.01)
    correction = converted.encoder.get_layer(
        "river_encoder"
    ).river_attention.correction
    correction.kernel.assign(tf.random.normal(correction.kernel.shape) * 0.01)
    trained = tmp_path / "trained.weights.h5"
    converted.save_weights(trained)
    loaded = export.load_network(trained, inputs)
    converted.compile_training()
    expected_policy = converted.actor(inputs)
    actual_policy = loaded.actor(inputs)
    for name in actions.TRAIN_POLICY_HEADS:
        numpy.testing.assert_allclose(
            actual_policy[name],
            expected_policy[name],
            rtol=1e-5,
            atol=1e-5,
        )
    tf.saved_model.save(
        module,
        str(tmp_path / "saved_model"),
        signatures={"policy": module.policy.get_concrete_function(**inputs)},
    )
    restored = tf.saved_model.load(str(tmp_path / "saved_model"))
    expected_policy = module.policy(**inputs)
    for name, value in restored.signatures["policy"](**inputs).items():
        numpy.testing.assert_allclose(
            value, expected_policy[name], rtol=1e-5, atol=1e-5
        )
    # Already-converted files are not silently reinitialized or overwritten.
    with pytest.raises(ValueError, match="missing river attention"):
        initialize.initialize(output, tmp_path / "invalid.weights.h5", inputs)


def test_reused_suits_match_float32_gradient_accumulation(monkeypatch):
    tf.keras.utils.set_random_seed(941)
    inputs = {
        name: tf.zeros((1, *spec["shape"]), tf.as_dtype(spec["dtype"]))
        for name, spec in _native_collection.INPUTS.items()
    }
    # Reusing the same closed hand must sum every suit contribution before
    # rounding back to bfloat16, just as encoding each contribution in float32.
    inputs["candidate_hand_valid"] = tf.ones((1, 96), tf.int8)
    inputs["candidate_hand_counts"] = tf.broadcast_to(
        tf.concat([tf.ones(13, tf.int8), tf.zeros(21, tf.int8)], axis=0),
        (1, 96, 34),
    )
    bank = encoder.HandBank(dtype="mixed_bfloat16")
    upstream = tf.random.stateless_normal((1, 96, 512), [941, 8])
    with tf.GradientTape() as tape:
        actual = bank(inputs)
        loss = tf.reduce_sum(tf.cast(actual, tf.float32) * upstream)
    actual_gradients = tape.gradient(loss, bank.trainable_variables)
    gather = tf.gather

    @tf.custom_gradient
    def reference_gather(values, indices):
        def gradient(upstream):
            return (
                tf.math.unsorted_segment_sum(
                    upstream, indices, tf.shape(values)[0]
                ),
                None,
            )

        return gather(values, indices), gradient

    def dense_suit_gather(params, indices, axis=None, **kwargs):
        if params.shape[-2:] == (9, 128):
            if axis == 1:
                return reference_gather(params[0], indices)[None]
            return reference_gather(params, indices)
        return gather(params, indices, axis=axis, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(tf, "gather", dense_suit_gather)
        with tf.GradientTape() as tape:
            expected = bank(inputs)
            loss = tf.reduce_sum(tf.cast(expected, tf.float32) * upstream)
        expected_gradients = tape.gradient(loss, bank.trainable_variables)
    numpy.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        numpy.testing.assert_allclose(
            tf.convert_to_tensor(actual),
            tf.convert_to_tensor(expected),
            rtol=1e-5,
            atol=1e-5,
        )


def test_retired_analysis_and_magnet_inputs_do_not_enter_encoder(monkeypatch):
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    state = replay.Replay("skip-completion", events)
    next(state.policy_decisions())
    actor = state.history[-1]["actor"]

    def rejected_python_analysis(*args, **kwargs):
        raise AssertionError("feature extraction must stay native")

    monkeypatch.setattr(shanten, "native", rejected_python_analysis)
    packed = features.pack(state, actor, "DISCARD")
    assert not any(name.startswith("discard_completion_") for name in packed)
    for name in _native_collection.IGNORED_INPUTS:
        numpy.testing.assert_array_equal(packed[name], 0)
        packed[name].fill(1)
    fresh = features.pack(state, actor, "DISCARD")
    for name in _native_collection.IGNORED_INPUTS:
        numpy.testing.assert_array_equal(fresh[name], 0)
    packed = fresh
    inputs, board, _, _, _ = encoder.encode_inputs()
    board_model = tf.keras.Model(inputs, board)
    batch = {name: value[None] for name, value in packed.items()}
    expected = board_model(batch).numpy()
    # Retired fields and fixed-policy scoring never enter candidate learning.
    for name in (
        *_native_collection.IGNORED_INPUTS,
        "magnet_win_value_range",
        "magnet_win_value_valid",
    ):
        batch[name] = numpy.ones_like(batch[name])
    numpy.testing.assert_array_equal(board_model(batch), expected)

    # Exact candidate distances reach the encoder, unlike retired inputs.
    batch["candidate_yaku_shanten"] = batch["candidate_yaku_shanten"] + 1
    assert not numpy.allclose(board_model(batch), expected)
