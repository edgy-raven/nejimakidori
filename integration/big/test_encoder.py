"""Encoder inputs and shared-suit gradients preserve their contracts."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import _native_collection, features, replay, shanten
from model import encoder


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


def test_retired_analysis_is_skipped_and_cached_inputs_are_ignored(monkeypatch):
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
    # Retired fields in cached records cannot influence the encoder.
    for name in _native_collection.IGNORED_INPUTS:
        batch[name] = numpy.ones_like(batch[name])
    numpy.testing.assert_array_equal(board_model(batch), expected)

    # Exact candidate distances reach the encoder, unlike retired inputs.
    batch["candidate_yaku_shanten"] = batch["candidate_yaku_shanten"] + 1
    assert not numpy.allclose(board_model(batch), expected)
