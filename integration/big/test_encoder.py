"""Encoder inputs and shared-suit gradients preserve their contracts."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import (
    _native_collection,
    call_states,
    features,
    replay,
    shanten,
)
from model import encoder, input_features


def test_river_attention_preserves_initial_encoding_and_reads_observed_events():
    tf.keras.utils.set_random_seed(941)
    inputs = {
        name: tf.zeros((2, *spec["shape"]), tf.as_dtype(spec["dtype"]))
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["river_pre_valid"] = tf.tensor_scatter_nd_update(
        inputs["river_pre_valid"], [[0, 0, 0], [0, 1, 0]], [1, 1]
    )
    inputs["river_pre_tile_id"] = tf.tensor_scatter_nd_update(
        inputs["river_pre_tile_id"], [[0, 0, 0], [0, 1, 0]], [4, 13]
    )
    inputs["river_post_valid"] = tf.tensor_scatter_nd_update(
        inputs["river_post_valid"], [[0, 0, 0]], [1]
    )
    inputs["river_post_tile_id"] = tf.tensor_scatter_nd_update(
        inputs["river_post_tile_id"], [[0, 0, 0]], [27]
    )
    inputs = input_features.NeuralFeatures(dtype="float32")(inputs)
    river = encoder.RiverEncoding(dtype="float32")
    board, local = river(inputs)
    numpy.testing.assert_array_equal(board[1], 0)
    numpy.testing.assert_array_equal(local[1], 0)
    permutation = [2, 0, 1]
    reordered_board, reordered_local = river(
        {
            name: (
                tf.gather(value, permutation, axis=1)
                if name.startswith("river_")
                else value
            )
            for name, value in inputs.items()
        }
    )
    numpy.testing.assert_allclose(
        reordered_local, tf.gather(local, permutation, axis=1), atol=1e-6
    )
    numpy.testing.assert_allclose(
        tf.reshape(reordered_board, (-1, 3, 128)),
        tf.gather(tf.reshape(board, (-1, 3, 128)), permutation, axis=1),
        atol=1e-6,
    )
    river.event_attention.correction.kernel.assign(tf.eye(64))
    board, local = river(inputs)
    numpy.testing.assert_array_equal(board[1], 0)
    numpy.testing.assert_array_equal(local[1], 0)
    padded = {
        **inputs,
        **{
            f"river_{phase}_tile_id": tf.where(
                inputs[f"river_{phase}_valid"] > 0,
                inputs[f"river_{phase}_tile_id"],
                33,
            )
            for phase in ("pre", "post")
        },
    }
    for actual, expected in zip(river(padded), (board, local)):
        numpy.testing.assert_array_equal(actual, expected)
    # Another opponent and the same opponent's post-riichi stream can both
    # inform the first opponent's pre-riichi summary.
    for field, position in (
        ("river_pre_tile_id", [0, 1, 0]),
        ("river_post_tile_id", [0, 0, 0]),
    ):
        changed = {
            **inputs,
            field: tf.tensor_scatter_nd_update(inputs[field], [position], [22]),
        }
        assert (
            float(
                tf.reduce_max(
                    abs(river(changed)[1][0, 0, :128] - local[0, 0, :128])
                )
            )
            > 1e-6
        )
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(river(inputs)[1][0, 0, :128] ** 2)
    gradients = tape.gradient(
        loss,
        [
            river.event_attention.attention.query_dense.kernel,
            river.event_attention.attention.key_dense.kernel,
            river.event_attention.attention.value_dense.kernel,
            river.event_attention.correction.kernel,
        ],
    )
    assert all(
        numpy.isfinite(value).all() and float(tf.linalg.norm(value)) > 0
        for value in gradients
    )
    restored = tf.keras.layers.deserialize(tf.keras.layers.serialize(river))
    restored(inputs)
    restored.set_weights(river.get_weights())
    for actual, expected in zip(restored(inputs), (board, local)):
        numpy.testing.assert_array_equal(actual, expected)


def test_reused_suits_match_float32_gradient_accumulation(monkeypatch):
    tf.keras.utils.set_random_seed(941)
    inputs = {
        name: tf.zeros((1, *spec["shape"]), tf.as_dtype(spec["dtype"]))
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["meld_tiles"] = tf.fill((1, 4, 4, 4), tf.constant(37, tf.int8))
    # Reusing the same closed hand must sum every suit contribution before
    # rounding back to bfloat16, just as encoding each contribution in float32.
    inputs["candidate_hand_valid"] = tf.ones((1, 16), tf.int8)
    inputs["candidate_hand_counts"] = tf.broadcast_to(
        tf.concat([tf.ones(13, tf.int8), tf.zeros(21, tf.int8)], axis=0),
        (1, 16, 34),
    )
    inputs = input_features.NeuralFeatures(dtype="float32")(inputs)
    bank = encoder.HandBank(dtype="mixed_bfloat16")
    shared = tf.random.stateless_normal((1, 512), [941, 9])
    upstream = tf.random.stateless_normal((1, 16, 512), [941, 8])
    with tf.GradientTape() as tape:
        tape.watch(shared)
        actual = bank([inputs, shared])
        loss = tf.reduce_sum(tf.cast(actual, tf.float32) * upstream)
    actual_gradients = tape.gradient(loss, [*bank.trainable_variables, shared])
    assert float(tf.linalg.norm(actual_gradients[-1])) > 0
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
            tape.watch(shared)
            expected = bank([inputs, shared])
            loss = tf.reduce_sum(tf.cast(expected, tf.float32) * upstream)
        expected_gradients = tape.gradient(
            loss, [*bank.trainable_variables, shared]
        )
    numpy.testing.assert_array_equal(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        numpy.testing.assert_allclose(
            tf.convert_to_tensor(actual_gradient),
            tf.convert_to_tensor(expected_gradient),
            rtol=1e-5,
            atol=1e-5,
        )
    # Public availability can differ for otherwise identical resulting hands.
    # Its flags belong only to the selected hand, with shared context fixed.
    changed = {
        **inputs,
        "candidate_hand_yaku_possibility": tf.tensor_scatter_nd_update(
            inputs["candidate_hand_yaku_possibility"], [[0, 7, 1]], [1]
        ),
    }
    with tf.GradientTape() as tape:
        changed_hands = bank([changed, shared])
        loss = tf.cast(changed_hands[0, 7, 0], tf.float32)
    gradient = tape.gradient(loss, bank.projection.layers[0].kernel)
    assert float(tf.linalg.norm(gradient[-19:])) > 0
    numpy.testing.assert_array_equal(changed_hands[:, :7], actual[:, :7])
    numpy.testing.assert_array_equal(changed_hands[:, 8:], actual[:, 8:])
    assert not numpy.allclose(changed_hands[:, 7], actual[:, 7])


def test_native_inputs_share_hand_board_and_tile_encoding(monkeypatch):
    tf.keras.utils.set_random_seed(941)
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    state = replay.Replay("skip-completion", events)
    for _ in state.policy_decisions():
        actor = state.history[-1]["actor"]
        if state.history[-1]["type"] == "tsumo" and any(
            state.rivers[seat] for seat in range(4) if seat != actor
        ):
            break

    def rejected_python_analysis(*args, **kwargs):
        raise AssertionError("feature extraction must stay native")

    monkeypatch.setattr(shanten, "native", rejected_python_analysis)
    packed = call_states.base(features.pack(state, actor, "DISCARD"))
    assert packed.keys() == _native_collection.INPUTS.keys()
    policy = tf.keras.mixed_precision.global_policy().name
    inputs, board, candidates, _, _, _ = encoder.encode_inputs()
    assert tf.keras.mixed_precision.global_policy().name == policy
    board_model = tf.keras.Model(inputs, board)
    batch = {name: value[None] for name, value in packed.items()}
    expected = board_model(batch).numpy()
    compiled = tf.function(tf.keras.Model(inputs, [board, candidates]))
    compiled_board, compiled_candidates = compiled(batch)
    # Fused BF16 graph kernels round at different boundaries from eager ops.
    for actual, reference in (
        (compiled_board, expected),
        (compiled_candidates, tf.keras.Model(inputs, candidates)(batch)),
    ):
        assert numpy.linalg.norm(actual - reference) < (
            0.025 * numpy.linalg.norm(reference)
        )
        assert numpy.isfinite(actual).all()
    assert compiled_board.dtype == tf.float32
    assert compiled_candidates.dtype == tf.float32
    assert all(
        board_model.get_layer(name).compute_dtype == "bfloat16"
        for name in (
            "hand_bank",
            "river_encoder",
            "table_context",
            "board_fusion",
        )
    )
    # Hand-bank addresses are routing metadata, not board evidence.
    for name in (
        "discard_candidate_hand",
        "kan_candidate_hand",
    ):
        batch[name] = (batch[name] + 1) % 16
    numpy.testing.assert_array_equal(board_model(batch), expected)
    batch = {name: value[None] for name, value in packed.items()}
    candidate_model = tf.keras.Model(inputs, candidates)
    hand_model = tf.keras.Model(
        inputs, board_model.get_layer("hand_bank").output
    )
    expected_candidates = candidate_model(batch).numpy()
    expected_hands = hand_model(batch).numpy()
    for name in encoder.CANDIDATE_SCORING_INPUTS:
        batch[name] = batch[name] + 1
    numpy.testing.assert_array_equal(board_model(batch), expected)
    numpy.testing.assert_array_equal(hand_model(batch), expected_hands)
    assert not numpy.allclose(candidate_model(batch), expected_candidates)
    # Both raw hand content and table context influence the fused board and
    # action candidates. The table/river path also conditions hand encoding.
    batch = {name: value[None] for name, value in packed.items()}
    joint_model = tf.keras.Model(inputs, [board, candidates])
    identity = next(
        value
        for value in joint_model.get_layer("neural_features").output
        if tuple(value.shape) == (34, 8)
    )
    assert all(
        any(value is identity for value in tf.nest.flatten(branch.input))
        for branch in (
            joint_model.get_layer("hand_bank"),
            joint_model.get_layer("river_encoder"),
            joint_model.get_layer("table_features"),
            joint_model.get_layer("decision_candidates"),
        )
    )
    expected_joint = joint_model(batch)
    for changed in (
        {
            "candidate_hand_counts": numpy.roll(
                batch["candidate_hand_counts"], 1, axis=-1
            )
        },
        {
            "scores": batch["scores"]
            + numpy.array([[5000, -5000, 0, 0]], numpy.int32)
        },
    ):
        actual = joint_model({**batch, **changed})
        for result, reference in zip(actual, expected_joint):
            assert not numpy.allclose(result, reference)
    with tf.GradientTape(persistent=True) as tape:
        actual_board, actual_candidates = joint_model(batch)
        board_loss = actual_board[0, 0]
        candidate_loss = actual_candidates[0, 0]
    for loss in (board_loss, candidate_loss):
        gradients = tape.gradient(
            loss,
            [
                joint_model.get_layer("neural_features").tile_projection.kernel,
                joint_model.get_layer("river_encoder").projection.kernel,
                joint_model.get_layer("hand_bank").tiles.kernel,
                joint_model.get_layer("hand_bank").context.kernel,
                joint_model.get_layer("table_context").project.layers[0].kernel,
                *[
                    joint_model.get_layer(f"board_block_{block}_hidden").kernel
                    for block in range(6)
                ],
            ],
        )
        assert all(
            gradient is not None
            and numpy.isfinite(gradient).all()
            and tf.linalg.global_norm([gradient]) > 0
            for gradient in gradients
        )
    del tape
    # Exact candidate distances reach action scoring, never board inference.
    batch["candidate_yaku_shanten"] = batch["candidate_yaku_shanten"] + 1
    numpy.testing.assert_array_equal(board_model(batch), expected)
    assert not numpy.allclose(candidate_model(batch), expected_candidates)
