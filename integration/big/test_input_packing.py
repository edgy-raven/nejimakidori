"""Compact transport preserves the learned semantic axes and projection."""

import numpy
import tensorflow as tf

from log_dataset import _native_collection, call_states, features, replay
from model import beliefs, input_features


def test_compact_melds_preserve_semantics_and_padding():
    features = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    features["candidate_meld_tiles"][...] = 34
    features["meld_tiles"][...] = 37
    features["last_tedashi_tile_id"][...] = 37
    features["candidate_meld_tiles"][0, 0, :3] = [1, 4, 27]
    features["candidate_meld_types"][0, 0, :3] = [0, 1, 3]
    features["candidate_meld_red"][0, 0, 0] = 1
    features["meld_tiles"][0, 0, 0, :3] = [0, 1, 2]
    features["meld_tiles"][0, 1, 0, :3] = [4, 34, 4]
    features["meld_tiles"][0, 1, 1] = 27
    features["meld_type"][0, 0, 0] = 1
    features["meld_type"][0, 1, :2] = [2, 4]
    features["last_tedashi_tile_id"][0, 1] = 34
    features["known_unavailable_counts"][0, 1] = 4
    features["river_pre_tile_id"][0, 1, :2] = [4, 31]
    features["river_pre_valid"][0, 1, :2] = 1
    features["river_post_tile_id"][0, 2, 0] = 22
    features["river_post_valid"][0, 2, 0] = 1
    decoded = input_features.neural_features(
        {name: tf.constant(value) for name, value in features.items()}
    )
    assert "meld_valid" not in input_features.NEURAL_INPUTS
    assert "meld_valid" not in decoded
    numpy.testing.assert_array_equal(
        decoded["meld_tile_valid"][0, 1],
        [[1, 1, 1, 0], [1, 1, 1, 1], [0] * 4, [0] * 4],
    )
    numpy.testing.assert_array_equal(
        decoded["meld_tile_valid"][0, 0, 0], [1, 1, 1, 0]
    )
    assert int(tf.reduce_sum(decoded["meld_tile_red"])) == 1
    expected = numpy.zeros((1, 16, 34, 6), numpy.int8)
    expected[0, 0, [1, 4, 27, 4], [0, 1, 3, 5]] = 1
    numpy.testing.assert_array_equal(decoded["candidate_melds"], expected)
    numpy.testing.assert_array_equal(
        decoded["meld_tile_ids"][0, 1, 0], [4, 4, 4, 0]
    )
    numpy.testing.assert_array_equal(
        decoded["meld_tile_red"][0, 1, 0], [0, 1, 0, 0]
    )
    numpy.testing.assert_array_equal(
        decoded["meld_tile_valid"][0, 1, 0], [1, 1, 1, 0]
    )
    assert decoded["last_tedashi_tile_id"][0, 1] == 34
    numpy.testing.assert_array_equal(
        decoded["last_tedashi_is_red"], [[0, 1, 0, 0]]
    )
    assert decoded["blocker_counts"][0, 0, 0] == 1

    projection = input_features.NeuralFeatures(dtype="float32")
    projection(features)
    projection.tile_projection.kernel.assign(
        tf.reshape(tf.range(34 * 8, dtype=tf.float32), (34, 8)) / 8
    )
    neural = projection(features)
    assert len(projection.trainable_variables) == 1
    assert not projection.tile_projection.use_bias
    numpy.testing.assert_array_equal(
        neural["tile_identity"], projection.tile_projection.kernel
    )
    # A red five shares its base identity everywhere; color is separate.
    codes = tf.constant([4, 13, 22, 34, 35, 36, 37], tf.int8)
    embedded = input_features.tile_features(codes, neural["tile_identity"])
    numpy.testing.assert_array_equal(embedded[:3], embedded[3:6])
    numpy.testing.assert_array_equal(embedded[6], 0)
    semantic = input_features.SemanticFields(
        ["meld_tile_ids", "last_tedashi_tile_id", "last_tedashi_is_red"],
        dtype="float32",
    )(neural)
    numpy.testing.assert_array_equal(
        semantic["meld_tile_ids"][0, 1, 0],
        tf.stack([embedded[0], embedded[0], embedded[0], tf.zeros(8)]),
    )
    numpy.testing.assert_array_equal(
        semantic["last_tedashi_tile_id"][0, 1], embedded[0]
    )
    numpy.testing.assert_array_equal(semantic["last_tedashi_tile_id"][0, 0], 0)
    numpy.testing.assert_array_equal(
        semantic["last_tedashi_is_red"][..., 0], [[0, 1, 0, 0]]
    )
    restored = tf.keras.layers.deserialize(
        tf.keras.layers.serialize(projection),
        custom_objects={"NeuralFeatures": input_features.NeuralFeatures},
    )
    restored(features)
    restored.set_weights(projection.get_weights())
    numpy.testing.assert_array_equal(
        restored(features)["tile_identity"], neural["tile_identity"]
    )


def test_semantic_fields_keep_masks_numeric_counts_and_nominal_identity():
    inputs = {
        "decision_phase": tf.constant([0, 4], tf.int8),
        "hands_remaining": tf.constant([7, -1], tf.int8),
        "scores": tf.constant([[25000], [-10000]], tf.int32),
        "ranks": tf.constant([[3], [4]], tf.int8),
        "self_efficiency": tf.constant([[-0.125, 0.5, 1.1], [0.25, 0.2, 0.8]]),
        "action_win_value_range": tf.constant(
            [[1000.0, 48000.0], [0.0, 8000.0]]
        ),
        "genbutsu_to_seat": tf.constant([[0, 1], [1, 0]], tf.int8),
        "known_unavailable_counts": tf.constant([[0, 4], [2, 1]], tf.int8),
        "current_draw_tile_id": tf.constant([0, 27], tf.int8),
        "current_draw_valid": tf.ones(2, tf.int8),
        "current_draw_is_red": tf.constant([0, 1], tf.int8),
        "river_pre_origin": tf.constant([[1, 0], [5, 0]], tf.int8),
        "river_pre_call_resolution": tf.constant([[1, 0], [5, 0]], tf.int8),
        "river_pre_valid": tf.constant([[1, 0], [1, 0]], tf.int8),
        "river_pre_event_age": tf.constant([[0.12345, 0], [0.9, 0]]),
    }
    layer = input_features.SemanticFields(inputs, dtype="mixed_bfloat16")
    inputs["tile_identity"] = input_features.NeuralFeatures(dtype="float32")(
        {
            name: tf.zeros((2, *spec["shape"]), tf.as_dtype(spec["dtype"]))
            for name, spec in _native_collection.INPUTS.items()
        }
    )["tile_identity"]
    encoded = layer(inputs)
    numpy.testing.assert_array_equal(
        encoded["self_efficiency"][..., 0],
        tf.cast(inputs["self_efficiency"], tf.bfloat16),
    )
    numpy.testing.assert_array_equal(
        encoded["scores"][..., 0], tf.cast([[0.25], [-0.1]], tf.bfloat16)
    )
    numpy.testing.assert_array_equal(
        encoded["ranks"][..., 0], tf.cast([[0.75], [1.0]], tf.bfloat16)
    )
    numpy.testing.assert_array_equal(
        encoded["hands_remaining"][..., 0],
        tf.cast(
            tf.math.asinh(tf.cast(inputs["hands_remaining"], tf.float32)),
            tf.bfloat16,
        ),
    )
    numpy.testing.assert_array_equal(
        encoded["action_win_value_range"][..., 0],
        tf.cast(tf.math.asinh(inputs["action_win_value_range"]), tf.bfloat16),
    )
    numpy.testing.assert_array_equal(
        encoded["decision_phase"], numpy.eye(5)[[0, 4]]
    )
    numpy.testing.assert_array_equal(
        encoded["genbutsu_to_seat"][..., 0], inputs["genbutsu_to_seat"]
    )
    numpy.testing.assert_array_equal(
        encoded["known_unavailable_counts"][..., 0],
        tf.cast(
            tf.math.asinh(
                tf.cast(inputs["known_unavailable_counts"], tf.float32)
            ),
            tf.bfloat16,
        ),
    )
    numpy.testing.assert_array_equal(
        encoded["river_pre_event_age"][..., 0],
        tf.cast(inputs["river_pre_event_age"], tf.bfloat16),
    )
    assert not numpy.array_equal(
        encoded["current_draw_tile_id"][0],
        encoded["current_draw_tile_id"][1],
    )
    numpy.testing.assert_array_equal(
        encoded["current_draw_is_red"][..., 0], [0, 1]
    )
    numpy.testing.assert_array_equal(
        layer.river_categories["call_resolution"].embeddings.shape, [6, 2]
    )
    assert not numpy.array_equal(
        encoded["river_pre_call_resolution"][0, 0],
        encoded["river_pre_call_resolution"][1, 0],
    )
    changed = {
        **inputs,
        "river_pre_origin": tf.constant([[1, 5], [5, 3]], tf.int8),
        "river_pre_call_resolution": tf.constant([[1, 4], [5, 2]], tf.int8),
        "river_pre_event_age": tf.constant([[0.12345, 9], [0.9, 7]]),
    }
    for name, value in layer(changed).items():
        numpy.testing.assert_array_equal(value, encoded[name])
    compiled = tf.function(layer)(inputs)
    for name, value in compiled.items():
        numpy.testing.assert_array_equal(value, encoded[name])


def test_native_concealed_counterfactuals_reach_opponent_completion_evidence():
    packed_rows = []
    # Unknown opposing hands leave a consistent physical wall for each swap.
    # Visible discards, indicator, seat state and our current draw stay fixed.
    for hand in (
        "1s 2s 2s 2s 2s 3s 4s 5sr 6s 7s 8s E E",
        "1m 2m 2m 2m 2m 3m 4m 5s 6m 7m 8m E E",
        "1m 2m 2m 2m 2m 3m 4m 5m 6m 7m 8m E E",
    ):
        state = replay.Replay.from_start(
            "concealed-counterfactual",
            dict(
                type="start_kyoku",
                bakaze="E",
                dora_marker="4s",
                honba=0,
                kyoku=2,
                kyotaku=0,
                oya=1,
                scores=[25000] * 4,
                tehais=[hand.split(), *[["?"] * 13 for _ in range(3)]],
            ),
        )
        for seat, tile in ((1, "9m"), (2, "9p"), (3, "N")):
            state.apply(dict(type="tsumo", actor=seat, pai=tile))
            state.apply(
                dict(type="dahai", actor=seat, pai=tile, tsumogiri=True)
            )
        state.apply(dict(type="tsumo", actor=0, pai="9s"))
        packed_rows.append(call_states.base(features.pack(state, 0, "DISCARD")))
    for name in packed_rows[0]:
        if name.startswith(("river_", "meld_")) or name in (
            "dora_multiplicity",
            "public_visible_red_counts",
            "scores",
            "ranks",
        ):
            for packed in packed_rows[1:]:
                numpy.testing.assert_array_equal(
                    packed[name], packed_rows[0][name]
                )
    inputs = {
        name: tf.constant(numpy.stack([row[name] for row in packed_rows]))
        for name in packed_rows[0]
    }
    numpy.testing.assert_array_equal(
        inputs["candidate_hand_counts"][:, 0].numpy().sum(-1), [14] * 3
    )
    numpy.testing.assert_array_equal(
        tf.gather(inputs["known_unavailable_counts"], [1, 19, 22], axis=1),
        [[0, 4, 1], [4, 0, 1], [4, 0, 0]],
    )
    assert numpy.all(inputs["known_unavailable_counts"] <= 4)
    lower, available, size = beliefs.held_count_context(inputs)
    numpy.testing.assert_array_equal(lower, 0)
    numpy.testing.assert_array_equal(size, 13)
    numpy.testing.assert_array_equal(available[:, 36], [0, 1, 1])
    logits = tf.random.stateless_normal((3, 3, 37, 5), [914, 2])
    projected = beliefs.CountProjection()(
        [
            beliefs.HeldCountMask()([logits, inputs]),
            inputs,
        ]
    )
    probability = tf.nn.softmax(projected, -1)
    # Four concealed copies and our sole red five cannot also be opponents'.
    numpy.testing.assert_array_equal(probability[0, :, 19, 1:], 0)
    numpy.testing.assert_array_equal(probability[0, :, 36, 1:], 0)
    numpy.testing.assert_array_equal(probability[1:, :, 1, 1:], 0)
    expected_counts = tf.reduce_sum(
        probability * tf.range(5, dtype=tf.float32), -1
    )
    assert numpy.all(expected_counts.numpy().sum(1) <= available.numpy() + 1e-5)
    evidence_layer = beliefs.CompletionEvidence()
    evidence = evidence_layer.evidence(inputs, projected)
    # Own/supply blocks expose suit totals, ordinary-dora and red availability.
    for coordinate, expected in (
        (2, numpy.array([12, 2, 1]) / 14),
        (6, numpy.array([1, 1, 0]) / 14),
        (10, [1, 0, 0]),
        (13, numpy.array([23, 33, 34]) / 14),
        (17, numpy.array([3, 3, 4]) / 14),
        (21, [0, 1, 1]),
    ):
        numpy.testing.assert_allclose(
            evidence[..., coordinate],
            numpy.repeat(numpy.array(expected)[:, None], 3, axis=1),
            atol=1e-7,
        )
    selected = tf.constant([2, 0, 1, 0])
    numpy.testing.assert_array_equal(
        evidence_layer.evidence(
            {
                name: tf.gather(value, selected)
                for name, value in inputs.items()
            },
            tf.gather(projected, selected),
        ),
        tf.gather(evidence, selected),
    )
