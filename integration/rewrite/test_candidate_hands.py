"""All policy families address the same post-action hand bank."""

import numpy
import tensorflow as tf

import feature_vector
from model import decision_layers


def test_action_families_share_tile_hand_indices():
    inputs = {
        "discard_candidate_hand": tf.constant([numpy.arange(37)], tf.int8),
        "chi_candidate_hand": tf.constant(
            (numpy.arange(111) % 96).reshape(1, 3, 37), tf.int8
        ),
        "pon_candidate_hand": tf.constant(
            numpy.arange(20, 57).reshape(1, 1, 37), tf.int8
        ),
        "kan_candidate_hand": tf.constant([numpy.arange(1, 35)], tf.int8),
    }
    indices = decision_layers.candidate_hand_indices(inputs).numpy()
    assert indices.shape == (1, 295)
    numpy.testing.assert_array_equal(
        indices[
            0, [0, 34, 37, 71, 74, 108, 111, 112, 222, 223, 259, 260, 261, 294]
        ],
        [0, 34, 0, 34, 0, 34, 0, 0, 14, 20, 56, 0, 1, 34],
    )


def test_joint_hand_board_encoding_context_gradients_and_serialization():
    tf.keras.utils.set_random_seed(239)
    inputs = {
        name: numpy.zeros((3, *spec["shape"]), spec["dtype"])
        for name, spec in feature_vector.INPUTS.items()
    }
    inputs["candidate_hand_valid"][:, :2] = 1
    inputs["candidate_hand_counts"][:, :2, :9] = 1
    inputs["dora_multiplicity"][1, 3] = 1
    inputs["live_wall_count"][:] = [60, 60, 20]
    inputs = {name: tf.constant(value) for name, value in inputs.items()}
    separate = decision_layers.HandBank()(inputs)
    numpy.testing.assert_array_equal(separate[0], separate[1])
    numpy.testing.assert_array_equal(separate[0], separate[2])

    encoder = decision_layers.HandBank(board_context=True)
    with tf.GradientTape() as tape:
        encoded = encoder(inputs)
        loss = tf.reduce_sum(encoded[:, :2, :7])
    gradients = tape.gradient(
        loss, [encoder.tiles.kernel, encoder.board_projection.kernel]
    )
    assert all(numpy.isfinite(value).all() for value in gradients)
    assert all(numpy.any(value != 0) for value in gradients)
    assert not numpy.allclose(encoded[0, 0], encoded[1, 0])
    assert not numpy.allclose(encoded[0, 0], encoded[2, 0])
    numpy.testing.assert_array_equal(encoded[:, 0], encoded[:, 1])
    numpy.testing.assert_array_equal(encoded[:, 2:], 0)

    restored = tf.keras.layers.deserialize(tf.keras.layers.serialize(encoder))
    restored(inputs)
    restored.set_weights(encoder.get_weights())
    numpy.testing.assert_array_equal(restored(inputs), encoded)


def test_pre_and_post_riichi_rivers_are_independently_padded_and_encoded():
    import types

    from model import features, records

    state = types.SimpleNamespace(
        start={"oya": 0, "kyoku": 1},
        hanchan=None,
        declared=[False, True, False, False],
        accepted=[False, True, False, False],
        ippatsu=[False] * 4,
        melds=[[] for _ in range(4)],
        meld_events=[[] for _ in range(4)],
        history=[{"type": "reach_accepted", "actor": 1}] * 8,
        rivers=[[], [], [], []],
    )
    state.rivers[1] = [
        {
            "pai": tile,
            "riichi": turn == 1,
            "tsumogiri": turn >= 2,
            "event_index": turn + 1,
            "resolution": 1,
            "caller": None,
            "origin": 1,
            "own_turns": [0] * 4,
            "live_wall": 60 - turn,
            "dora_markers": ["1p"],
        }
        for turn, tile in enumerate(["1m", "5mr", "9p", "E"])
    ]
    state.rivers[0] = [{**state.rivers[1][0], "pai": "S", "riichi": False}]
    packed = feature_vector.FeatureVector().__dict__
    features.table_features(
        packed, state, 0, types.SimpleNamespace(scores=[25000] * 4)
    )
    numpy.testing.assert_array_equal(
        packed["river_pre_tile_id"][0, -2:], [0, 4]
    )
    numpy.testing.assert_array_equal(
        packed["river_post_tile_id"][0, -2:], [17, 27]
    )
    numpy.testing.assert_array_equal(packed["river_pre_riichi"][0, -2:], [0, 1])
    assert packed["river_pre_is_red"][0, -1] == 1
    assert packed["river_post_tsumogiri"][0, -2:].sum() == 2
    for phase in ("pre", "post"):
        assert packed["river_" + phase + "_valid"].sum() == 2
        assert not packed["river_" + phase + "_valid"][:, :-2].any()
    assert packed["genbutsu_to_seat"][0, 28] == 1
    inputs = {
        name: tf.constant(value[None])
        for name, value in packed.items()
        if name.startswith("river_")
    }
    encoder = decision_layers.RiverEncoding()
    with tf.GradientTape() as tape:
        encoded = encoder(inputs)
        loss = tf.reduce_sum(encoded)
    gradients = tape.gradient(loss, encoder.trainable_variables)
    assert all(value is not None for value in gradients)
    assert encoded.shape == (1, 384)
    numpy.testing.assert_array_equal(encoded.numpy().reshape(3, 2, 64)[1:], 0)
    changed = dict(inputs)
    changed["river_post_tile_id"] = tf.tensor_scatter_nd_update(
        inputs["river_post_tile_id"], [[0, 0, 31]], [28]
    )
    modified = encoder(changed).numpy().reshape(3, 2, 64)
    numpy.testing.assert_array_equal(
        modified[:, 0], encoded.numpy().reshape(3, 2, 64)[:, 0]
    )
    assert not numpy.allclose(
        modified[0, 1], encoded.numpy().reshape(3, 2, 64)[0, 1]
    )
    for name, value in inputs.items():
        numpy.testing.assert_array_equal(
            records.decode(
                "feature/" + name,
                records.encode("feature/" + name, value[0].numpy()),
                feature_vector.INPUTS[name],
            ),
            value[0],
        )
