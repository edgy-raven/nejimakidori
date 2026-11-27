"""Candidate interactions preserve legal sets and predictive ownership."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import call_states, features, replay, scoring, training_data
from model import candidate_fusion, critic, export, losses, network


def test_candidate_set_symmetry_padding_and_selected_loss():
    tf.keras.utils.set_random_seed(221122)
    layer = candidate_fusion.CandidateFusion(width=8)
    tokens = tf.Variable(tf.random.normal([5, 24]))
    owners = tf.constant([0, 0, 0, 1, 1])
    batch = tf.constant(3)
    numpy.testing.assert_array_equal(layer([tokens, owners, batch]), tokens)
    # Zero initialization preserves the incumbent function. One selected-row
    # update opens the residual path; alternatives still get no direct label.
    with tf.GradientTape() as tape:
        loss = tf.square(layer([tokens, owners, batch])[0, 0] - 2.0)
    optimizer = tf.keras.optimizers.SGD(0.03)
    optimizer.apply_gradients(
        zip(
            tape.gradient(loss, layer.trainable_variables),
            layer.trainable_variables,
            strict=True,
        )
    )
    with tf.GradientTape() as tape:
        result = layer([tokens, owners, batch])
        selected_loss = tf.square(result[0, 0] - 2.0)
    gradient = tape.gradient(selected_loss, tokens).numpy()
    assert numpy.all(numpy.linalg.norm(gradient[:3], axis=-1) > 0)
    numpy.testing.assert_array_equal(gradient[3:], 0.0)
    order = [2, 0, 1, 4, 3]
    numpy.testing.assert_allclose(
        layer([tf.gather(tokens, order), tf.gather(owners, order), batch]),
        tf.gather(result, order),
        atol=1e-6,
    )
    # An unrelated position increases the internal padding capacity. Neither
    # its values nor that extra padding may affect the original positions.
    padded = tf.concat([tokens, tf.fill([9, 24], 10000.0)], axis=0)
    padded_owners = tf.concat([owners, tf.fill([9], 2)], axis=0)
    numpy.testing.assert_allclose(
        layer([padded, padded_owners, batch])[:5], result, atol=1e-6
    )
    clone = tf.keras.layers.deserialize(tf.keras.layers.serialize(layer))
    clone([tokens, owners, batch])
    clone.set_weights(layer.get_weights())
    numpy.testing.assert_array_equal(clone([tokens, owners, batch]), result)


def test_sparse_attention_matches_padded_definition_and_gradients():
    tf.keras.utils.set_random_seed(1729)
    layer = candidate_fusion.CandidateFusion(width=8)
    tokens = tf.Variable(tf.random.normal([5, 24]))
    owners = tf.constant([0, 0, 0, 1, 1])
    batch = tf.constant(3)
    layer([tokens, owners, batch])
    layer.normalization.beta.assign(tf.random.normal([24], stddev=0.1))
    layer.query_key_value.bias.assign(tf.random.normal([24], stddev=0.1))
    layer.correction.kernel.assign(
        tf.random.normal(layer.correction.kernel.shape, stddev=0.02)
    )
    target = tf.random.normal(tokens.shape)
    with tf.GradientTape(persistent=True) as tape:
        actual = layer([tokens, owners, batch])
        # Independent dense definition, including biased projected padding.
        # Its seven-slot capacity exceeds every real position's length.
        dense = tf.scatter_nd(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]], tokens, [3, 7, 24]
        )
        valid = tf.sequence_mask([3, 2, 0], 7)
        query, key, value = tf.split(
            layer.query_key_value(layer.normalization(dense)), 3, axis=-1
        )
        query, key = tf.nn.gelu(query), tf.nn.gelu(key)
        context = tf.matmul(
            tf.nn.softmax(
                tf.where(
                    valid[:, None, :],
                    tf.matmul(query, key, transpose_b=True) / numpy.sqrt(8),
                    -1e9,
                ),
                axis=-1,
            ),
            tf.where(valid[..., None], value, 0.0),
        )
        expected = tf.boolean_mask(
            dense + layer.correction(tf.concat([query, key, context], -1)),
            valid,
        )
        actual_loss = tf.reduce_sum(actual * target)
        expected_loss = tf.reduce_sum(expected * target)
    numpy.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
    for actual_gradient, expected_gradient in zip(
        tape.gradient(actual_loss, [tokens, *layer.trainable_variables]),
        tape.gradient(expected_loss, [tokens, *layer.trainable_variables]),
        strict=True,
    ):
        numpy.testing.assert_allclose(
            tf.convert_to_tensor(actual_gradient),
            tf.convert_to_tensor(expected_gradient),
            rtol=2e-5,
            atol=1e-5,
        )


def test_complete_call_set_reaches_beliefs_critic_and_detached_policy():
    tf.keras.utils.set_random_seed(941)
    start = {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "9s",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000] * 4,
        "tehais": [["?"] * 13 for _ in range(4)],
    }
    start["tehais"][1] = "1m 2m 3m 4m 4m 5m 6m 7m 7p 8p 9p E E".split()
    state = replay.Replay.from_start("fusion-calls", start)
    state.apply({"type": "tsumo", "actor": 0, "pai": "4m"})
    state.apply({"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": True})
    rows = tf.nest.map_structure(
        tf.constant, call_states.batch([features.pack(state, 1, "RESPONSE")])
    )
    model = network.JointPolicyPayment()
    initial = model.actor(rows)
    chosen = initial["selected"][:, 1].numpy()
    assert 111 in chosen
    assert set((chosen[(chosen >= 112) & (chosen < 260)] - 112) // 37) == {
        0,
        1,
        2,
        3,
    }
    # Emulate a nonzero residual after fitting, then exercise actual consumers.
    model.actor.fusion.correction.kernel.assign(
        tf.random.normal(
            model.actor.fusion.correction.kernel.shape, stddev=0.02
        )
    )
    with tf.GradientTape(persistent=True) as tape:
        outputs = model.actor(rows)
        belief_loss = tf.reduce_sum(outputs["route_affinity_logits"] ** 2)
        policy_loss = outputs["response_policy_logits"][0, 113]
    assert (
        tf.linalg.norm(
            tape.gradient(belief_loss, model.actor.fusion.correction.kernel)
        )
        > 0
    )
    assert tape.gradient(
        policy_loss, model.actor.fusion.trainable_variables
    ) == ([None] * len(model.actor.fusion.trainable_variables))
    assert tape.gradient(
        policy_loss,
        model.actor.policy.get_layer("neural_features").trainable_variables,
    ) == [None]
    assert (
        tf.linalg.norm(
            tape.gradient(
                policy_loss, model.actor.policy.get_layer("final_scores").kernel
            )
        )
        > 0
    )
    del tape
    assert (
        numpy.max(numpy.abs(initial["candidate"] - outputs["candidate"])) > 0.01
    )
    numpy.testing.assert_array_equal(
        outputs["candidate_context"][:, :384], outputs["candidate"]
    )
    assert (
        numpy.max(
            numpy.abs(
                initial["route_affinity_logits"]
                - outputs["route_affinity_logits"]
            )
        )
        > 1e-4
    )
    assert (
        numpy.max(
            numpy.abs(
                initial["response_policy_logits"]
                - outputs["response_policy_logits"]
            )
        )
        > 1e-4
    )
    probability = tf.nn.softmax(outputs["response_policy_logits"])
    numpy.testing.assert_allclose(
        [
            tf.reduce_sum(part)
            for part in tf.split(probability, [1, 37, 37, 37, 37, 1], axis=-1)
        ],
        [0.2] * 5 + [0],
        atol=1e-6,
    )
    events = json.loads(
        pathlib.Path(__file__)
        .parents[1]
        .joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    outcomes = scoring.round_labels(events + [{"type": "end_game"}])
    observations = list(replay.Replay("fusion-payment", events).observations())[
        :2
    ]
    parsed = training_data.parse_batch(
        tf.constant(
            [
                training_data.serialize(row, outcomes[row.kyoku_id])
                for row in observations
            ]
        )
    )
    with tf.GradientTape() as tape:
        statistics = model(parsed)["critic_statistics"]
        settlement_loss = losses.packed_means(statistics)[
            critic.CRITIC_METRICS.index("payment_nll")
        ]
    gradients = tape.gradient(
        settlement_loss,
        [
            model.actor.fusion.correction.kernel,
            model.actor.fusion.query_key_value.kernel,
            model.actor.policy.get_layer(
                "neural_features"
            ).tile_projection.kernel,
            model.actor.policy.get_layer("route_affinity").kernel,
        ],
    )
    assert all(tf.linalg.norm(value) > 0 for value in gradients[:3])
    assert gradients[3] is None


def test_attention_checkpoint_uses_default_export_path(tmp_path):
    events = json.loads(
        pathlib.Path(__file__)
        .parents[1]
        .joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    outcomes = scoring.round_labels(events + [{"type": "end_game"}])
    parsed = training_data.parse_batch(
        tf.constant(
            [
                training_data.serialize(row, outcomes[row.kyoku_id])
                for row in list(
                    replay.Replay("fusion-checkpoint", events).observations()
                )[:2]
            ]
        )
    )
    tf.keras.utils.set_random_seed(941)
    model = network.JointPolicyPayment()
    initial = model(parsed)["candidate"].numpy()
    model.actor.fusion.correction.kernel.assign(
        tf.random.normal(
            model.actor.fusion.correction.kernel.shape, stddev=0.02
        )
    )
    model.compile_training()
    expected = {name: value.numpy() for name, value in model(parsed).items()}
    assert numpy.max(numpy.abs(expected["candidate"] - initial)) > 0.01
    fusion_weights = [value.numpy() for value in model.actor.fusion.weights]
    checkpoint = tmp_path / "attention.weights.h5"
    model.save_weights(checkpoint)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(1729)
    restored = export.load_network(checkpoint, parsed[0])
    for value, saved in zip(
        restored.actor.fusion.weights, fusion_weights, strict=True
    ):
        numpy.testing.assert_array_equal(value, saved)
    for name, value in restored(parsed).items():
        numpy.testing.assert_array_equal(value, expected[name])
