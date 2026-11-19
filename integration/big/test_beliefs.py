"""Opponent counts obey public hand sizes and physical tile limits."""

import numpy
import tensorflow as tf

from model import beliefs


def test_count_projection_respects_melds_reds_and_shared_ankou_blockers(
    tmp_path,
):
    features = {
        "meld_tile_ids": numpy.zeros((1, 4, 4, 4), numpy.int8),
        "meld_tile_red": numpy.zeros((1, 4, 4, 4), numpy.int8),
        "meld_tile_valid": numpy.zeros((1, 4, 4, 4), numpy.int8),
        "meld_valid": numpy.zeros((1, 4, 4), numpy.int8),
        "known_unavailable_counts": numpy.zeros((1, 34), numpy.int8),
        "candidate_hand_red": numpy.zeros((1, 96, 3), numpy.int8),
        "public_visible_red_counts": numpy.array([[1, 0, 0]], numpy.int8),
        "seat_wind": numpy.array([[0, 1, 2, 3]], numpy.int8),
        "prevailing_wind": numpy.array([0], numpy.int8),
        "dora_multiplicity": numpy.eye(34, dtype=numpy.int8)[[0]],
        "riichi_state": numpy.array([[0, 0, 1, 2]], numpy.int8),
    }
    # Our 111m and ten other tiles; opponent 1 has melded 550m.
    features["known_unavailable_counts"][0, [0, 4]] = 3
    features["known_unavailable_counts"][0, 18:28] = 1
    features["meld_tile_ids"][0, 1, 0, :3] = 4
    features["meld_tile_red"][0, 1, 0, 2] = 1
    features["meld_tile_valid"][0, 1, 0, :3] = 1
    features["meld_valid"][0, 1, 0] = 1
    features = {key: tf.constant(value) for key, value in features.items()}
    counts = numpy.zeros((1, 3, 37), numpy.int32)
    counts[0, :, [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]] = 1
    counts[0, 1:, [15, 16, 17]] = 1
    counts[0, 0, [4, 34]] = [2, 1]
    logits = tf.Variable(numpy.zeros((1, 3, 37, 5), numpy.float32))
    lower, available, size = beliefs.held_count_context(features)
    # Projection must resolve the shared last copy while preserving the
    # exposed red meld and each concealed hand size. Gradients stay usable.
    with tf.GradientTape() as tape:
        projected = beliefs.CountProjection()(
            [beliefs.HeldCountMask()([logits, features]), features]
        )
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=counts, logits=projected
            )
        )
    gradient = tape.gradient(loss, logits).numpy()
    assert numpy.isfinite(gradient).all()
    assert numpy.any(gradient != 0)
    probability = tf.nn.softmax(projected, -1).numpy()
    concealed = (probability * numpy.arange(5)).sum(-1) - lower.numpy()
    numpy.testing.assert_allclose(concealed.sum(-1), size.numpy(), atol=1e-5)
    assert numpy.all(concealed.sum(1) <= available.numpy() + 1e-5)
    numpy.testing.assert_array_equal(probability[0, :, 0, 2:], 0)
    assert probability[0, :, 0, 1].sum() <= 1 + 1e-5
    numpy.testing.assert_array_equal(probability[0, 0, 34], [0, 1, 0, 0, 0])
    numpy.testing.assert_allclose(
        tf.nn.softmax(beliefs.CountProjection()([projected, features]), -1),
        probability,
        atol=1e-5,
    )

    # Check the constrained derivative against the forward solve, independent
    # of the backward implementation and its handling of active copy limits.
    direction = tf.random.stateless_normal(logits.shape, [941, 3])
    difference = (
        sum(
            sign
            * tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=counts,
                    logits=beliefs.CountProjection()(
                        [
                            beliefs.HeldCountMask()(
                                [logits + sign * 0.01 * direction, features]
                            ),
                            features,
                        ]
                    ),
                )
            )
            for sign in (-1, 1)
        )
        / 0.02
    )
    numpy.testing.assert_allclose(
        tf.reduce_sum(gradient * direction), difference, atol=2e-3, rtol=2e-3
    )

    # Saturated count heads previously exhausted the solver's 80 steps and
    # returned physically inconsistent means. Keep their gradients finite.
    logits.assign(tf.random.stateless_normal(logits.shape, [941, 2]) * 32)
    with tf.GradientTape() as tape:
        projected = beliefs.CountProjection()(
            [beliefs.HeldCountMask()([logits, features]), features]
        )
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=counts, logits=projected
            )
        )
    gradient = tape.gradient(loss, logits).numpy()
    assert numpy.isfinite(gradient).all()
    assert numpy.any(gradient != 0)
    concealed = tf.reduce_sum(
        tf.nn.softmax(projected)
        * (tf.range(5, dtype=tf.float32) - lower[..., None]),
        -1,
    ).numpy()
    numpy.testing.assert_allclose(concealed.sum(-1), size, atol=1e-4)
    assert numpy.all(concealed.sum(1) <= available.numpy() + 1e-4)

    # Pending and accepted riichi agree with Q's guaranteed-tenpai facts.
    # The detached constraint target still trains the raw shanten head.
    heads = beliefs.OpponentBeliefs(encoder=None)
    with tf.GradientTape() as tape:
        outputs = heads(features, hidden=tf.zeros((1, 512)))
        penalty = tf.reduce_sum(outputs["belief_constraint_loss"])
    probability = tf.nn.softmax(outputs[beliefs.BELIEF_HEADS[0]]).numpy()
    numpy.testing.assert_array_equal(probability[0, 1:], [[1, 0, 0, 0]] * 2)
    assert numpy.all(probability[0, 0] > 0)
    gradient = (
        tape.gradient(penalty, heads.heads[beliefs.BELIEF_HEADS[0]].bias)
        .numpy()
        .reshape(3, 4)
    )
    numpy.testing.assert_array_equal(gradient[0], 0)
    assert numpy.all(gradient[1:, 0] < 0)
    assert numpy.all(gradient[1:, 1:] > 0)

    # Four exposed melds leave one concealed tile: hadaka tanki is known
    # tenpai without riichi, and its raw head receives the same correction.
    hadaka = {name: value.numpy().copy() for name, value in features.items()}
    hadaka["meld_valid"][0, 1] = 1
    hadaka["meld_tile_ids"][0, 1, 1:, :3] = numpy.arange(1, 4)[:, None]
    hadaka["meld_tile_valid"][0, 1, :, :3] = 1
    hadaka["known_unavailable_counts"][0, 1:4] = 3
    assert hadaka["riichi_state"][0, 1] == 0
    with tf.GradientTape() as tape:
        outputs = heads(hadaka, hidden=tf.zeros((1, 512)))
        penalty = tf.reduce_sum(outputs["belief_constraint_loss"])
    numpy.testing.assert_array_equal(
        tf.nn.softmax(outputs[beliefs.BELIEF_HEADS[0]])[0],
        [[1, 0, 0, 0]] * 3,
    )
    gradient = (
        tape.gradient(penalty, heads.heads[beliefs.BELIEF_HEADS[0]].bias)
        .numpy()
        .reshape(3, 4)
    )
    assert numpy.all(gradient[:, 0] < 0)
    assert numpy.all(gradient[:, 1:] > 0)

    # Attention starts as an identity correction, then both blocks learn
    # through the existing count/wait objectives. Check weight round-tripping.
    hidden = tf.zeros((1, 512))
    initial = heads(features, hidden=hidden)
    numpy.testing.assert_array_equal(
        initial[beliefs.BELIEF_HEADS[1]],
        tf.reshape(heads.heads[beliefs.BELIEF_HEADS[1]](hidden), [1, 3, 4, 34]),
    )
    optimizer = tf.keras.optimizers.SGD(0.001)
    for _ in range(2):
        with tf.GradientTape() as tape:
            outputs = heads(features, hidden=hidden)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=counts, logits=outputs[beliefs.BELIEF_HEADS[2]]
                )
            ) + tf.reduce_mean(tf.nn.softplus(outputs[beliefs.BELIEF_HEADS[1]]))
        gradients = tape.gradient(loss, heads.trainable_variables)
        for gradient in gradients:
            assert gradient is not None
            assert bool(tf.reduce_all(tf.math.is_finite(gradient)))
        optimizer.apply_gradients(zip(gradients, heads.trainable_variables))
    for attention in heads.tile_attention.attention:
        assert any(
            bool(tf.reduce_any(gradient != 0))
            for variable, gradient in zip(heads.trainable_variables, gradients)
            if any(variable is weight for weight in attention.trainable_weights)
        )
    weights = tmp_path / "beliefs.weights.h5"
    heads.save_weights(weights)
    restored = beliefs.OpponentBeliefs(encoder=None)
    restored(features, hidden=hidden)
    restored.load_weights(weights)
    for name in beliefs.BELIEF_HEADS:
        numpy.testing.assert_allclose(
            restored(features, hidden=hidden)[name],
            heads(features, hidden=hidden)[name],
            atol=1e-6,
        )
