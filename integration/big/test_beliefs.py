"""Opponent counts obey public hand sizes and physical tile limits."""

import numpy
import tensorflow as tf

from model import beliefs


def test_count_projection_respects_melds_reds_and_shared_ankou_blockers(
    tmp_path,
):
    features = {
        "meld_tiles": numpy.full((1, 4, 4, 4), 37, numpy.int8),
        "known_unavailable_counts": numpy.zeros((1, 34), numpy.int8),
        "candidate_hand_red": numpy.zeros((1, 16, 3), numpy.int8),
        "public_visible_red_counts": numpy.array([[1, 0, 0]], numpy.int8),
        "seat_wind": numpy.array([[0, 1, 2, 3]], numpy.int8),
        "prevailing_wind": numpy.array([0], numpy.int8),
        "dora_multiplicity": numpy.eye(34, dtype=numpy.int8)[[0]],
        "candidate_hand_counts": numpy.zeros((1, 16, 34), numpy.int8),
        "trigger_tile_id": numpy.zeros(1, numpy.int8),
        "trigger_source": numpy.zeros(1, numpy.int8),
        "trigger_valid": numpy.zeros(1, numpy.int8),
        "candidate_fu": numpy.zeros((1, 16), numpy.int8),
        "public_bonus": numpy.zeros((1, 4), numpy.int8),
        "meld_type": numpy.zeros((1, 4, 4), numpy.int8),
        "dealer_relative_seat": numpy.zeros(1, numpy.int8),
        "opponent_discard_count": numpy.zeros((1, 4), numpy.int8),
        "live_wall_count": numpy.full(1, 50, numpy.int8),
        "last_tedashi_tile_id": numpy.full((1, 4), 37, numpy.int8),
        "riichi_state": numpy.array([[0, 0, 1, 2]], numpy.int8),
    }
    # Our 111m and ten other tiles; opponent 1 has melded 550m.
    features["known_unavailable_counts"][0, [0, 4]] = 3
    features["known_unavailable_counts"][0, 18:28] = 1
    features["meld_tiles"][0, 1, 0, :3] = 4
    features["meld_tiles"][0, 1, 0, 2] = 34
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
    rivers = tf.zeros((1, 3, 256))
    heads = beliefs.Beliefs()
    with tf.GradientTape() as tape:
        outputs = heads(features, hidden=tf.zeros((1, 512)), rivers=rivers)
        penalty = tf.reduce_sum(outputs["belief_constraint_loss"])
    probability = tf.nn.softmax(outputs[beliefs.BELIEF_HEADS[0]]).numpy()
    numpy.testing.assert_array_equal(probability[0, 1:], [[1, 0, 0, 0]] * 2)
    assert numpy.all(probability[0, 0] > 0)
    gradient = tape.gradient(
        penalty, heads.heads[beliefs.BELIEF_HEADS[0]].readout.bias
    ).numpy()
    assert gradient[0] < 0
    assert numpy.all(gradient[1:] > 0)

    # Four exposed melds leave one concealed tile: hadaka tanki is known
    # tenpai without riichi, and its raw head receives the same correction.
    hadaka = {name: value.numpy().copy() for name, value in features.items()}
    hadaka["meld_tiles"][0, 1, 1:, :3] = numpy.arange(1, 4)[:, None]
    hadaka["known_unavailable_counts"][0, 1:4] = 3
    assert hadaka["riichi_state"][0, 1] == 0
    with tf.GradientTape() as tape:
        outputs = heads(hadaka, hidden=tf.zeros((1, 512)), rivers=rivers)
        penalty = tf.reduce_sum(outputs["belief_constraint_loss"])
    numpy.testing.assert_array_equal(
        tf.nn.softmax(outputs[beliefs.BELIEF_HEADS[0]])[0],
        [[1, 0, 0, 0]] * 3,
    )
    gradient = tape.gradient(
        penalty, heads.heads[beliefs.BELIEF_HEADS[0]].readout.bias
    ).numpy()
    assert gradient[0] < 0
    assert numpy.all(gradient[1:] > 0)

    # Attention starts as an identity correction, then both blocks learn
    # through the existing count/wait objectives. Check weight round-tripping.
    hidden = tf.zeros((1, 512))
    initial = heads(features, hidden=hidden, rivers=rivers)
    numpy.testing.assert_array_equal(
        initial[beliefs.BELIEF_HEADS[1]],
        tf.reshape(heads.heads[beliefs.BELIEF_HEADS[1]](hidden), [1, 3, 4, 34]),
    )
    # Only opponent yaku receives a projected-count residual. The self
    # yaku and self-only score readouts remain unchanged by that residual.
    numpy.testing.assert_array_equal(
        heads.completion_evidence(
            features, initial[beliefs.BELIEF_HEADS[2]], rivers
        ),
        0,
    )
    heads.completion_evidence.readout.kernel.assign(
        tf.random.stateless_normal(
            heads.completion_evidence.readout.kernel.shape, [17, 8]
        )
        * 0.01
    )
    with tf.GradientTape(persistent=True) as tape:
        conditioned = heads(features, hidden=hidden, rivers=rivers)
        count = tf.reduce_sum(conditioned[beliefs.BELIEF_HEADS[2]])
        improving = tf.reduce_sum(conditioned[beliefs.BELIEF_HEADS[1]])
        yaku = tf.reduce_sum(
            tf.nn.sigmoid(conditioned[beliefs.COMPLETION_HEADS[0]][:, 1:])
        )
        score = tf.reduce_sum(
            tf.nn.softmax(conditioned[beliefs.COMPLETION_HEADS[1]])
            * tf.range(
                conditioned[beliefs.COMPLETION_HEADS[1]].shape[-1],
                dtype=tf.float32,
            )
        )
    for name in beliefs.COMPLETION_HEADS:
        numpy.testing.assert_array_equal(
            conditioned[name][:, 0], initial[name][:, 0]
        )
    assert numpy.any(
        conditioned[beliefs.COMPLETION_HEADS[0]][:, 1:]
        != initial[beliefs.COMPLETION_HEADS[0]][:, 1:]
    )
    assert conditioned[beliefs.COMPLETION_HEADS[1]].shape[1:3] == (1, 2)
    labels = {
        "round_completion_yaku": tf.fill((1, 4, 2, 23), -1),
        "round_completion_score": tf.zeros((1, 4, 2), tf.int32),
    }
    native_loss = beliefs.completion_loss_terms(labels, conditioned)[
        "completion_score_loss"
    ]
    labels["round_completion_score"] = tf.concat(
        [labels["round_completion_score"][:, :1], tf.fill((1, 3, 2), -1)],
        axis=1,
    )
    numpy.testing.assert_array_equal(
        native_loss,
        beliefs.completion_loss_terms(labels, conditioned)[
            "completion_score_loss"
        ],
    )
    for objective in (yaku, score):
        assert (
            tape.gradient(objective, heads.heads[beliefs.BELIEF_HEADS[2]].bias)
            is None
        )
    assert (
        tape.gradient(score, heads.heads[beliefs.COMPLETION_HEADS[0]].bias)
        is None
    )
    for objective, other in (
        (count, beliefs.BELIEF_HEADS[1]),
        (improving, beliefs.BELIEF_HEADS[2]),
    ):
        assert tape.gradient(objective, heads.heads[other].bias) is None
        assert (
            tape.gradient(
                objective, heads.heads[beliefs.BELIEF_HEADS[0]].readout.bias
            )
            is None
        )
    for objective, name in zip((yaku, score), beliefs.COMPLETION_HEADS):
        assert (
            tf.linalg.global_norm(
                tape.gradient(objective, heads.heads[name].trainable_variables)
            )
            > 0
        )
    del tape
    optimizer = tf.keras.optimizers.SGD(0.001)
    for _ in range(2):
        with tf.GradientTape() as tape:
            outputs = heads(features, hidden=hidden, rivers=rivers)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=counts, logits=outputs[beliefs.BELIEF_HEADS[2]]
                )
            ) + tf.reduce_mean(tf.nn.softplus(outputs[beliefs.BELIEF_HEADS[1]]))
            loss -= tf.reduce_mean(
                tf.nn.log_softmax(outputs[beliefs.BELIEF_HEADS[0]])[..., 0]
            )
            loss += tf.add_n(
                list(
                    beliefs.completion_loss_terms(
                        {
                            "round_completion_yaku": tf.zeros(
                                (1, 4, 2, 23), tf.int32
                            ),
                            "round_completion_score": tf.zeros(
                                (1, 4, 2), tf.int32
                            ),
                        },
                        outputs,
                    ).values()
                )
            )
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
    restored = beliefs.Beliefs()
    restored(features, hidden=hidden, rivers=rivers)
    restored.load_weights(weights)
    for name in beliefs.BELIEF_HEADS + beliefs.COMPLETION_HEADS:
        numpy.testing.assert_allclose(
            restored(features, hidden=hidden, rivers=rivers)[name],
            heads(features, hidden=hidden, rivers=rivers)[name],
            atol=1e-6,
        )


def test_bimodal_count_logits_converge_to_physical_hand_sizes():
    # The old clipped Newton update oscillated between almost empty and
    # overfull hands when the count head preferred zero or four copies.
    features = {
        "meld_tiles": tf.fill((1, 4, 4, 4), tf.constant(37, tf.int8)),
        "known_unavailable_counts": tf.constant([[1] * 14 + [0] * 20], tf.int8),
        "candidate_hand_red": tf.zeros((1, 16, 3), tf.int8),
        "public_visible_red_counts": tf.zeros((1, 3), tf.int8),
        "seat_wind": tf.constant([[0, 1, 2, 3]], tf.int8),
        "prevailing_wind": tf.constant([0], tf.int8),
        "dora_multiplicity": tf.one_hot([8], 34, dtype=tf.int8),
    }
    logits = tf.broadcast_to([0.0, -20.0, -20.0, -20.0, -5.0], (1, 3, 37, 5))
    logits = beliefs.HeldCountMask()([logits, features])
    projected = beliefs.CountProjection()([logits, features])
    counts = tf.reduce_sum(
        tf.nn.softmax(projected) * tf.range(5, dtype=tf.float32), -1
    )
    numpy.testing.assert_allclose(
        tf.reduce_sum(counts, -1), [[13] * 3], atol=1e-4
    )
    _, available, _ = beliefs.held_count_context(features)
    assert tf.reduce_max(tf.reduce_sum(counts, 1) - available) <= 1e-4
