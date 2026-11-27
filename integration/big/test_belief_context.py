"""Conditional marginals and moments retain their distributional meaning."""

import numpy
import tensorflow as tf

from model import critic


def test_moments_weight_conditional_probabilities_and_bound_count_variance():
    probability = tf.Variable(
        tf.concat(
            [
                tf.reshape(tf.tile([[1.0, 0.0, 0.0, 0.0]], [3, 1]), [1, 12]),
                tf.reshape(
                    tf.concat(
                        [tf.zeros((1, 3, 1, 34)), tf.ones((1, 3, 3, 34))], 2
                    ),
                    [1, 408],
                ),
                tf.reshape(
                    tf.tile([[[[0.5, 0.0, 0.0, 0.0, 0.5]]]], [1, 3, 37, 1]),
                    [1, 555],
                ),
            ],
            -1,
        )
    )
    layer = critic.BeliefContext(384)
    with tf.GradientTape() as tape:
        features = layer.moments(probability)
        total = tf.reduce_sum(features)
    shanten, wait, improve, mean, variance, entropy = tf.split(
        features, [4, 34, 34, 37, 37, 1], -1
    )
    numpy.testing.assert_array_equal(shanten, [[[1, 0, 0, 0]] * 3])
    numpy.testing.assert_array_equal(wait, 0)
    numpy.testing.assert_array_equal(improve, 0)
    numpy.testing.assert_array_equal(mean, 0.5)
    numpy.testing.assert_array_equal(variance, 1)
    numpy.testing.assert_array_equal(entropy, 0)
    assert numpy.isfinite(tape.gradient(total, probability)).all()
    probability.assign(
        tf.concat(
            [
                tf.fill((1, 12), 0.25),
                probability[:, 12:420],
                tf.fill((1, 555), 0.2),
            ],
            -1,
        )
    )
    _, wait, improve, mean, variance, entropy = tf.split(
        layer.moments(probability), [4, 34, 34, 37, 37, 1], -1
    )
    numpy.testing.assert_array_equal(wait, 0)
    numpy.testing.assert_allclose(improve, 0.75)
    numpy.testing.assert_allclose(mean, 0.5)
    numpy.testing.assert_allclose(variance, 0.5)
    numpy.testing.assert_allclose(entropy, 1)


def test_shanten_readout_shares_weights_across_opponents():
    from model import beliefs

    features = {
        "meld_tiles": tf.fill((2, 4, 4, 4), tf.constant(37, tf.int8)),
        "dealer_relative_seat": tf.zeros(2, tf.int8),
        "riichi_state": tf.constant([[0, 0, 0, 2]] * 2, tf.int8),
        "opponent_discard_count": tf.constant([[0, 3, 3, 8]] * 2, tf.int8),
        "live_wall_count": tf.constant([50, 20], tf.int8),
        "last_tedashi_tile_id": tf.constant([[37, 4, 4, 27]] * 2, tf.int8),
    }
    hidden = tf.random.stateless_normal((2, 512), [7, 8])
    rivers = tf.zeros((2, 3, 256))
    layer = beliefs.OpponentShanten()
    numpy.testing.assert_array_equal(layer([hidden, features, rivers]), 0)
    optimizer = tf.keras.optimizers.SGD(0.1)
    for _ in range(2):
        with tf.GradientTape() as tape:
            logits = layer([hidden, features, rivers])
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=[[1, 1, 0], [1, 1, 0]], logits=logits
                )
            )
        gradients = tape.gradient(loss, layer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, layer.trainable_variables))
    assert all(numpy.isfinite(value).all() for value in gradients)
    assert all(numpy.any(value) for value in gradients)
    logits = layer([hidden, features, rivers]).numpy()
    numpy.testing.assert_array_equal(logits[:, 0], logits[:, 1])
    assert not numpy.array_equal(logits[:, 0], logits[:, 2])

    # Identical coarse observations can still have different river histories.
    changed = tf.tensor_scatter_nd_update(rivers, [[0, 1, 0]], [1.0])
    different = layer([hidden, features, changed]).numpy()
    numpy.testing.assert_array_equal(different[:, 0], logits[:, 0])
    assert not numpy.array_equal(different[0, 0], different[0, 1])


def test_critic_moment_residual_starts_inert_under_mixed_global_policy():
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    try:
        model = critic.PaymentCritic(
            belief_offset=384,
            shanten_offset=384 + critic.BELIEF_WIDTH + 644,
            player_width=465,
        )
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
    state = tf.Variable(
        tf.concat(
            [
                tf.random.stateless_normal((2, 384), [8, 1]),
                tf.fill((2, 12), 0.25),
                tf.fill((2, 408), 0.5),
                tf.fill((2, 555), 0.2),
                tf.random.stateless_normal(
                    (2, 4356 - 384 - critic.BELIEF_WIDTH), [8, 2]
                ),
            ],
            -1,
        )
    )
    hidden, players = model.context(state)
    assert hidden.dtype == players.dtype == tf.float32
    assert len(model.refinement) == 6
    for layer in model.refinement:
        assert layer.compute_dtype == "float32"
        numpy.testing.assert_array_equal(layer(hidden), 0)
    base = model.state(state[:, : -4 * model.player_width])
    numpy.testing.assert_array_equal(hidden, base)
    assert model.self_player.compute_dtype == "float32"
    assert model.opponent_players.compute_dtype == "float32"
    assert model.event_context.compute_dtype == "float32"
    numpy.testing.assert_array_equal(
        model.belief_context([state, players[:, 1:]]), 0
    )
    model.belief_context.projection.kernel.assign(
        tf.random.stateless_normal(
            model.belief_context.projection.kernel.shape, [8, 3]
        )
        * 0.01
    )
    updated_hidden, updated_players = model.context(state)
    numpy.testing.assert_array_equal(updated_hidden, hidden)
    numpy.testing.assert_array_equal(updated_players[:, 0], players[:, 0])
    assert numpy.any(updated_players[:, 1:] != players[:, 1:])
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(model.event_logits(state))
    gradient = tape.gradient(total, state)
    assert numpy.isfinite(gradient).all()
    assert numpy.any(gradient[:, 384 : 384 + critic.BELIEF_WIDTH])


def test_structural_wait_and_marginal_brier_use_conditional_probabilities():
    from model import beliefs

    outputs = {
        "opponent_shanten_logits": tf.constant(
            [
                [
                    [10000.0, -10000.0, -10000.0, -10000.0],
                    [-10000.0, 0.0, 1.0, 2.0],
                    [1.0, 2.0, 3.0, 4.0],
                ]
            ]
        ),
        "opponent_ukeire_by_shanten_logits": tf.random.stateless_normal(
            (1, 3, 4, 34), [91, 17]
        ),
    }
    outputs["opponent_wait_logits"] = beliefs.StructuralWait()(outputs)
    joint = tf.nn.softmax(outputs["opponent_shanten_logits"])[
        ..., None
    ] * tf.nn.sigmoid(outputs["opponent_ukeire_by_shanten_logits"])
    numpy.testing.assert_allclose(
        tf.nn.sigmoid(outputs["opponent_wait_logits"]),
        joint[..., 0, :],
        atol=1e-7,
    )
    labels = {
        "opponent_shanten": tf.constant([[0, 2, 3]]),
        "opponent_ukeire": tf.cast(
            tf.random.stateless_uniform((1, 3, 34), [91, 18]) > 0.5,
            tf.float32,
        ),
    }
    _, metrics = beliefs.ukeire_loss(labels, outputs)
    numpy.testing.assert_allclose(
        metrics["opponent_ukeire_brier"],
        tf.reduce_mean(
            tf.square(tf.reduce_sum(joint, -2) - labels["opponent_ukeire"])
        ),
        atol=1e-7,
    )
