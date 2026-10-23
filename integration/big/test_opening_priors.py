"""Opening priors favor efficiency and retain live-game symmetry semantics."""

import numpy
import tensorflow as tf

from log_dataset import _native_collection
from model import actor as actor_layers


def test_efficiency_ranking_decay_masking_and_override():
    inputs = {
        name: tf.zeros([4, *spec["shape"]], spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs.update(
        {
            "discard_action_all_shanten": tf.constant(
                [[2, 2, 2, 3, 5] + [0] * 32] * 4, tf.int8
            ),
            "discard_action_all_ukeire_count": tf.constant(
                [[20, 30, 30, 100, 100] + [0] * 32] * 4, tf.int16
            ),
            "discard_action_all_upgrade_weighted_gain": tf.constant(
                [[0, 0, 1360, 0, 0] + [0] * 32] * 4, tf.int16
            ),
            "discard_action_all_upgrade_valid": tf.constant(
                [[1] * 37] * 3 + [[0] * 37], tf.int8
            ),
            "opponent_discard_count": tf.constant(
                [[0, 0, 0, 0], [0, 2, 1, 1], [0, 5, 4, 4], [0, 0, 0, 0]],
                tf.int8,
            ),
        }
    )
    legal = tf.constant([[True] * 5 + [False] * 32] * 4)
    logits = tf.Variable(tf.zeros((4, 37)))
    with tf.GradientTape() as tape:
        biased = actor_layers.opening_discard_efficiency(logits, inputs, legal)
        loss = tf.reduce_sum(biased)
    values = biased.numpy()
    assert values[0, 2] == values[0, 1]
    assert values[0, 1] > values[0, 0] > values[0, 3]
    assert values[0, 4] == -1.5
    numpy.testing.assert_allclose(values[1], values[0] * 0.6, atol=1e-7)
    numpy.testing.assert_array_equal(values[2], 0)
    numpy.testing.assert_array_equal(values[:, 5:], 0)
    assert values[3, 1] == values[3, 2]
    numpy.testing.assert_array_equal(tape.gradient(loss, logits), 1)
    # A learned value/safety preference can outweigh even the capped penalty.
    logits.assign(tf.one_hot([4] * 4, 37) * 3)
    assert (
        numpy.argmax(
            actor_layers.opening_discard_efficiency(logits, inputs, legal)[0]
        )
        == 4
    )


def test_dealer_override_and_raw_policy_prior_gradients():
    inputs = {
        name: tf.zeros([1, *spec["shape"]], spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["candidate_hand_counts"] = tf.tensor_scatter_nd_update(
        inputs["candidate_hand_counts"], [[0, 0, 0], [0, 0, 9]], [1, 1]
    )
    inputs["discard_action_legal"] = tf.constant(
        [[1] + [0] * 8 + [1] + [0] * 27], tf.int8
    )
    inputs["riichi_discard_legal_mask"] = inputs["discard_action_legal"]
    inputs["live_wall_count"] = tf.constant([69], tf.int8)
    logits = tf.Variable(tf.one_hot([9], 37) * 2)
    discard = actor_layers.PolicyMask("discard_policy_logits")
    prior = actor_layers.PolicyPrior("discard_policy_logits")
    with tf.GradientTape() as tape:
        loss = prior([logits, inputs])
    gradient = tape.gradient(loss, logits).numpy()
    assert gradient[0, 0] < 0 < gradient[0, 9]
    numpy.testing.assert_allclose(gradient.sum(), 0, atol=1e-7)
    output = discard([logits, inputs])
    numpy.testing.assert_allclose(output[0, 0], output[0, 9])
    numpy.testing.assert_allclose(
        tf.reduce_sum(tf.exp(tf.gather(output[0], [0, 9]))),
        tf.reduce_sum(tf.exp(tf.gather(logits[0], [0, 9]))),
        rtol=1e-6,
    )
    # Non-dealers' first cuts and later dealer cuts keep raw preferences.
    for dealer, counts in ((3, [0, 1, 0, 0]), (0, [1, 1, 1, 1])):
        inputs["dealer_relative_seat"] = tf.constant([dealer], tf.int8)
        inputs["opponent_discard_count"] = tf.constant([counts], tf.int8)
        numpy.testing.assert_array_equal(
            tf.gather(discard([logits, inputs])[0], [0, 9]), [0, 2]
        )
        assert 0 < prior([logits, inputs])[0] < loss[0]
    inputs["opponent_discard_count"] = tf.constant([[2, 5, 5, 5]], tf.int8)
    numpy.testing.assert_allclose(prior([logits, inputs]), 0, atol=1e-7)
    inputs["opponent_discard_count"] = tf.zeros((1, 4), tf.int8)
    inputs["dora_multiplicity"] = tf.one_hot([0], 34, dtype=tf.int8)
    assert discard([logits, inputs])[0, 9] > discard([logits, inputs])[0, 0]
    numpy.testing.assert_allclose(prior([logits, inputs]), 0, atol=1e-7)
    # Both branches learn symmetry; unavailable riichi actions stay masked.
    inputs["dora_multiplicity"] = tf.zeros((1, 34), tf.int8)
    riichi_logits = tf.Variable(tf.concat([logits, logits + 0.7], axis=-1))
    with tf.GradientTape() as tape:
        loss = actor_layers.PolicyPrior("riichi_policy_logits")(
            [riichi_logits, inputs]
        )
    gradient = tape.gradient(loss, riichi_logits).numpy()
    assert gradient[0, 0] < 0 < gradient[0, 9]
    assert gradient[0, 37] < 0 < gradient[0, 46]
    inputs["riichi_discard_legal_mask"] = tf.one_hot([0], 37, dtype=tf.int8)
    output = actor_layers.PolicyMask("riichi_policy_logits")(
        [riichi_logits, inputs]
    )
    assert output[0, 46] == -1e4
    traced = tf.function(lambda x: prior([x, inputs]))(logits)
    numpy.testing.assert_allclose(traced, prior([logits, inputs]), atol=1e-6)
