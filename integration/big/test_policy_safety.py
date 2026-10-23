"""Hard honor proofs and late noten priors share actor/critic safety."""

import numpy
import tensorflow as tf

from log_dataset import _native_collection
from model import actions
from model import actor as actor_layers


def test_fourth_honor_requires_an_independent_kokushi_blocker():
    inputs = {
        name: numpy.zeros((4, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["known_unavailable_counts"][:, 29] = 4
    inputs["known_unavailable_counts"][1, 0] = 4
    inputs["known_unavailable_counts"][2, 28] = 4
    inputs["known_unavailable_counts"][3, 4] = 4  # Middle tiles do not block.
    inputs["meld_valid"][0, 2, 0] = 1
    inputs["genbutsu_to_seat"][:, 1, 16] = 1
    safe = actions.guaranteed_safe_tiles(inputs).numpy()
    numpy.testing.assert_array_equal(
        safe[:, :, 29],
        [[False, True, False], [True] * 3, [True] * 3, [False] * 3],
    )
    assert safe[:, 0, 16].all()
    # Four accounted-for 1m still permits a 23m sequence wait.
    assert not safe[1, :, 0].any()
    assert not safe[3, :, 4].any()
    inputs["known_unavailable_counts"][1, 29] = 3
    assert not actions.guaranteed_safe_tiles(inputs)[1, :, 29].numpy().any()


def test_late_noten_ramp_equal_safe_override_and_exceptions():
    inputs = {
        name: numpy.zeros((7, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    # West complaint: safe 1s and fourth West; unsafe 8p. No tenpai choice.
    inputs["known_unavailable_counts"][:, [18, 29]] = 4
    inputs["genbutsu_to_seat"][:, 1:, 18] = 1
    inputs["genbutsu_to_seat"][:, 0, 4] = 1  # Nagashi already impossible.
    inputs["discard_action_legal"][:, [16, 18, 29]] = 1
    inputs["discard_action_all_shanten"][:] = 2  # Stored shanten + 1.
    inputs["discard_action_all_shanten"][:, 29] = 3
    inputs["opponent_discard_count"][:] = 16
    inputs["live_wall_count"][:] = [9, 8, 4, 0, 0, 0, 0]
    inputs["discard_action_all_shanten"][4, 16] = 1  # Can secure tenpai.
    inputs["genbutsu_to_seat"][5, 1:] = 0
    inputs["known_unavailable_counts"][5] = 0  # No proven-safe option.
    inputs["genbutsu_to_seat"][6, 1:, 16] = 1  # Safe but forfeits nagashi.
    inputs["genbutsu_to_seat"][6, 0] = 0
    inputs["genbutsu_to_seat"][6, 0, 27] = 1  # Possible nagashi.
    inputs = {name: tf.constant(value) for name, value in inputs.items()}
    logits = tf.Variable(numpy.tile(numpy.arange(37), (7, 1)), dtype=tf.float32)
    policy = actor_layers.PolicyMask("discard_policy_logits")
    with tf.GradientTape() as tape:
        result = policy([logits, inputs])
        probabilities = tf.nn.softmax(result)
        loss = -tf.reduce_sum(tf.nn.log_softmax(result)[:, 18])
    assert numpy.isfinite(tape.gradient(loss, logits)).all()
    original = tf.nn.softmax(
        tf.where(inputs["discard_action_legal"] > 0, logits, -1e4)
    ).numpy()
    target = numpy.zeros(37)
    target[[18, 29]] = 0.5
    numpy.testing.assert_allclose(probabilities[:2], original[:2])
    numpy.testing.assert_allclose(probabilities[2], original[2])
    with tf.GradientTape() as tape:
        prior_loss = actor_layers.PolicyPrior("discard_policy_logits")(
            [logits, inputs]
        )
    gradient = tape.gradient(prior_loss, logits).numpy()
    numpy.testing.assert_allclose(
        prior_loss.numpy()[[0, 1, 4, 5]], 0, atol=1e-6
    )
    assert 0 < prior_loss[2] < prior_loss[3]
    assert gradient[3, 16] > 0
    assert gradient[3, 18] < 0 < gradient[3, 29]
    assert numpy.isfinite(gradient).all()
    numpy.testing.assert_array_equal(probabilities[3], target)
    numpy.testing.assert_allclose(probabilities[4:6], original[4:6])
    numpy.testing.assert_array_equal(probabilities[6], target)
    traced = tf.function(lambda x: policy([x, inputs]))(logits)
    numpy.testing.assert_allclose(traced, result, atol=1e-5)
    # Dama uses the same prior; a nonexistent riichi branch stays illegal.
    riichi = actor_layers.PolicyMask("riichi_policy_logits")(
        [tf.concat([logits, logits], axis=1), inputs]
    )
    numpy.testing.assert_array_equal(tf.nn.softmax(riichi)[3, :37], target)
    assert not tf.nn.softmax(riichi)[3, 37:].numpy().any()
