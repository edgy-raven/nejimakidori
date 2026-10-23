"""Joint training preserves AdamW math for dense and sparse parameters."""

import numpy
import tensorflow as tf

from model import actions
from model import actor as actor_layers
from model import optimization


def test_offline_schedule_warms_up_and_decays():
    schedule = optimization.OfflineSchedule(total_steps=1000)
    numpy.testing.assert_allclose(
        [float(schedule(step)) for step in (0, 500, 1000)],
        [2e-5, 2e-4, 2e-5],
        rtol=1e-6,
    )


def test_awr_ess_matches_effective_weights_and_survives_tiny_advantages():
    for values in (
        numpy.log([1.0, 2.0, 100.0, 1e10]),
        numpy.array([-1000.0, -1001.0, -1002.0, 0.0]),
    ):
        nll = tf.Variable([1.0, 2.0, 4.0, 100.0])
        advantages = tf.Variable(values, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            terms = actor_layers.awr_policy_terms(
                nll, advantages, tf.constant([True, True, True, False])
            )
        weights = numpy.exp(
            numpy.minimum(values[:3], numpy.log(20))
            - numpy.minimum(values[:3], numpy.log(20)).max()
        )
        weights /= weights.sum()
        numpy.testing.assert_allclose(
            terms["loss"], weights @ [1, 2, 4], rtol=1e-6
        )
        numpy.testing.assert_allclose(
            terms["effective_sample_fraction"],
            1 / (3 * numpy.square(weights).sum()),
            rtol=1e-6,
        )
        numpy.testing.assert_allclose(
            tape.gradient(terms["loss"], nll), [*weights, 0], rtol=1e-6
        )
        assert tape.gradient(terms["loss"], advantages) is None


def test_conditional_call_values_do_not_reweight_response_imitation(
    monkeypatch,
):

    from model import critic

    features = {
        "decision_phase": tf.constant([0, 1, 1, 1, 1]),
        "trigger_tile_id": tf.zeros(5, tf.int32),
    }
    labels = {name: tf.fill([5], -1) for name in actions.TRAIN_POLICY_HEADS}
    labels["discard_policy_logits"] = tf.constant([0, -1, -1, -1, -1])
    labels["response_policy_logits"] = tf.constant([-1, 0, 1, 112, 149])
    selected = tf.constant(
        [
            [0, 0],
            [0, 1],
            [1, 111],
            [1, 112],
            [2, 112],
            [2, 111],
            [3, 223],
            [3, 111],
            [4, 261],
            [4, 111],
        ]
    )
    monkeypatch.setattr(
        critic,
        "action_values",
        lambda network, features, outputs, state: (
            selected,
            state,
            tf.reduce_mean(tf.reshape(state, [5, 2]), 1),
        ),
    )
    for scale in (1.0, -10.0):
        values = tf.Variable(
            [6, -6, *[scale * x for x in (12, -12, -12, 12, 30, -30, -30, 30)]],
            dtype=tf.float32,
        )
        nll = tf.Variable(tf.ones(5))
        with tf.GradientTape(persistent=True) as tape:
            advantages = critic.advantages.python_function(
                None,
                (features, labels),
                {"critic_candidate_state": values},
                3.0,
            )
            terms = actor_layers.awr_policy_terms(
                nll, advantages, tf.ones(5, tf.bool)
            )
        numpy.testing.assert_array_equal(advantages, [2, 0, 0, 0, 0])
        numpy.testing.assert_allclose(
            tape.gradient(terms["loss"], nll),
            numpy.array([numpy.exp(2), 1, 1, 1, 1]) / (numpy.exp(2) + 4),
            rtol=1e-6,
        )
        assert tape.gradient(terms["loss"], values) is None
