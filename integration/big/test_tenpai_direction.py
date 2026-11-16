"""Settlement responds monotonically to readiness and candidate-tile waits."""

import numpy
import tensorflow as tf
from integration.small.test_joint_settlement import inputs
from model import critic, joint_settlement


def test_reversed_belief_effects_train_settlement_without_moving_beliefs():
    public = {
        name: tf.repeat(value, 2, axis=0) for name, value in inputs().items()
    }
    public["action_tile"] = tf.constant([31, 37])
    public["state"] = tf.Variable(
        tf.concat(
            [tf.fill((2, 12), 0.25), tf.fill((2, 408), 0.5), tf.ones((2, 92))],
            axis=1,
        )
    )
    model = critic.PaymentCritic(belief_offset=0)
    model.event(model.state(public["state"]))
    # Route opponent 1's tenpai and conditional tile-31 wait into one hidden
    # feature, then make its winning/ready outcomes decrease as that grows.
    state_kernel = numpy.zeros(model.state.kernel.shape, numpy.float32)
    state_kernel[[0, 12 + 31], 0] = 1
    model.state.kernel.assign(state_kernel)
    model.state.bias.assign(tf.zeros_like(model.state.bias))
    event_kernel = numpy.zeros(model.event.kernel.shape, numpy.float32)
    event_kernel[0] = [
        2
        * bool(
            (kind in (0, 1) and winners & 2 and winners.bit_count() < 3)
            or (kind == 2 and ready & 2)
        )
        for kind, _, winners, ready in joint_settlement.EVENTS
    ]
    model.event.bias.assign(tf.zeros_like(model.event.bias))
    for direction in (-1, 1):
        model.event.kernel.assign(direction * event_kernel)
        with tf.GradientTape(persistent=True) as tape:
            penalties = model.direction_losses(public)
            total = tf.reduce_sum(penalties)
        assert tape.gradient(total, public["state"]) is None
        assert float(penalties[1, 1]) == 0  # No candidate tile means no wait.
        if direction < 0:
            assert numpy.all(penalties.numpy()[:, 0] > 0)
            assert float(penalties[0, 1]) > 0
            for variables in (
                model.state.trainable_variables,
                model.event.trainable_variables,
            ):
                assert (
                    float(
                        tf.linalg.global_norm(tape.gradient(total, variables))
                    )
                    > 0
                )
        else:
            numpy.testing.assert_allclose(penalties, 0, atol=1e-7)
