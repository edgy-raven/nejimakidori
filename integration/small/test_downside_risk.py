"""Risk counts net losing hands, not winning upside or negative match score."""

import numpy
import tensorflow as tf

from model import critic, joint_settlement


def test_downside_uses_loss_probability_and_below_mean_shortfall(monkeypatch):
    probability = tf.constant(
        [
            [[0.25], [0.25], [0.5]],
            [[0.25], [0.25], [0.5]],
            [[0.0], [0.5], [0.5]],
        ]
    )
    payoff = tf.constant(
        [
            [[-10.0], [50.0], [100.0]],
            [[-10.0], [50.0], [100.0]],
            [[0.0], [50.0], [70.0]],
        ]
    )
    # Row 0: a positive-ST3 loss counts, a negative-ST3 win does not.
    # Row 1: all wins, despite substantial outcome variance.
    # Row 2: doubling below-mean loss probability doubles downside variance;
    # the remaining above-mean loss contributes zero.
    scores = tf.constant(
        [
            [[[1000.0]], [[-1000.0]], [[0.0]]],
            [[[1000.0]], [[1000.0]], [[1000.0]]],
            [[[1000.0]], [[-1000.0]], [[-1000.0]]],
        ]
    )
    monkeypatch.setattr(critic.PaymentCritic, "distributions", lambda *args: {})
    monkeypatch.setattr(
        joint_settlement, "probabilities", lambda inputs: (probability, None)
    )
    monkeypatch.setattr(
        joint_settlement, "position_payoffs", lambda inputs: payoff
    )
    monkeypatch.setattr(
        joint_settlement,
        "settlement_states",
        lambda inputs: {"joint_scores": scores},
    )
    network = critic.PaymentCritic(belief_offset=0, player_width=8)
    inputs = {"scores": tf.zeros([3, 1])}
    expected = [[60.0, 5.0], [60.0, 0.0], [60.0, numpy.sqrt(50.0)]]
    numpy.testing.assert_allclose(
        network.projected_mean_downside(inputs), expected, atol=1e-6
    )
    numpy.testing.assert_allclose(
        network.projected_mean_downside(inputs, payoff), expected, atol=1e-6
    )
    # Diagnostics retain raw ST3 moments, not 100 times the new objective.
    objective = tf.constant(
        [[[-2.0], [1.0], [2.0]], [[-2.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]]]
    )
    numpy.testing.assert_allclose(
        network.projected_mean_downside(
            inputs, tf.stack([objective, payoff], axis=-1)
        ),
        [
            [0.75, 0, 60, 5],
            [0.75, 0, 60, 0],
            [1.5, numpy.sqrt(0.125), 60, numpy.sqrt(50)],
        ],
        atol=1e-6,
    )
