"""Payment moments and active legal support for detached policy guidance."""

import itertools

import numpy
import tensorflow as tf

from model import critic_utility, objectives, outcomes, payment_bonus


def test_st3_uncertainty_changes_risk_preference_with_score_context():
    scores = numpy.repeat(
        [[45000, 25000, 20000, 10000], [18000, 33000, 26000, 23000]],
        2,
        axis=0,
    ).astype(numpy.float32)
    variance = numpy.tile([1500.0**2, 6000.0**2], 2).astype(numpy.float32)
    prediction = numpy.full(4, 500, numpy.float32)
    actual = critic_utility.placement_utility(
        prediction=prediction,
        variance=variance,
        scores=scores,
        remaining=tf.zeros(4),
        scale=4720.0,
    ).numpy()
    assert actual[1] < actual[0]
    assert actual[3] > actual[2]

    # Independent 24-order calculation checks the cheaper shared rank prior.
    nodes, weights = numpy.polynomial.hermite.hermgauss(17)
    delta = prediction[:, None] + numpy.sqrt(2 * variance[:, None]) * nodes
    location = (
        scores[:, None]
        + delta[..., None] * numpy.array([1, -1 / 3, -1 / 3, -1 / 3])
    ) / 4720
    rates = numpy.exp(location - location.max(axis=-1, keepdims=True))
    probability = numpy.zeros((4, 17, 4))
    for order in itertools.permutations(range(4)):
        joint = numpy.ones((4, 17))
        for place in range(3):
            joint *= rates[..., order[place]] / rates[..., order[place:]].sum(
                axis=-1
            )
        probability[..., order.index(0)] += joint
    numpy.testing.assert_allclose(
        actual,
        prediction / 1000
        + numpy.sum(
            (probability @ [125, 60, -5, -255])
            * (weights / numpy.sqrt(numpy.pi)),
            axis=-1,
        ),
        atol=0.001,
    )
    numpy.testing.assert_allclose(
        objectives.placement_prior(location, tf.fill((4, 17), 0), scale=1),
        probability,
        atol=2e-5,
    )


def test_active_policy_support_includes_response_kan_without_mixing_heads():
    features = {
        "decision_phase": tf.constant([0, 1, 2, 3]),
        "trigger_tile_id": tf.constant([0, 4, 0, 0], tf.int8),
    }
    outputs = {
        name: tf.zeros((4, size))
        for name, size in (
            ("discard_policy_logits", 37),
            ("riichi_policy_logits", 74),
            ("response_policy_logits", 150),
            ("kan_action_logits", 35),
        )
    }
    selected = tf.cast(tf.where(tf.ones((4, 295), tf.bool)), tf.int32)
    active = critic_utility.active_candidates(features, selected)
    selected = tf.boolean_mask(selected, active)
    probability = critic_utility.candidate_probabilities(features, outputs)
    numpy.testing.assert_array_equal(
        numpy.bincount(selected[:, 0]), [37, 150, 35, 74]
    )
    numpy.testing.assert_allclose(
        tf.math.unsorted_segment_sum(
            tf.gather_nd(probability, selected), selected[:, 0], 4
        ),
        1,
        atol=2e-6,
    )
    numpy.testing.assert_allclose(probability[1, 265], 1 / 150)
    assert probability[1, 260] == 0
    assert numpy.all(
        probability.numpy()[
            ~tf.scatter_nd(
                selected, tf.ones(len(selected), tf.bool), (4, 295)
            ).numpy()
        ]
        == 0
    )


def test_net_variance_accounts_for_correlated_incoming_and_outgoing_payments():
    cells = numpy.asarray(payment_bonus.CELLS)
    coefficient = (
        (cells[:, 2] == 0).astype(float)
        - (cells[:, 1] == 0)
        + ((cells[:, 0] == 3) & (cells[:, 1] == 0))
    )
    incoming = payment_bonus.CELLS.index((0, 1, 0))
    outgoing = payment_bonus.CELLS.index((0, 0, 1))
    probability = numpy.zeros((1, 40, 51), numpy.float32)
    probability[:, :, 1] = 1
    amount = int(numpy.flatnonzero(outcomes.PAYMENT_VALUES == 8000)[0])
    probability[:, [incoming, outgoing], 1] = 0.5
    probability[:, [incoming, outgoing], amount] = 0.5
    prediction, variance = critic_utility.point_moments(
        tf.constant(probability), tf.eye(19)
    )
    numpy.testing.assert_allclose(prediction, 0)
    numpy.testing.assert_allclose(variance, 32000000)
    correlation = numpy.eye(19, dtype=numpy.float32)
    positions = numpy.flatnonzero(coefficient)
    incoming_position = int(numpy.flatnonzero(positions == incoming)[0])
    outgoing_position = int(numpy.flatnonzero(positions == outgoing)[0])
    correlation[incoming_position, outgoing_position] = 1
    correlation[outgoing_position, incoming_position] = 1
    _, variance = critic_utility.point_moments(
        tf.constant(probability), tf.constant(correlation)
    )
    numpy.testing.assert_allclose(variance, 0, atol=1)
