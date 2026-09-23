"""Independent outcome critics and detached, optional dealership advice."""

import numpy
import tensorflow as tf

import feature_vector
from model import objectives, outcomes


def test_placement_prior_is_independent_of_dealership_advice():
    features = {
        "scheduled_hands_remaining": tf.constant([7, 7, 3, 0, 0, 7]),
        "dealer_relative_seat": tf.constant([1, 0, 0, 0, 0, 0]),
        "seat_context": tf.constant(
            [
                [[score, 0] for score in scores]
                for scores in (
                    [0.25] * 4,
                    [0.25] * 4,
                    [0.25] * 4,
                    [0.55, 0.15, 0.15, 0.15],
                    [0.15, 0.55, 0.15, 0.15],
                    [0.55, 0.15, 0.15, 0.15],
                )
            ]
        ),
    }
    prior = objectives.placement_value_prior(features).numpy()
    numpy.testing.assert_allclose(prior[:3], 0.2434210526, atol=1e-6)
    numpy.testing.assert_array_equal(
        objectives.dealership_placement_weight(features), [0, 0.5, 1, 1, 1, 0.5]
    )
    assert prior[3] > prior[5] > prior[2] > prior[4]
    numpy.testing.assert_allclose(
        objectives.placement_value_prior(
            {**features, "seat_context": features["seat_context"] + 0.1}
        ),
        prior,
        atol=1e-6,
    )


def test_component_targets_alternatives_masks_and_detached_advice():
    features = {
        name: tf.zeros([3, *definition["shape"]], definition["dtype"])
        for name, definition in feature_vector.INPUTS.items()
    }
    features["seat_context"] = tf.constant([[[0.25, 0.0]] * 4] * 3)
    features["discard_tedashi_legal"] = tf.ones((3, 37), tf.int8)
    features["scheduled_hands_remaining"] = tf.constant([7, 7, 0])
    features["honba"] = tf.constant([1, 0, 0], tf.int16)
    # Row 0 has one future dealership; row 1 also has point labels but no
    # final placement. Row 2 has labels but no discretionary action.
    payment = numpy.zeros((3, 4, 4, 4), numpy.int32)
    payment[0, 0, 1, 0] = 8000
    payment[0, 3, 0, 0] = -1000
    labels = {
        "terminal_payment": tf.constant(outcomes.payment_classes(payment)),
        "final_placement": tf.constant([0, -1, 3]),
        "ron_payments": tf.tensor_scatter_nd_update(
            tf.fill((3, 3, 37), -1),
            [[0, 0, 1], [0, 1, 1], [0, 2, 1], [0, 0, 2], [0, 1, 2], [0, 2, 2]],
            [8300, 0, 0, 8300, 0, 0],
        ),
        "ron_return": tf.tensor_scatter_nd_update(
            tf.zeros((3, 37)), [[0, 1]], [-0.915]
        ),
        "ron_return_valid": tf.tensor_scatter_nd_update(
            tf.zeros((3, 37), tf.int8), [[0, 1]], [1]
        ),
        "abort_points": tf.tensor_scatter_nd_update(
            tf.fill((3, 2, 37), -1), [[0, 0, 3]], [-1000]
        ),
    }
    outputs = {}
    for name, width in zip(
        feature_vector.TRAIN_POLICY_HEADS, (38, 76, 150, 35)
    ):
        labels[name] = tf.constant(
            [0, 0, -1] if name == "discard_policy_logits" else [-1] * 3
        )
        outputs[name] = tf.Variable(tf.zeros((3, width)))
    outputs["payoff_action_values"] = tf.Variable(tf.zeros((3, 298, 2)))
    baseline = tf.Variable(tf.zeros((3, 2)))
    with tf.GradientTape(persistent=True) as tape:
        result = objectives.offline_policy_terms(
            labels=labels,
            logits=outputs,
            baseline=baseline,
            features=features,
        )
    prior = float(objectives.placement_value_prior(features)[0])
    numpy.testing.assert_allclose(
        result["placement_critic_loss"],
        ((1 - prior) ** 2 + (-1 - prior) ** 2) / 2,
        rtol=1e-6,
    )
    numpy.testing.assert_allclose(
        result["points_critic_loss"],
        (
            numpy.log1p(0.7) ** 2
            + numpy.log1p(0.8) ** 2 * 2
            + numpy.log1p(0.1) ** 2
        )
        / 8,
        rtol=1e-6,
    )
    numpy.testing.assert_allclose(
        result["points_baseline_loss"],
        numpy.log1p(0.7) ** 2 / 2,
        rtol=1e-6,
    )
    gradient = tape.gradient(
        result["payoff_critic_loss"], outputs["payoff_action_values"]
    ).numpy()
    assert gradient[0, 0, 0] < 0 < gradient[0, 1, 0]
    assert gradient[0, 0, 1] < 0 < gradient[0, 1, 1]
    assert gradient[0, 2, 1] > 0  # Known payment, unknown final placement.
    assert gradient[0, 3, 1] > 0  # Abort deposit valid at any dealership.
    numpy.testing.assert_array_equal(gradient[0, 2:, 0], 0)
    numpy.testing.assert_array_equal(gradient[1:, :, 0], 0)
    numpy.testing.assert_array_equal(gradient[2], 0)
    for loss in ("offline_payoff_loss", "dealership_advice_loss"):
        assert tape.gradient(result[loss], baseline) is None
        assert (
            tape.gradient(result[loss], outputs["payoff_action_values"]) is None
        )
        assert (
            tape.gradient(result[loss], outputs["discard_policy_logits"])
            is not None
        )
    # Changing only point predictions changes advice, never placement AWR.
    outputs["payoff_action_values"].assign(
        tf.tensor_scatter_nd_update(
            outputs["payoff_action_values"], [[0, 0, 1]], [2.0]
        )
    )
    changed = objectives.offline_policy_terms(
        labels=labels,
        logits=outputs,
        baseline=baseline,
        features=features,
        advice_strength=0,
    )
    assert float(changed["payoff_awr_effective_sample_fraction"]) == 1
    assert float(changed["advice_awr_effective_sample_fraction"]) < 1
    assert float(changed["dealership_advice_strength"]) == 0
    numpy.testing.assert_allclose(
        changed["placement_critic_loss"], result["placement_critic_loss"]
    )


def test_signed_log_payment_distance_compresses_large_errors():
    # Losses and gains remain ordered, zero is finite, and high payouts
    # contribute less than on the old linear scale.
    amounts = numpy.array(
        [-96000, -12000, -8000, -1000, 0, 1000, 8000, 12000, 96000],
        numpy.float32,
    )
    transformed = objectives.signed_log(amounts / 10000).numpy()
    assert numpy.all(numpy.diff(transformed) > 0)
    numpy.testing.assert_array_equal(transformed, -transformed[::-1])
    numpy.testing.assert_allclose(
        transformed[6:8], [0.5877867, 0.7884574], rtol=1e-6
    )
    assert transformed[-1] < amounts[-1] / 10000
    # Categorical CE and occurrence still use the exact payment classes.
    # Both CDF terms measure the signed-log gap between known nonzero bins.
    values = outcomes.PAYMENT_VALUES
    target = tf.one_hot([list(values).index(12000)], len(values))
    logits = tf.Variable(
        tf.where(
            tf.one_hot(
                [list(values).index(8000)],
                len(values),
                on_value=True,
                off_value=False,
            ),
            40.0,
            -40.0,
        )
    )
    with tf.GradientTape() as tape:
        terms = objectives.payment_terms(target, logits)
        loss = tf.add_n(list(terms.values()))
    expected = numpy.log1p(1.2) - numpy.log1p(0.8)
    for name in ("payment_distance_loss", "payment_conditional_distance_loss"):
        numpy.testing.assert_allclose(terms[name], [expected], rtol=1e-6)
    numpy.testing.assert_allclose(terms["payment_cross_entropy"], [80.0])
    numpy.testing.assert_allclose(
        terms["payment_occurrence_loss"], [0.0], atol=1e-30
    )
    assert numpy.isfinite(tape.gradient(loss, logits)).all()


def test_entropy_bonus_flattens_only_active_legal_choices():
    labels = {
        name: (
            tf.constant([0, -1])
            if name == "discard_policy_logits"
            else tf.constant([-1, -1])
        )
        for name in feature_vector.TRAIN_POLICY_HEADS
    }
    logits = {
        name: tf.Variable([[2.0, 0.0, -1e4], [2.0, 0.0, -1e4]])
        for name in feature_vector.TRAIN_POLICY_HEADS
    }
    with tf.GradientTape() as tape:
        loss = -objectives.POLICY_ENTROPY_WEIGHT * objectives.policy_entropy(
            labels, logits
        )
    gradient = tape.gradient(loss, logits["discard_policy_logits"]).numpy()
    assert float(loss) < 0
    assert gradient[0, 0] > 0 > gradient[0, 1]
    numpy.testing.assert_array_equal(gradient[:, 2], 0)
    numpy.testing.assert_array_equal(gradient[1], 0)


def test_tenpai_quality_baseline_uses_joint_outcome_and_masks_unknowns():
    reached = tf.constant([0.0, 1.0, 1.0, -1.0])
    quality = tf.constant([-1.0, 3.0, -1.0, 3.0])
    reward = objectives.tenpai_quality_reward(reached, quality)
    numpy.testing.assert_allclose(reward, [0.0, 0.75, -1.0, -1.0])
    baseline = tf.Variable(tf.zeros((4, 5)))
    rewards = tf.stack([reached, reached, reached, reward, reward], axis=-1)
    with tf.GradientTape() as tape:
        _, loss = objectives.expert_reward_terms(
            rewards, tf.zeros((4, 2, 5)), baseline, tf.ones(4, tf.bool)
        )
    gradient = tape.gradient(loss, baseline).numpy()
    numpy.testing.assert_array_equal(gradient[0, 3:], 0)
    assert numpy.all(gradient[1, 3:] < 0)
    numpy.testing.assert_array_equal(gradient[2:, 3:], 0)
