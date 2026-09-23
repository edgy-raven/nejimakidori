"""Distant placement supervision is broad without discarding score evidence."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import rebuild
from model import model, objectives, records, replay, train


def test_final_rank_prior_respects_scores_horizon_and_unknown_labels():
    features = {
        "scheduled_hands_remaining": tf.constant([7, 7, 0, 7, 7, 1]),
        "seat_context": tf.constant(
            [
                [[score, 0] for score in scores]
                for scores in (
                    [0.25, 0.25, 0.25, 0.25],
                    [0.55, 0.15, 0.15, 0.15],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.15, 0.55, 0.15, 0.15],
                    [0.25, 0.25, 0.25, 0.25],
                )
            ]
        ),
    }
    target = objectives.final_placement_targets(
        tf.constant([0, 0, 0, -1, 0, 0]), features
    ).numpy()
    numpy.testing.assert_allclose(
        target[0], [0.2875, 0.2375, 0.2375, 0.2375], atol=1e-6
    )
    assert target[1, 0] > target[0, 0] > target[4, 0]
    numpy.testing.assert_array_equal(target[2], [1, 0, 0, 0])
    numpy.testing.assert_array_equal(target[3], 0)
    numpy.testing.assert_allclose(target.sum(1), [1, 1, 1, 0, 1, 1], atol=1e-6)
    assert target[0, 0] < target[5, 0] < target[2, 0]
    # Early outcome differences retain only five percent of their signal.
    alternate = objectives.final_placement_targets(
        tf.constant([3, 2, 0, -1, 1, 3]), features
    ).numpy()
    numpy.testing.assert_allclose(
        alternate[[0, 1, 4]] - target[[0, 1, 4]],
        0.05 * (numpy.eye(4)[[3, 2, 1]] - numpy.eye(4)[[0, 0, 0]]),
        atol=1e-6,
    )
    # Other players' ordering and a common point offset cannot change ours.
    shifted = {
        **features,
        "seat_context": tf.gather(
            features["seat_context"], [0, 3, 1, 2], axis=1
        )
        + tf.constant([0.1, 0.0]),
    }
    numpy.testing.assert_allclose(
        objectives.final_placement_targets(
            tf.constant([0, 0, 0, -1, 0, 0]), shifted
        ),
        target,
        atol=1e-6,
    )


def test_trainer_uses_one_soft_target_for_both_rank_losses():
    events = json.loads(
        (
            pathlib.Path(__file__).parents[1]
            / "fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    observations = list(replay.Replay("placement-prior", events).observations())
    features, labels, _ = records.parse_batch(
        [rebuild.serialize_observation(row, True) for row in observations[:3]]
    )
    features["scheduled_hands_remaining"] = tf.constant([7, 1, 7])
    labels["final_placement"] = tf.constant([0, 3, -1])
    learner = train.GroupedModel(model.build_model())
    outputs = learner(features)
    numpy.testing.assert_array_equal(outputs["payoff_action_values"], 0)
    numpy.testing.assert_array_equal(outputs["payoff_baseline"], 0)
    outputs["final_placement_logits"] = tf.Variable(tf.zeros((3, 4)))
    target = objectives.final_placement_targets(
        labels["final_placement"], features
    )
    with tf.GradientTape(persistent=True) as tape:
        loss, metrics = learner.loss_and_metrics(
            x=features, y=labels, y_pred=outputs
        )
        expected_distance = tf.reduce_mean(
            objectives.ordered_distance(
                target[:2], outputs["final_placement_logits"][:2], tf.range(4)
            )
        )
        expected = expected_distance + tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=target[:2], logits=outputs["final_placement_logits"][:2]
            )
        )
    numpy.testing.assert_allclose(metrics["final_placement_loss"], expected)
    numpy.testing.assert_allclose(
        metrics["final_placement_distance_loss"], expected_distance
    )
    numpy.testing.assert_allclose(
        tape.gradient(
            metrics["final_placement_loss"], outputs["final_placement_logits"]
        ),
        tape.gradient(expected, outputs["final_placement_logits"]),
        atol=1e-6,
    )
    numpy.testing.assert_allclose(
        loss, learner.loss_terms(x=features, y=labels, y_pred=outputs)[0]
    )

    learner.advice_strength = 0.0
    _, without_advice = learner.loss_and_metrics(
        x=features, y=labels, y_pred=outputs
    )
    numpy.testing.assert_allclose(
        without_advice["weighted_offline_payoff_loss"],
        0.5 * without_advice["offline_payoff_loss"],
    )
    for component in ("placement", "points"):
        numpy.testing.assert_allclose(
            without_advice[component + "_critic_loss"],
            metrics[component + "_critic_loss"],
        )
    numpy.testing.assert_allclose(
        loss,
        sum(
            value
            for name, value in metrics.items()
            if name.startswith("weighted_")
            and name != "weighted_offline_payoff_loss"
        ),
        rtol=1e-6,
    )
    # Specialist monetary supervision uses the same log units as its
    # baseline/advantage inputs, including known immediate danger.
    labels["specialist_outcomes"] = tf.constant([[3.2, 1.0, 1.2]] * 3)
    labels["defense_ron"] = tf.fill((3, 37), 0.8)
    labels["ron_payments"] = tf.fill((3, 3, 37), -1)
    outputs["attack_predictions"] = tf.zeros((3, 298, 3))
    outputs["defense_predictions"] = tf.zeros((3, 298, 6))
    specialist_metrics = {}
    learner.specialist_loss(features, labels, outputs, specialist_metrics)
    for name, amount in (
        ("attack_receipts_loss", 3.2),
        ("defense_dealin_loss", 1.2),
        ("defense_ron_loss", 0.8),
    ):
        numpy.testing.assert_allclose(
            specialist_metrics[name],
            (numpy.log(2) - numpy.log1p(amount)) ** 2,
            rtol=1e-5,
        )
