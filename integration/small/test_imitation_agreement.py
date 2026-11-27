"""Human and Mortal agreement use each target's joint action and call branch."""

import numpy
import tensorflow as tf

from model import actions, actor
from training import evaluate_imitation


def test_joint_decisions_branch_mass_and_teacher_conditional_cut():
    # Pass is the largest single action. Chi has greater total probability.
    response = [0.40, 0.35, 0.25]
    outputs = {
        name: tf.zeros((3, actions.ACTOR_OUTPUTS[name][0]))
        for name in actions.TRAIN_POLICY_HEADS
    }
    outputs["response_policy_logits"] = tf.pad(
        tf.math.log([response] * 3), [[0, 0], [0, 147]]
    )
    batch = {
        "head": tf.constant([2, 2, 0]),
        "legal": tf.pad(
            tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]]) > 0,
            [[0, 0], [0, 147]],
        ),
    }
    log_policy = actor.distribution(outputs, batch["head"], batch["legal"])
    expert = evaluate_imitation.summarize(
        evaluate_imitation.decision_totals(
            batch, log_policy, tf.constant([0, 0, 0]), tf.ones(3, tf.bool)
        ).numpy()
    )
    mortal = evaluate_imitation.summarize(
        evaluate_imitation.decision_totals(
            batch,
            log_policy,
            tf.constant([1, 0, 1]),
            tf.constant([True, True, False]),
        ).numpy()
    )
    assert expert["all"]["top1"] == 1
    assert expert["all"]["matches"] == 3 and expert["all"]["errors"] == 0
    assert expert["response_policy"]["branch_top1"] == 0.5
    assert expert["response_policy"]["conditional_discard_top1"] is None
    assert mortal["all"]["top1"] == 1 / 3
    assert mortal["all"]["matches"] == 1 and mortal["all"]["errors"] == 2
    assert mortal["all"]["supported_decisions"] == 2
    numpy.testing.assert_allclose(
        mortal["all"]["supported_nll"], -numpy.log(0.35) / 2
    )
    assert mortal["response_policy"]["branch_top1"] == 1
    assert mortal["response_policy"]["conditional_discard_top1"] == 1
    assert mortal["kan_action"]["top1"] is None
