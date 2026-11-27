"""Expanded-settlement payoff oracle for the chunked critic implementation."""

import tensorflow as tf

from log_dataset import scoring
from model import joint_settlement, losses, payment_scoring


def projected_st3(outputs, inputs):
    """Fixed continuation retains each draw's distinct dealer-repeat state."""
    return joint_settlement.expectation(
        outputs["joint_probability"],
        joint_settlement.continuation_forecast(
            scores=outputs["joint_scores"],
            ranks=outputs["joint_ranks"],
            support=outputs["joint_support"],
            dealer=inputs["dealer"],
            hand_number=inputs["hand_number"],
            pot=inputs["pot"],
        )[0],
    )


def expanded_terms(labels, outputs, mean=losses.mean_valid):
    """Exact observed-payment likelihood over expanded settlement support.

    Liability/nagashi allocations outside the ordinary support receive no
    settlement supervision. Coverage is reported.
    """
    same_transfer = tf.reduce_all(
        (
            outputs["joint_classes"]
            == tf.cast(labels["payment"][:, None, :36], tf.int64)
        )
        | (
            (labels["payment"][:, None, :36] < 0)
            & (outputs["joint_classes"] != 1)
        ),
        -1,
    )
    actual_bank = tf.gather(
        joint_settlement.payment_values(labels),
        labels["payment"][:, 36:],
        axis=1,
        batch_dims=1,
    )
    matches = same_transfer[:, :, None] & tf.reduce_all(
        outputs["joint_bank"] == actual_bank[:, None, None], -1
    )
    matches &= outputs["joint_probability"] > 0
    if "event" in labels:
        matches &= (
            outputs["joint_support"][:, :, 5]
            == tf.cast(labels["event"][:, None], tf.int64)
        )[..., None]
    supported = tf.reduce_any(matches, axis=(1, 2))
    log_likelihood = tf.reduce_logsumexp(
        tf.where(
            matches,
            tf.math.log(tf.maximum(outputs["joint_probability"], 1e-30)),
            -1e30,
        ),
        axis=(1, 2),
    )
    return {
        "payment_nll": mean(-log_likelihood, supported),
        "settlement_multi_ron_fraction": tf.reduce_mean(
            tf.cast(
                tf.reduce_sum(
                    tf.cast(
                        tf.linalg.matmul(
                            tf.cast(labels["payment"][:, :12] != 1, tf.float32),
                            tf.one_hot(
                                [cell[2] for cell in scoring.CELLS[:12]],
                                4,
                            ),
                        )
                        > 0,
                        tf.int32,
                    ),
                    -1,
                )
                > 1,
                tf.float32,
            )
        ),
        "settlement_supported_fraction": tf.reduce_mean(
            tf.cast(supported, tf.float32)
        ),
    }


def inputs():
    return {
        "dealer": tf.constant([0]),
        "score_probability": tf.ones((1, 24, len(payment_scoring.SCORES)))
        / len(payment_scoring.SCORES),
        "event_logits": tf.zeros((1, len(joint_settlement.EVENTS))),
        "ending_draw_kind": tf.zeros(1, tf.int32),
        "ending_tenpai": tf.fill((1, 4), -1),
        "ending_safe": tf.zeros((1, 3), tf.bool),
        "deposit_required": tf.constant([[False] * 4]),
        "riichi_required": tf.zeros((1, 4), tf.bool),
        "tenpai_required": tf.zeros((1, 4), tf.bool),
        "action_tile": tf.fill((1,), 37),
        "pot": tf.constant([14000.0]),
        "honba": tf.constant([2.0]),
        "scores": tf.constant([[24000.0, 24000, 19000, 19000]]),
        "first_dealer": tf.constant([2]),
        "hand_number": tf.constant([7]),
        "deposit_allowed": tf.constant([[True, True, False, True]]),
        "deposit_logits": tf.zeros((1, len(joint_settlement.EVENTS), 16)),
    }
