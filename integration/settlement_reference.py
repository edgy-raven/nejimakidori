"""Expanded-settlement payoff oracle for the chunked critic implementation."""

from model import joint_settlement


def projected_st3(outputs, inputs):
    """Fixed continuation retains each draw's distinct dealer-repeat state."""
    return joint_settlement.expectation(
        outputs["joint_probability"],
        joint_settlement.continuation_payoffs(
            scores=outputs["joint_scores"],
            ranks=outputs["joint_ranks"],
            support=outputs["joint_support"],
            dealer=inputs["dealer"],
            hand_number=inputs["hand_number"],
            pot=inputs["pot"],
        ),
    )
