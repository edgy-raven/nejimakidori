"""Native draw payments charge accepted riichi deposits exactly once."""

import numpy
import pytest

from model import outcomes


@pytest.mark.parametrize(
    "reason,accepted,deltas,corrected",
    [
        (
            "nagashimangan",
            [1],
            [-2000, -5000, 8000, -2000],
            [-2000, -4000, 8000, -2000],
        ),
        (
            "nagashimangan",
            [3, 2],
            [8000, -2000, -5000, -3000],
            [8000, -2000, -4000, -2000],
        ),
        ("exhaustive_draw", [1], [0, -1000, 0, 0], [0, 0, 0, 0]),
    ],
)
def test_draws_charge_accepted_deposits_once(
    reason, accepted, deltas, corrected
):
    # Recorded draw-payment failures: final nagashi, two deposits, all-tenpai.
    events = [
        {"type": "start_kyoku", "scores": [25000] * 4, "honba": 0},
        *[{"type": "reach_accepted", "actor": seat} for seat in accepted],
        {"type": "ryukyoku", "reason": reason, "deltas": deltas},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    outcomes.normalize_draw_payments(events)
    assert events[-3]["deltas"] == corrected
    payment = outcomes.terminal_payments(events)
    outcomes.payment_classes(payment)
    numpy.testing.assert_array_equal(
        outcomes.terminal_scores(events),
        25000 + outcomes.settlement(payment)[:4],
    )
    assert sum(outcomes.terminal_scores(events)) == 100000 - 1000 * len(
        accepted
    )
    outcomes.normalize_draw_payments(events)
    assert events[-3]["deltas"] == corrected
    events[-3]["deltas"][0] -= 100
    with pytest.raises(ValueError, match="Invalid native draw payment"):
        outcomes.normalize_draw_payments(events)
