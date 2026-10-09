"""Swapping discard order preserves a route, not a counterfactual return."""

import json
import pathlib

import numpy

from log_dataset import shanten
from log_dataset import records, replay, sakigiri_regret


def test_exchange_keeps_completion_without_phantom_or_reacquired_tiles():
    # 123m 123p 123s 45m EES: record E now, draw 6m, then discard S.
    # Swapping S and E reaches the identical E-tanki hand; the intermediate
    # EE pair improves rather than damages structural readiness.
    before = numpy.bincount(
        [0, 1, 2, 9, 10, 11, 18, 19, 20, 3, 4, 27, 27, 28], minlength=37
    )
    after = before - numpy.eye(37, dtype=int)[27]
    final = after + numpy.eye(37, dtype=int)[5]
    steps = [
        {
            "counts": before,
            "after": after,
            "cut": 27,
            "legal": [27, 28],
            "distance": shanten.calculate_shanten(after[:34]),
        }
    ]
    assert sakigiri_regret.preserves_completion(steps, final, 28)
    numpy.testing.assert_array_equal(
        before
        - numpy.eye(37, dtype=int)[28]
        + numpy.eye(37, dtype=int)[5]
        - numpy.eye(37, dtype=int)[27],
        final - numpy.eye(37, dtype=int)[28],
    )
    assert (
        shanten.calculate_shanten((final - numpy.eye(37, dtype=int)[28])[:34])
        == 0
    )
    assert not sakigiri_regret.preserves_completion(steps, final, 27)
    steps[0]["legal"] = [27]
    assert not sakigiri_regret.preserves_completion(steps, final, 28)
    steps[0]["legal"] = [27, 28]
    # A later new S is not the original retained liability.
    assert not sakigiri_regret.preserves_completion(
        steps, final + numpy.eye(37, dtype=int)[28], 28
    )
    released = after + numpy.eye(37, dtype=int)[5]
    steps.append(
        {
            "counts": released,
            "after": released - numpy.eye(37, dtype=int)[28],
            "cut": 28,
            "legal": [27, 28],
            "distance": 0,
        }
    )
    # Even with one S at the endpoint, a release followed by a new draw
    # is not a continuously retained copy.
    assert not sakigiri_regret.preserves_completion(steps, released, 28)


def test_exchange_rejects_delayed_hand_progress():
    # Swapping out 1m instead of S breaks the 123m sequence.
    before = numpy.bincount(
        [0, 1, 2, 9, 10, 11, 18, 19, 20, 3, 4, 27, 27, 28], minlength=37
    )
    after = before - numpy.eye(37, dtype=int)[28]
    steps = [
        {
            "counts": before,
            "after": after,
            "cut": 28,
            "legal": [0, 28],
            "distance": shanten.calculate_shanten(after[:34]),
        }
    ]
    assert not sakigiri_regret.preserves_completion(
        steps, after + numpy.eye(37, dtype=int)[5], 0
    )


def test_verified_order_labels_reach_only_the_early_discard_records():
    # Public log 2024050601gm-00a9-0000-ce829462, E4-1. North first,
    # then 6s deals in at first tenpai; 6s first preserves the same live hand.
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/sakigiri_swap.json")
        .read_text()
    )
    pairs = sakigiri_regret.round_pairs("swap", events)
    assert pairs == {
        22: ([23, 1], 1 / 3),
        30: ([23, 30], 1 / 3),
        38: ([23, 30], 1 / 3),
    }
    rows = list(replay.Replay("swap", events).observations())
    selected = [row for row in rows if row.sakigiri_weight > 0]
    assert len(selected) == 3
    for row in selected:
        assert row.features["decision_phase"] == 0
        assert row.discard_action == row.sakigiri_pair[1]
        assert row.features["discard_action_legal"][row.sakigiri_pair].all()
    assert numpy.isclose(sum(row.sakigiri_weight for row in selected), 1)
    for row in rows:
        if row.sakigiri_weight == 0:
            numpy.testing.assert_array_equal(row.sakigiri_pair, [-1, -1])
    parsed = records.parse_batch(
        [records.serialize_observation(row, False) for row in selected]
    )
    numpy.testing.assert_array_equal(
        parsed[1]["sakigiri_pair"], [row.sakigiri_pair for row in selected]
    )
    numpy.testing.assert_allclose(parsed[1]["sakigiri_weight"], 1 / 3)
