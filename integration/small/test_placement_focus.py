"""Placement focus uses strict score gaps and the trailing player's dealer."""

import numpy

from integration import focus_reference as focus_sampling
from log_dataset import _native_collection


def test_placement_gaps_follow_rank_with_no_future_dealerships(monkeypatch):
    monkeypatch.setattr(
        focus_sampling.scoring_reference,
        "yaku_path_fractions",
        lambda packed: numpy.zeros(len(focus_sampling.scoring.NAMES)),
    )
    # Scores are in rank order. Rotate actor-relative seats for every case;
    # dealer_rank identifies the player whose dealer threshold can matter.
    for scores, dealer_rank, expected in (
        ([41000, 30000, 20000, 9000], 3, [True, False, False, False]),
        ([41000, 30000, 20000, 9000], 2, [False, True, False, False]),
        ([41000, 30000, 20000, 9000], 4, [True, False, True, True]),
        ([39000, 30000, 20000, 11000], 1, [False, True, True, True]),
        ([31000, 22000, 20000, 15000], 1, [False, False, True, True]),
        # Exact threshold equality satisfies neither strict comparison.
        ([40000, 30000, 20000, 10000], 1, [False, False, False, False]),
        ([46000, 30000, 20000, 10000], 2, [False, False, False, False]),
        ([35000, 30000, 26000, 10000], 4, [False, True, False, False]),
        ([30000, 26000, 20000, 10000], 4, [False, False, True, True]),
        # Stored ranks resolve ties without deriving a new seat tie-break.
        ([30000, 30000, 20000, 20000], 1, [False, False, True, True]),
    ):
        for actor in range(4):
            packed = {
                name: numpy.zeros(spec["shape"], dtype=spec["dtype"])
                for name, spec in _native_collection.INPUTS.items()
            }
            order = (actor + numpy.arange(4)) % 4
            packed["seat_context"][:] = numpy.stack(
                [numpy.array(scores)[order] / 100000, (order + 1) / 4],
                axis=-1,
            )
            packed["seat_wind"][:] = (order - (dealer_rank - 1)) % 4
            packed["discard_action_legal"][:2] = 1
            packed["kyoku_number"][...] = 3
            packed["live_wall_count"][...] = 60
            assert focus_sampling.placement_gap_focus(packed) == expected[actor]
            assert (
                bool(focus_sampling.categories(packed) & 32) == expected[actor]
            )
            # East 1, including repeats, is excluded even if gaps qualify.
            packed["kyoku_number"][...] = 0
            packed["honba"][...] = 3
            assert not focus_sampling.placement_gap_focus(packed)
            assert not focus_sampling.categories(packed) & 32
            # South 1 is eligible; late-game status alone doesn't qualify.
            packed["prevailing_wind"][...] = 1
            assert focus_sampling.placement_gap_focus(packed) == expected[actor]
            packed["discard_action_legal"][1] = 0
            assert focus_sampling.categories(packed) == 0


def test_remaining_dealership_and_retained_placement_contexts(monkeypatch):
    monkeypatch.setattr(
        focus_sampling.scoring_reference,
        "yaku_path_fractions",
        lambda packed: numpy.zeros(len(focus_sampling.scoring.NAMES)),
    )
    packed = {
        name: numpy.zeros(spec["shape"], dtype=spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    packed["seat_context"][:] = [
        [0.41, 0.25],
        [0.30, 0.5],
        [0.20, 0.75],
        [0.09, 1.0],
    ]
    packed["seat_wind"][:] = [0, 1, 2, 3]
    packed["discard_action_legal"][:2] = 1
    packed["prevailing_wind"][...] = 1
    packed["kyoku_number"][...] = 1
    packed["live_wall_count"][...] = 60
    # In South 2, second place deals next: use 16k despite not being dealer.
    packed["scheduled_hands_remaining"][...] = 2
    assert not focus_sampling.placement_gap_focus(packed)
    assert not focus_sampling.categories(packed) & 32
    # A completed dealership behind the current dealer does not remain.
    packed["seat_wind"][:] = [2, 3, 0, 1]
    assert focus_sampling.placement_gap_focus(packed)
    # East 2 of a hanchan still gives everyone a scheduled dealership.
    packed["prevailing_wind"][...] = 0
    packed["scheduled_hands_remaining"][...] = 6
    assert not focus_sampling.placement_gap_focus(packed)
    # All-last retains the current dealer's opportunity, not future ones.
    packed["kyoku_number"][...] = 3
    packed["scheduled_hands_remaining"][...] = 0
    packed["seat_wind"][:] = [3, 0, 1, 2]
    assert not focus_sampling.placement_gap_focus(packed)
    packed["seat_wind"][:] = [0, 1, 2, 3]
    assert focus_sampling.placement_gap_focus(packed)
    # Equal scores make the first-place gap rule false in every context.
    packed["seat_context"][:, 0] = 0.25
    for wind, hand, wall, shanten, expected in (
        (0, 1, 60, 3, False),
        (1, 1, 60, 3, False),
        (1, 2, 60, 3, True),
        (1, 3, 60, 3, True),
        (2, 0, 60, 3, True),
        (0, 0, 16, 2, True),
        (0, 0, 16, 1, True),
        (0, 0, 17, 2, False),
        (0, 0, 16, 3, False),
    ):
        packed["prevailing_wind"][...] = wind
        packed["kyoku_number"][...] = hand
        packed["live_wall_count"][...] = wall
        for family in ("normal", "chiitoitsu", "kokushi"):
            packed[family + "_self_shanten"][...] = shanten
        assert bool(focus_sampling.categories(packed) & 32) == expected
    packed["prevailing_wind"][...] = 1
    packed["kyoku_number"][...] = 3
    packed["discard_action_legal"][1] = 0
    assert focus_sampling.categories(packed) == 0
