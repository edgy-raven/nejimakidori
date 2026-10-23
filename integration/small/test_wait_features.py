"""Wait summaries describe obtainable wins; furiten remains structural."""

import numpy

from integration import scoring_reference
from log_dataset import scoring


def test_wait_values_exclude_exhausted_red_and_base_tiles():
    # 123456m 789p 67s 22p: pinfu waiting on 5s/8s, red 5s exposed.
    tiles = [0, 4, 8, 12, 17, 20, 60, 64, 68, 92, 96, 40, 41]
    context = dict(
        missed_agari=False,
        own_discards=[],
        unavailable_counts=numpy.bincount(
            [tile // 4 for tile in tiles] + [22], minlength=34
        ),
        unavailable_red=[0, 0, 1],
        dora_indicators=[],
        player_wind=1,
        round_wind=0,
    )
    summary, rows = scoring.score_waits(tiles, [], context)
    numpy.testing.assert_array_equal(summary[:, 0], 2)
    numpy.testing.assert_array_equal(summary[:, 1], 7)
    numpy.testing.assert_array_equal(
        summary[:, 2:5],
        [
            [1000, 1000, 1000],
            [1500, 1500, 1500],
            [2000, 2000, 2000],
            [2700, 2700, 2700],
        ],
    )
    # Raw outcome rows retain zero-availability entries for target consumers.
    assert next(row for row in rows if row["tile"] == 36)["unseen"] == 0
    context["unavailable_counts"][22] = 4
    summary, _ = scoring.score_waits(tiles, [], context)
    numpy.testing.assert_array_equal(summary[:, :2], [[1, 4]] * 4)
    context["own_discards"] = [22]
    summary, _ = scoring.score_waits(tiles, [], context)
    numpy.testing.assert_array_equal(summary[[0, 2], :7], 0)
    numpy.testing.assert_array_equal(summary[[0, 2], 7], 1)
    numpy.testing.assert_array_equal(summary[[1, 3], :2], [[1, 4]] * 2)
    context["unavailable_counts"][25] = 4
    summary, _ = scoring.score_waits(tiles, [], context)
    numpy.testing.assert_array_equal(summary[:, :7], 0)


def test_action_ranges_expand_one_shanten_and_distinguish_open_no_yaku():
    import riichienv

    # 123456m 789p 6s East 22p: one shanten, then draw/discard into tenpai.
    tiles = [0, 4, 8, 12, 17, 20, 60, 64, 68, 92, 108, 40, 41]
    context = dict(
        missed_agari=False,
        own_discards=[],
        unavailable_counts=numpy.bincount(
            [tile // 4 for tile in tiles], minlength=34
        ),
        unavailable_red=numpy.zeros(3, numpy.int8),
        dora_indicators=[],
        player_wind=1,
        round_wind=0,
    )
    values = scoring_reference.action_value_ranges(tiles, [], context)
    assert values["eligible"]
    numpy.testing.assert_array_equal(values["valid"], [[1, 1], [0, 0]])
    numpy.testing.assert_array_equal(
        values["range"][0], [[1300, 3900], [2000, 5200]]
    )
    # Structural one-shanten eligibility is distinct from an available path.
    exhausted = scoring_reference.action_value_ranges(
        tiles,
        [],
        {
            **context,
            "unavailable_counts": numpy.full(34, 4),
            "unavailable_red": numpy.ones(3),
        },
    )
    assert exhausted["eligible"] and not exhausted["valid"].any()
    assert not exhausted["restricted"].any()
    # Fixed open 123m removes pinfu/menzen: the ready remainder has no yaku.
    tiles[10] = 96
    context["unavailable_counts"] = numpy.bincount(
        [tile // 4 for tile in tiles], minlength=34
    )
    melds = [riichienv.Meld(riichienv.MeldType.Chi, [0, 4, 8], True)]
    values = scoring_reference.action_value_ranges(tiles[3:], melds, context)
    assert values["eligible"]
    assert not values["valid"].any()
    numpy.testing.assert_array_equal(values["restricted"], [1, 1])
    # Dora does not fix the absence of a scoring yaku.
    context["dora_indicators"] = [36]
    values = scoring_reference.action_value_ranges(tiles[3:], melds, context)
    assert not values["valid"].any()
    # A fixed white-dragon pon gives actual open value and no riichi branch.
    melds = [riichienv.Meld(riichienv.MeldType.Pon, [124, 125, 126], True)]
    context["unavailable_counts"] = numpy.bincount(
        [tile // 4 for tile in tiles[3:]] + [31, 31, 31], minlength=34
    )
    context["dora_indicators"] = []
    values = scoring_reference.action_value_ranges(tiles[3:], melds, context)
    numpy.testing.assert_array_equal(values["valid"], [[0, 0], [1, 1]])
    assert not values["restricted"].any()
    assert (values["range"][1] > 0).all()
    # Completely exhausted completions are distinct from no-yaku routes.
    context["unavailable_counts"][[22, 25]] = 4
    context["unavailable_red"][2] = 1
    values = scoring_reference.action_value_ranges(tiles[3:], melds, context)
    assert values["eligible"] and not values["valid"].any()
    assert not values["restricted"].any()
