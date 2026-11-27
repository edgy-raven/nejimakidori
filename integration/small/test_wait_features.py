"""Wait summaries describe obtainable wins; furiten remains structural."""

import numpy

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
