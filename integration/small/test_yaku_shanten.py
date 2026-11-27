"""Inclusive structural distances honor fixed melds and physical tile copies."""

import numpy

from log_dataset import shanten


def test_complete_hands_and_draws_reach_the_three_structural_families():
    # Tanyao, honitsu/chanta, and junchan.
    for tiles, expected, family in (
        ([1, 2, 3, 10, 11, 12, 19, 20, 21, 5, 5, 5, 14, 14], (-1, 7, 7), 0),
        ([0, 1, 2, 0, 1, 2, 6, 7, 8, 27, 27, 27, 8, 8], (7, -1, -1), 1),
        ([0, 1, 2, 9, 10, 11, 18, 19, 20, 26, 26, 26, 8, 8], (7, 7, -1), 2),
    ):
        counts = numpy.bincount(tiles, minlength=34)
        assert shanten.calculate_yaku_shanten(counts) == expected
        assert (
            shanten.calculate_yaku_shanten(
                counts - numpy.eye(34, dtype=int)[tiles[-1]]
            )[family]
            == 0
        )
        # Suit identities do not change any of these distances.
        assert (
            shanten.calculate_yaku_shanten(
                numpy.concatenate([counts[9:27], counts[:9], counts[27:]])
            )
            == expected
        )
    # Closed seven pairs participates in tanyao and flush.
    assert (
        shanten.calculate_yaku_shanten(
            numpy.bincount([1, 3, 5, 7, 10, 12, 14] * 2, minlength=34)
        )[0]
        == -1
    )
    assert (
        shanten.calculate_yaku_shanten(
            numpy.bincount([0, 1, 3, 4, 6, 7, 27] * 2, minlength=34)
        )[1]
        == -1
    )
    # Pure flush and terminal triplets already complete their families.
    assert (
        shanten.calculate_yaku_shanten(
            numpy.bincount(
                [0, 1, 2, 0, 1, 2, 6, 7, 8, 6, 6, 6, 7, 7], minlength=34
            )
        )[1]
        == -1
    )
    assert (
        shanten.calculate_yaku_shanten(
            numpy.bincount(
                [0, 0, 0, 8, 8, 8, 9, 9, 9, 17, 17, 17, 18, 18], minlength=34
            )
        )[2]
        == -1
    )
    # Pure-flush seven pairs, honroutou seven pairs, and all honors.
    for pairs, family in (
        ([0, 1, 2, 3, 4, 5, 6], 1),
        ([0, 8, 9, 17, 18, 26, 27], 2),
        ([27, 28, 29, 30, 31, 32, 33], 2),
    ):
        counts = numpy.bincount(pairs * 2, minlength=34)
        assert shanten.calculate_yaku_shanten(counts)[family] == -1
        counts[pairs[-1]] -= 1
        assert shanten.calculate_yaku_shanten(counts)[family] == 0


def test_fixed_calls_and_kans_constrain_routes_without_fifth_copies():
    counts = numpy.bincount([1], minlength=34)
    # Four fixed simple triplets leave one copy of 2m, not a possible 2m pair.
    assert shanten.calculate_yaku_shanten(
        counts, [(1, 1, 1), (3, 3, 3), (5, 5, 5), (7, 7, 7)]
    ) == (1, 1, 14)
    # A kan is still one fixed meld, and all four copies are unavailable.
    assert shanten.calculate_yaku_shanten(
        numpy.bincount([2], minlength=34),
        [(1, 1, 1, 1), (3, 3, 3), (5, 5, 5), (7, 7, 7)],
    ) == (0, 0, 14)
    assert shanten.calculate_yaku_shanten(
        numpy.bincount([8], minlength=34),
        [(0, 1, 2), (9, 10, 11), (18, 19, 20), (26, 26, 26)],
    ) == (14, 14, 0)
    assert shanten.calculate_yaku_shanten(
        numpy.bincount([8], minlength=34),
        [(0, 1, 2), (0, 1, 2), (6, 7, 8), (27, 27, 27)],
    ) == (14, 0, 0)
    # Honor groups and terminal triplets need no sequence to qualify.
    assert shanten.calculate_yaku_shanten(
        numpy.bincount([8], minlength=34),
        [(0, 0, 0), (9, 9, 9), (18, 18, 18), (27, 27, 27)],
    ) == (14, 14, 0)
