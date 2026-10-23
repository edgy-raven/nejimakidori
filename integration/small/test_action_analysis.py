"""Immediate discards preserve exact distances and available improving draws."""

import numpy

from integration import scoring_reference
from log_dataset import shanten


def test_immediate_discards_match_search_and_available_draws():
    for tiles in (
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
        [0, 1, 2, 9, 10, 11, 18, 19, 20, 24, 25, 26, 27, 28],
        [0, 0, 3, 3, 9, 9, 12, 12, 18, 18, 21, 21, 27, 28],
        [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 33],
        [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28],
    ):
        counts = numpy.bincount(tiles, minlength=34)
        unavailable = numpy.maximum(counts, numpy.arange(34) % 5)
        immediate = scoring_reference.analyze_actions(
            counts, known_unavailable_counts=unavailable
        )
        searched = scoring_reference.analyze_actions(
            counts,
            known_unavailable_counts=unavailable,
            calculate_completion=True,
        )
        for mode in (0, 1):
            for name in ("shanten", "ukeire_mask", "ukeire_count"):
                numpy.testing.assert_array_equal(
                    immediate[mode][name], searched[mode][name]
                )
            for cut in numpy.flatnonzero(counts):
                after = counts - numpy.eye(34, dtype=int)[cut]
                distance = shanten.calculate_shanten(after, bool(mode))
                improving = [
                    tile
                    for tile in range(34)
                    if unavailable[tile] < 4
                    and shanten.calculate_shanten(
                        after + numpy.eye(34, dtype=int)[tile], bool(mode)
                    )
                    < distance
                ]
                assert immediate[mode]["shanten"][cut] == distance
                assert immediate[mode]["ukeire_mask"][cut] == sum(
                    1 << tile for tile in improving
                )
                assert immediate[mode]["ukeire_count"][cut] == sum(
                    4 - unavailable[tile] for tile in improving
                )
