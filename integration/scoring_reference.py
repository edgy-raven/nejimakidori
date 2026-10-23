"""Python completion-range and packed-route oracles for native parity tests."""

import ctypes

import numpy
import riichienv

from log_dataset import scoring, shanten

ACTION_DTYPE = numpy.dtype(
    [
        ("shanten", numpy.int8, 34),
        ("ukeire_mask", numpy.uint64, 34),
        ("ukeire_count", numpy.uint16, 34),
        ("upgrade_mask", numpy.uint64, 34),
        ("upgrade_type_count", numpy.uint8, 34),
        ("upgrade_tile_count", numpy.uint16, 34),
        ("upgrade_weighted_gain", numpy.uint16, 34),
        ("completion_probability", numpy.float32, 34),
        ("completion_valid", numpy.uint8, 34),
    ],
    align=True,
)


def analyze_actions(
    counts,
    tile_ids=None,
    known_unavailable_counts=None,
    modes=(0, 1),
    calculate_upgrades=False,
    calculate_completion=False,
):
    dll = shanten.native()
    dll.nyanten_analyze_modes.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    dll.nyanten_analyze_modes.restype = None
    cuts = (
        numpy.flatnonzero(counts)
        if tile_ids is None
        else numpy.unique(shanten.BASE_IDS[numpy.asarray(tile_ids, dtype=int)])
    )
    values = numpy.zeros(len(modes), dtype=ACTION_DTYPE)
    dll.nyanten_analyze_modes(
        bytes(map(int, counts)),
        bytes(
            map(
                int,
                (
                    counts
                    if known_unavailable_counts is None
                    else known_unavailable_counts
                ),
            )
        ),
        sum(1 << int(cut) for cut in cuts),
        calculate_upgrades,
        calculate_completion,
        bytes(modes),
        len(modes),
        values.ctypes.data,
    )
    return {
        mode: {
            name: (
                values[index][name].astype(numpy.int64)
                if values.dtype[name].base.kind in "iu"
                else values[index][name]
            )
            for name in ACTION_DTYPE.names
        }
        for index, mode in enumerate(modes)
    }


def yaku_path_fractions(packed):
    """Visible shortest-route mass by family; unknown searches return -1."""
    melds = [
        riichienv.Meld(
            (
                riichienv.MeldType.Chi,
                riichienv.MeldType.Pon,
                riichienv.MeldType.Daiminkan,
                riichienv.MeldType.Ankan,
                riichienv.MeldType.Kakan,
            )[int(packed["meld_type"][0, meld]) - 1],
            [
                int(tile) * 4 + copy
                for tile, count in enumerate(
                    numpy.bincount(
                        packed["meld_tile_ids"][0, meld][
                            packed["meld_tile_valid"][0, meld] > 0
                        ],
                        minlength=34,
                    )
                )
                for copy in range(count)
            ],
            bool(packed["meld_type"][0, meld] != 4),
        )
        for meld in numpy.flatnonzero(packed["meld_valid"][0])
    ]
    return scoring.hand_yaku_fractions(
        counts=packed["candidate_hand_counts"][0],
        unavailable=packed["known_unavailable_counts"],
        melds=melds,
        player_wind=int(packed["seat_wind"][0]),
        round_wind=int(packed["prevailing_wind"]),
        cuts=(
            numpy.flatnonzero(packed["discard_action_legal"])
            if int(packed["decision_phase"]) in (0, 3)
            else None
        ),
    )


def action_value_ranges(tiles, melds, context, cut=None):
    """Exact improving one-shanten paths; ranges are conditional on a legal win."""
    counts = numpy.bincount(numpy.asarray(tiles) // 4, minlength=34)
    distance, draws, _ = shanten.improving(counts)
    result = {
        "range": numpy.zeros((2, 2, 2), numpy.float32),
        "valid": numpy.zeros((2, 2), numpy.int8),
        "restricted": numpy.zeros(2, numpy.int8),
        "eligible": int(distance in (0, 1)),
    }
    if not result["eligible"]:
        return result
    context = {
        **context,
        "own_discards": context["own_discards"]
        + ([] if cut is None else [cut]),
    }
    if distance == 0:
        summary, _ = scoring.score_waits(tiles, melds, context)
        opened = any(meld.opened for meld in melds)
        for kind in range(2):
            scenario = kind if opened else 2 + kind
            result["range"][int(opened), kind] = summary[scenario, 2:4]
            result["valid"][int(opened), kind] = summary[scenario, 1] > 0
        if opened:
            # Distinguish no-yaku completion paths from ron blocked by furiten.
            _, unrestricted = scoring.score_waits(
                tiles,
                melds,
                {**context, "missed_agari": False, "own_discards": []},
            )
            result["restricted"][:] = [
                any(
                    row["unseen"] > 0 and not row["values"][kind]["legal"]
                    for row in unrestricted
                )
                for kind in range(2)
            ]
        return result
    for draw in draws:
        five = draw in (4, 13, 22)
        for red in (False, True) if five else (False,):
            unseen_red = (
                1 - context["unavailable_red"][draw // 9] if five else 0
            )
            copies = (
                unseen_red
                if red
                else 4 - context["unavailable_counts"][draw] - unseen_red
            )
            if copies <= 0:
                continue
            tile = draw * 4 + int(five and not red)
            while tile in tiles:
                tile += 1
            hand = [*tiles, tile]
            after = {
                **context,
                "missed_agari": False,
                "unavailable_counts": (
                    context["unavailable_counts"] + (numpy.arange(34) == draw)
                ),
                "unavailable_red": numpy.array(context["unavailable_red"])
                + (numpy.arange(3) == draw // 9) * int(red),
            }
            cuts = sorted({tile // 4 for tile in hand})
            analysis = analyze_actions(
                numpy.bincount(numpy.asarray(hand) // 4, minlength=34),
                cuts,
                after["unavailable_counts"],
                modes=(1,),
            )[1]
            seen = set()
            for position, discarded in enumerate(hand):
                code = (
                    34 + discarded // 36
                    if discarded in (16, 52, 88)
                    else discarded // 4
                )
                if code in seen or analysis["shanten"][discarded // 4] != 0:
                    continue
                seen.add(code)
                candidate = action_value_ranges(
                    hand[:position] + hand[position + 1 :],
                    melds,
                    after,
                    discarded // 4,
                )
                merge_value_ranges(result, candidate)
    return result


def merge_value_ranges(result, candidate):
    """Union legal completion ranges without averaging decisions or paths."""
    new = candidate["valid"] > 0
    both = new & (result["valid"] > 0)
    result["range"][new & ~both] = candidate["range"][new & ~both]
    result["range"][..., 0][both] = numpy.minimum(
        result["range"][..., 0][both], candidate["range"][..., 0][both]
    )
    result["range"][..., 1][both] = numpy.maximum(
        result["range"][..., 1][both], candidate["range"][..., 1][both]
    )
    result["valid"] |= candidate["valid"]
    result["restricted"] |= candidate["restricted"]
    result["eligible"] |= candidate["eligible"]
