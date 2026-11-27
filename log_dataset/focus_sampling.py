"""Observed-state priorities for focused imitation sampling."""

import numpy

from log_dataset import scoring, shanten


def visible_threat(features, seat):
    """Historical public threat rule for one relative opponent seat."""
    opened = (features["meld_type"][seat] > 0) & (
        features["meld_type"][seat] != 4
    )
    if (
        features["riichi_state"][seat] > 0
        or opened.sum() >= 3
        or (opened.sum() >= 2 and features["opponent_discard_count"][seat] >= 9)
    ):
        return True
    han = 0
    suits = set()
    for slot, kind in enumerate(features["meld_type"][seat]):
        if not kind:
            continue
        tiles = features["meld_tiles"][seat, slot]
        base = shanten.BASE_IDS[tiles[tiles < 37]]
        han += int(base[0] >= 31)
        han += int(base[0] == 27 + features["seat_wind"][seat])
        han += int(base[0] == 27 + features["prevailing_wind"])
        han += int(features["dora_multiplicity"][base].sum())
        han += int(((tiles >= 34) & (tiles < 37)).sum())
        if opened[slot]:
            suits.update(base[base < 27] // 9)
    if opened.sum() >= 2 and len(suits) == 1:
        middle_suits = set()
        for segment in ("pre", "post"):
            tiles = features[f"river_{segment}_tile_id"][seat - 1]
            middle_suits.update(
                tiles[
                    (features[f"river_{segment}_valid"][seat - 1] > 0)
                    & (tiles < 27)
                    & (tiles % 9 >= 2)
                    & (tiles % 9 <= 6)
                ]
                // 9
            )
        if len(middle_suits | suits) == 3:
            han += 2
    return han >= 3


def metadata(observation):
    """Return focused multiplicity and the independent placement-gap flag.

    Only observed fields enter this classification. The route search retains
    legal discard support and its original bounded-search exclusion; rounded
    auxiliary labels are not used as sampling predicates.
    """
    features = observation.features
    phase = int(features["decision_phase"])
    calls = int(phase in (1, 2))
    legal = (
        "discard_action_legal",
        "response_legal_mask",
        "kan_legal_mask",
        "riichi_legal_mask",
    )
    if phase >= len(legal) or numpy.count_nonzero(features[legal[phase]]) < 2:
        # Response and kan opportunities qualify even
        # when the native category classifier found no other eligible choice.
        return calls, False

    paths = shanten.tenpai_paths(
        features["candidate_hand_counts"][0],
        features["known_unavailable_counts"],
        (
            numpy.flatnonzero(features["discard_action_legal"])
            if phase in (0, 3)
            else None
        ),
    )
    routes = numpy.zeros(3, bool)
    if paths is not None:
        melds = tuple(
            (
                int(kind) - 1,
                tuple(
                    int(shanten.BASE_IDS[tile]) * 4 + copy
                    for copy, tile in enumerate(tiles)
                    if tile < 37
                ),
                bool(kind != 4),
            )
            for kind, tiles in zip(
                features["meld_type"][0], features["meld_tiles"][0]
            )
            if kind
        )
        for hand in paths[0]:
            flags = scoring.path_completion_flags(
                counts=hand.tobytes(),
                melds=melds,
                player_wind=int(features["seat_wind"][0]),
                round_wind=int(features["prevailing_wind"]),
            )
            routes |= [
                flags[scoring.NAMES.index(name)]
                for name in ("tanyao", "honitsu", "chanta")
            ]
            if routes.all():
                break

    threat = any(visible_threat(features, seat) for seat in range(1, 4))

    late = features["hands_remaining"] <= 1 or (
        features["live_wall_count"] <= 16
        and min(
            int(features[name])
            for name in (
                "normal_self_shanten",
                "chiitoitsu_self_shanten",
                "kokushi_self_shanten",
            )
        )
        <= 2
    )
    gap = False
    if features["hands_remaining"] != 7:
        order = numpy.argsort(features["ranks"])
        ranked = features["scores"][order]
        thresholds = numpy.where(
            order <= max(0, int(features["hands_remaining"])), 16000, 10000
        )
        rank = int(features["ranks"][features["seat_wind"][0]])
        if rank == 1:
            gap = ranked[0] - ranked[1] > thresholds[1]
        elif rank == 2:
            gap = (
                ranked[0] - ranked[1] < thresholds[1]
                and ranked[1] - ranked[3] > thresholds[3]
            )
        else:
            gap = ranked[2] - ranked[3] < thresholds[3]
    return calls + int(routes @ [2, 2, 1]) + int(threat) + int(late), bool(gap)
