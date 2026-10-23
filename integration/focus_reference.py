"""Frozen packed-feature focus oracle for native extraction parity tests."""

import numpy

from integration import scoring_reference
from log_dataset import scoring, shanten


def yakuhai_han(packed, seat, opened_only=False):
    """Visible value-honor groups; double winds count twice, kans once."""
    tiles = packed["meld_tile_ids"][seat, :, 0][
        (packed["meld_valid"][seat] > 0)
        & ((packed["meld_type"][seat] != 4) if opened_only else True)
    ]
    return int(
        (tiles >= 31).sum()
        + (tiles == 27 + packed["seat_wind"][seat]).sum()
        + (tiles == 27 + packed["prevailing_wind"]).sum()
    )


def visible_opponent_threat(packed):
    """Riichi, visible three-han value, or sufficiently developed open hands."""
    if packed["riichi_state"][1:].any():
        return True
    for seat in range(1, 4):
        opened = (packed["meld_valid"][seat] > 0) & (
            packed["meld_type"][seat] != 4
        )
        if opened.sum() >= 3 or (
            opened.sum() >= 2 and packed["opponent_discard_count"][seat] >= 9
        ):
            return True
        valid = (packed["meld_tile_valid"][seat] > 0) & (
            packed["meld_valid"][seat, :, None] > 0
        )
        tiles = packed["meld_tile_ids"][seat][valid]
        han = int(
            packed["dora_multiplicity"][tiles].sum()
            + packed["meld_tile_red"][seat][valid].sum()
        )
        han += yakuhai_han(packed, seat)
        called = packed["meld_tile_ids"][seat][valid & opened[:, None]]
        suits = numpy.unique(called[called < 27] // 9)
        if opened.sum() >= 2 and len(suits) == 1:
            river = numpy.concatenate(
                [
                    packed["river_" + phase + "_tile_id"][seat - 1][
                        packed["river_" + phase + "_valid"][seat - 1] > 0
                    ]
                    for phase in ("pre", "post")
                ]
            )
            middle_suits = numpy.unique(
                river[(river < 27) & (river % 9 >= 2) & (river % 9 <= 6)] // 9
            )
            if all(
                suit in middle_suits for suit in range(3) if suit != suits[0]
            ):
                han += 2
        if han >= 3:
            return True
    return False


def placement_gap_focus(packed):
    """Rank-specific score-gap focus under the configured tsumo thresholds."""
    if int(packed["prevailing_wind"]) == 0 and int(packed["kyoku_number"]) == 0:
        return False
    order = numpy.argsort(packed["seat_context"][:, 1])
    scores = numpy.rint(
        packed["seat_context"][order, 0].astype(numpy.float64) * 100000
    ).astype(numpy.int64)
    thresholds = numpy.where(
        packed["seat_wind"][order] <= packed["scheduled_hands_remaining"],
        16000,
        10000,
    )
    return bool(
        placement_gap_conditions(
            scores, thresholds, numpy.rint(packed["seat_context"][0, 1] * 4)
        )
    )


def placement_gap_conditions(scores, thresholds, rank):
    """Shared scalar/array rule for extraction and batched sampling."""
    return (
        ((rank == 1) & (scores[..., 0] - scores[..., 1] > thresholds[..., 1]))
        | (
            (rank == 2)
            & (scores[..., 0] - scores[..., 1] < thresholds[..., 1])
            & (scores[..., 1] - scores[..., 3] > thresholds[..., 3])
        )
        | ((rank >= 3) & (scores[..., 2] - scores[..., 3] < thresholds[..., 3]))
    )


def late_placement_focus(packed):
    """Original South 3/4, extensions and keiten selection, scalar or batch."""
    return (
        (packed["prevailing_wind"] >= 2)
        | ((packed["prevailing_wind"] == 1) & (packed["kyoku_number"] >= 2))
        | (
            (packed["live_wall_count"] <= 16)
            & (
                (packed["normal_self_shanten"] <= 2)
                | (packed["chiitoitsu_self_shanten"] <= 2)
                | (packed["kokushi_self_shanten"] <= 2)
            )
        )
    )


def categories(packed):
    """Select opportunities, including declined calls, without outcome labels."""
    phase = int(packed["decision_phase"])
    legal = (
        packed[
            (
                "discard_action_legal",
                "response_legal_mask",
                "kan_legal_mask",
                "riichi_legal_mask",
            )[phase]
        ]
        if phase < 4
        else numpy.zeros(1)
    )
    if numpy.count_nonzero(legal) < 2:
        return 0
    fractions = scoring_reference.yaku_path_fractions(packed)
    routes = (
        fractions[
            [
                scoring.NAMES.index(name)
                for name in ("tanyao", "honitsu", "chanta")
            ]
        ]
        > 0
    )
    distance = (
        min(
            int(packed[name + "_self_shanten"])
            for name in ("normal", "chiitoitsu", "kokushi")
        )
        - 1
    )
    fast = False
    if phase == 1:
        for kind in ("chi", "pon"):
            valid = packed[kind + "_discard_legal"] > 0
            progress = (packed[kind + "_discard_shanten"] < distance + 1) | (
                (packed[kind + "_discard_shanten"] == distance + 1)
                & (
                    packed[kind + "_discard_ukeire_count"]
                    > packed["self_efficiency"][1]
                )
            )
            yakuhai = yakuhai_han(packed, 0, opened_only=True) > 0
            if kind == "pon":
                tile = int(packed["trigger_tile_id"])
                yakuhai |= tile >= 31 or tile in (
                    27 + int(packed["seat_wind"][0]),
                    27 + int(packed["prevailing_wind"]),
                )
            compatible = packed[kind + "_call_yaku_possibility"]
            # Native possibility only checks meld compatibility; route evidence
            # above is required as well. Indices are native tanyao/flush/chanta.
            viable = yakuhai | numpy.any(
                (compatible[:, [0, 6, 16]] > 0) & routes, axis=-1
            )
            # A reachable legal open ron is direct evidence even without a
            # pre-call shortest route, with no minimum-value threshold.
            viable = viable[:, None] | (
                packed["action_value_range"][
                    slice(112, 223) if kind == "chi" else slice(223, 260),
                    1,
                    0,
                    1,
                ].reshape(valid.shape)
                > 0
            )
            fast |= bool(numpy.any(valid & progress & viable))
    threat = visible_opponent_threat(packed)
    if threat and phase == 0:
        cuts = numpy.flatnonzero(legal)
        safety = packed["genbutsu_to_seat"][1:, shanten.BASE_IDS[cuts]]
        threat = bool(
            numpy.any(numpy.ptp(safety, axis=1))
            or numpy.ptp(packed["discard_action_all_shanten"][cuts])
            or numpy.ptp(packed["discard_action_all_ukeire_count"][cuts])
        )
    placement = late_placement_focus(packed) or placement_gap_focus(packed)
    return sum(
        1 << bit
        for bit, selected in enumerate((fast, *routes, threat, placement))
        if selected
    )
