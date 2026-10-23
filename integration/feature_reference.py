"""Independent Python feature oracle; never imported by production code."""

import json

import numpy
import riichienv

from integration import scoring_reference
from log_dataset import _native_collection, shanten
from model import outcomes

PHASES = {"DISCARD": 0, "RESPONSE": 1, "KAN": 2, "RIICHI": 3, "WIN": 4}
OUTSIDE = ((), (0,), (0, 1), (1, 2), (2, 6), (6, 7), (7, 8), (8,), ())


def native_tile_code(tile):
    return 34 + (tile // 4) // 9 if tile in (16, 52, 88) else tile // 4


def action_event(action, drawn_tile=None):
    event = json.loads(action.to_mjai())
    if event["type"] == "dahai":
        event["tsumogiri"] = action.tile == drawn_tile
    return event


def prefer_red_calls(legal):
    selected = {}
    for index, action in enumerate(legal):
        if action.action_type in (
            riichienv.ActionType.CHI,
            riichienv.ActionType.PON,
        ):
            key = (
                action.action_type,
                action.tile // 4,
                tuple(sorted(tile // 4 for tile in action.consume_tiles)),
            )
            if key in selected and any(
                tile in (16, 52, 88) for tile in selected[key].consume_tiles
            ):
                continue
        else:
            key = index
        selected[key] = action
    return list(selected.values())


def drawn_tile_code(packed):
    tile = int(packed["current_draw_tile_id"])
    return 34 + tile // 9 if packed["current_draw_is_red"] else tile


def chii_pattern(called, consumed):
    called = (
        shanten.mjai_tile_code(called)
        if isinstance(called, str)
        else native_tile_code(called)
    )
    consumed = [
        (
            shanten.mjai_tile_code(tile)
            if isinstance(tile, str)
            else native_tile_code(tile)
        )
        for tile in consumed
    ]
    return shanten.base_id(called) - min(
        shanten.base_id(tile) for tile in [called] + consumed
    )


def dora_counts(markers):
    result = numpy.zeros(34, numpy.int8)
    for marker in markers:
        tile = shanten.base_id(marker)
        next_tile = (
            tile // 9 * 9 + (tile + 1) % 9
            if tile < 27
            else 27 + (tile - 26) % 4 if tile < 31 else 31 + (tile - 30) % 3
        )
        result[next_tile] += 1
    return result


def yaku_flags(observation, actor, claim=None):
    melds = observation.melds[actor]
    exposed = numpy.bincount(
        [tile // 4 for tile in observation.dora_indicators]
        + [tile // 4 for river in observation.discards for tile in river],
        minlength=34,
    )
    for seat_melds in observation.melds:
        for meld in seat_melds:
            exposed += numpy.bincount(
                [tile // 4 for tile in meld.tiles], minlength=34
            )
            if meld.called_tile is not None:
                exposed[meld.called_tile // 4] -= 1
    exposed -= numpy.bincount(
        [tile // 4 for meld in melds for tile in meld.tiles], minlength=34
    )
    if claim is not None and claim["actor"] != actor:
        exposed[shanten.base_id(claim["pai"])] -= 1
    available = 4 - exposed
    tiles = [tile // 4 for meld in melds for tile in meld.tiles]
    fixed = {
        meld.tiles[0] // 4
        for meld in melds
        if meld.meld_type != riichienv.MeldType.Chi
    }
    free = 4 - len(melds)
    triplets = fixed | {
        tile for tile in range(34) if free and available[tile] >= 3
    }
    terminals = {0, 8, 9, 17, 18, 26}
    suits = {tile // 9 for tile in tiles if tile < 27}
    sequences = sum(meld.meld_type == riichienv.MeldType.Chi for meld in melds)
    outside = all(
        any(tile // 4 >= 27 or tile // 4 in terminals for tile in meld.tiles)
        for meld in melds
    )
    dragons = {31, 32, 33}
    flags = numpy.zeros(19, numpy.int8)
    flags[0] = all(tile < 27 and tile not in terminals for tile in tiles)
    flags[1:6] = [
        tile in triplets
        for tile in (
            31,
            32,
            33,
            27 + int(observation.round_wind),
            27 + (actor - observation.oya) % 4,
        )
    ]
    flags[6] = len(suits) <= 1
    flags[7] = len(suits) <= 1 and all(tile < 27 for tile in tiles)
    flags[8] = not sequences
    flags[9] = not melds
    flags[10] = any(
        pair not in fixed
        and available[pair] >= 2
        and dragons - {pair} <= triplets
        and len(dragons - {pair} - fixed) <= free
        for pair in dragons
    )
    flags[11] = dragons <= triplets and len(dragons - fixed) <= free
    flags[12] = all(tile >= 27 for tile in tiles)
    flags[13] = all(tile in terminals for tile in tiles)
    flags[14] = all(tile >= 27 or tile in terminals for tile in tiles)
    flags[15] = not melds and all(
        available[tile] > 0 for tile in terminals | set(range(27, 34))
    )
    flags[16] = outside and bool(sequences or free)
    flags[17] = flags[16] and all(tile < 27 for tile in tiles)
    flags[18] = all(not meld.opened for meld in melds) and free >= 2
    return flags


def hand_melds(melds):
    result = numpy.zeros((34, 6), numpy.int8)
    for meld in melds:
        result[min(tile // 4 for tile in meld.tiles), int(meld.meld_type)] += 1
        for tile in meld.tiles:
            if tile in (16, 52, 88):
                result[tile // 4, 5] += 1
    return result


def pack(
    state,
    actor,
    phase,
    legal_observation,
    east_only=False,
    *,
    stage="all",
    packed=None,
):
    if packed is None:
        packed = {
            name: numpy.zeros(spec["shape"], spec["dtype"])
            for name, spec in _native_collection.INPUTS.items()
        }
    for name in _native_collection.IGNORED_INPUTS:
        packed[name].fill(0)
    own = legal_observation.hand
    counts = numpy.bincount([tile // 4 for tile in own], minlength=34)
    reds = numpy.array([own.count(tile) for tile in (16, 52, 88)])
    packed["candidate_hand_counts"][0] = counts
    packed["candidate_hand_red"][0] = reds
    packed["closed_hand"][...] = all(
        not meld.opened for meld in state.melds[actor]
    )
    if stage in {"all", "analysis"}:
        for name, value in zip(
            ("normal", "chiitoitsu", "kokushi"),
            shanten.calculate_families(counts),
        ):
            packed[name + "_self_shanten"][...] = (
                5
                if name != "normal" and state.melds[actor]
                else min(4, value + 1)
            )
    if stage in {"all", "yaku"}:
        packed["self_yaku_possibility"][:] = yaku_flags(
            legal_observation, actor, state.pending_discard
        )
    if state.draws[actor] is not None:
        packed["current_draw_valid"][...] = 1
        packed["current_draw_tile_id"][...] = shanten.base_id(
            state.draws[actor]
        )
        packed["current_draw_is_red"][...] = shanten.is_red(state.draws[actor])
    packed["decision_phase"][...] = PHASES[phase]
    if phase == "WIN":
        packed["win_kind"][...] = (
            2
            if any(
                action.action_type == riichienv.ActionType.TSUMO
                for action in legal_observation.legal_actions()
            )
            else 1
        )
    packed["rinshan_state"][...] = state.rinshan_actor == actor
    packed["dora_multiplicity"][:] = dora_counts(state.dora_markers)
    for number, marker in enumerate(state.dora_markers):
        packed["dora_tile_id"][number] = shanten.base_id(marker)
        packed["dora_valid"][number] = 1
    visible = state.public_tiles()
    unavailable = counts + numpy.bincount(
        [shanten.base_id(tile) for tile in visible], minlength=34
    )
    packed["known_unavailable_counts"][:] = unavailable
    packed["public_visible_red_counts"][:] = [
        visible.count(tile) for tile in ("5mr", "5pr", "5sr")
    ]
    scores = numpy.array(legal_observation.scores)
    wind = "ESWN".index(state.start["bakaze"])
    packed["prevailing_wind"][...] = wind
    packed["kyoku_number"][...] = state.start["kyoku"] - 1
    packed["dealer_relative_seat"][...] = (state.start["oya"] - actor) % 4
    packed["honba"][...] = state.start["honba"]
    packed["riichi_sticks"][...] = legal_observation.riichi_sticks
    packed["live_wall_count"][...] = state.live_wall
    packed["total_kan_count"][...] = sum(
        meld.meld_type not in (riichienv.MeldType.Chi, riichienv.MeldType.Pon)
        for melds in state.melds
        for meld in melds
    )
    kan_owners = [
        seat
        for seat, melds in enumerate(state.melds)
        for meld in melds
        if meld.meld_type
        not in (riichienv.MeldType.Chi, riichienv.MeldType.Pon)
    ]
    for declaration in range(2):
        packed["discard_abort"][declaration] = int(
            (len(kan_owners) == 4 and len(set(kan_owners)) > 1)
            or all(
                state.declared[seat] or (seat == actor and declaration)
                for seat in range(4)
            )
        )
    if (
        not any(state.melds)
        and not state.rivers[actor]
        and all(
            len(state.rivers[seat]) == 1 for seat in range(4) if seat != actor
        )
    ):
        first = {
            state.rivers[seat][0]["pai"] for seat in range(4) if seat != actor
        }
        if len(first) == 1 and next(iter(first)) in "ESWN":
            packed["discard_abort"][:, shanten.base_id(next(iter(first)))] = 1
    hand_number = wind * 4 + state.start["kyoku"] - 1
    final_hand = 3 if east_only else 7
    packed["nominal_all_last"][...] = hand_number == final_hand
    packed["scheduled_hands_remaining"][...] = max(0, final_hand - hand_number)
    packed["dealer_currently_leads"][...] = (
        scores[state.start["oya"]] == scores.max()
    )
    packed["any_player_at_or_above_30000"][...] = scores.max() >= 30000
    table_features(packed, state, actor, legal_observation, stage=stage)
    for tile in range(27):
        for outside in OUTSIDE[tile % 9]:
            packed["blocker_counts"][tile // 9 * 9 + outside] += (
                unavailable[tile] == 4,
                unavailable[tile] == 3,
                unavailable[tile] == 2,
                unavailable[tile] == 1,
                unavailable[tile] == 0,
            )
    if state.pending_discard is not None:
        packed["trigger_valid"][...] = 1
        packed["trigger_tile_id"][...] = shanten.base_id(
            state.pending_discard["pai"]
        )
        packed["trigger_is_red"][...] = shanten.is_red(
            state.pending_discard["pai"]
        )
        packed["trigger_source"][...] = (
            state.pending_discard["actor"] - actor
        ) % 4
    candidate_features(packed, state, actor, legal_observation, stage=stage)
    public_scoring_features(packed)
    return packed


def public_scoring_features(packed):
    """Independent NumPy oracle for cached fixed-meld scoring contributions."""
    tiles = packed["meld_tile_ids"][..., 0]
    packed["public_fu"][...] = (
        numpy.take([0, 0, 2, 8, 16, 8], packed["meld_type"])
        * (1 + ((tiles >= 27) | (tiles % 9 == 0) | (tiles % 9 == 8)))
        * packed["meld_valid"]
    ).sum(axis=-1)
    packed["public_bonus"][...] = (
        (
            numpy.take_along_axis(
                packed["dora_multiplicity"][..., None, None, :],
                packed["meld_tile_ids"].astype(int),
                axis=-1,
            )
            + packed["meld_tile_red"]
        )
        * packed["meld_tile_valid"]
        * packed["meld_valid"][..., None]
    ).sum(axis=(-1, -2))
    tiles = numpy.arange(34)
    packed["candidate_public_fu"][...] = (
        packed["candidate_melds"][..., :5]
        * numpy.array([0, 2, 8, 16, 8])
        * (1 + ((tiles >= 27) | (tiles % 9 == 0) | (tiles % 9 == 8)))[:, None]
    ).sum(axis=(-1, -2))
    packed["candidate_public_bonus"][...] = (
        packed["candidate_melds"][..., :5]
        * numpy.stack(
            [
                packed["dora_multiplicity"]
                + numpy.roll(packed["dora_multiplicity"], -1, axis=-1)
                + numpy.roll(packed["dora_multiplicity"], -2, axis=-1),
                *[
                    packed["dora_multiplicity"] * count
                    for count in (3, 4, 4, 4)
                ],
            ],
            axis=-1,
        )[..., None, :, :]
    ).sum(axis=(-1, -2)) + packed["candidate_melds"][..., 5].sum(axis=-1)


def riichi_push_features(packed, state, actor):
    """Count pushes using each threat's genbutsu at the time of discard."""
    discards = sorted(
        (
            (seat, row)
            for seat, river in enumerate(state.rivers)
            for row in river
        ),
        key=lambda item: item[1]["event_index"],
    )
    for target in range(4):
        if not state.accepted[target]:
            continue
        declared_index = next(
            index
            for index, event in enumerate(state.history)
            if event["type"] == "reach_accepted" and event["actor"] == target
        )
        safe_tiles = {
            shanten.base_id(row["pai"])
            for row in state.rivers[target]
            if row["event_index"] < declared_index
        }
        for seat, row in discards:
            if row["event_index"] <= declared_index:
                continue
            tile = shanten.base_id(row["pai"])
            # Judge safety before this discard passes, never using later
            # genbutsu. Each column retains the specific riichi threat.
            if (
                seat != target
                and not row["tsumogiri"]
                and tile not in safe_tiles
            ):
                packed["riichi_push_count"][
                    (seat - actor) % 4, (target - actor) % 4
                ] += 1
            if row["resolution"]:
                safe_tiles.add(tile)


def table_features(  # noqa: C901 - independent legacy parity oracle
    packed, state, actor, legal_observation, *, stage="all"
):
    riichi_push_features(packed, state, actor)
    river_dora = {
        markers: dora_counts(markers)
        for markers in {
            tuple(row["dora_markers"])
            for river in state.rivers
            for row in river
        }
    }
    scores = numpy.array(legal_observation.scores)
    ranks = outcomes.score_ranks(
        scores,
        (state.start["oya"] - state.start["kyoku"] + 1) % 4,
    )
    packed["score_extrema"][:] = [scores.min() / 100000, scores.max() / 100000]
    for relative in range(4):
        seat = (actor + relative) % 4
        packed["last_tedashi_tile"][relative] = 0
        for row in reversed(state.rivers[seat]):
            if not row["tsumogiri"]:
                packed["last_tedashi_tile"][
                    relative, shanten.mjai_tile_code(row["pai"])
                ] = 1
                break
        packed["seat_wind"][relative] = (seat - state.start["oya"]) % 4
        packed["seat_context"][relative] = [
            scores[seat] / 100000,
            ranks[seat] / 4,
        ]
        packed["riichi_state"][relative] = int(state.declared[seat]) + int(
            state.accepted[seat]
        )
        packed["ippatsu_alive"][relative] = state.ippatsu[seat]
        packed["opponent_discard_count"][relative] = len(state.rivers[seat])
        if relative:
            packed["opponent_called_tile_count"][relative] = sum(
                len(meld.tiles) for meld in state.melds[seat] if meld.opened
            )
            if stage in {"all", "yaku"} and any(
                meld.opened for meld in state.melds[seat]
            ):
                packed["open_opponent_yaku_possibility"][relative] = yaku_flags(
                    legal_observation, seat, state.pending_discard
                )[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 16, 17]]
        for index, meld in enumerate(state.melds[seat]):
            event = state.meld_events[seat][index]
            packed["meld_valid"][relative, index] = 1
            packed["meld_type"][relative, index] = 1 + int(meld.meld_type)
            tiles = (
                ([] if event["type"] == "ankan" else [event["pai"]])
                + event["consumed"]
                + ([event["upgrade_tile"]] if event["type"] == "kakan" else [])
            )
            for number, tile in enumerate(tiles):
                packed["meld_tile_ids"][relative, index, number] = (
                    shanten.base_id(tile)
                )
                packed["meld_tile_red"][relative, index, number] = (
                    shanten.is_red(tile)
                )
                packed["meld_tile_valid"][relative, index, number] = 1
            packed["meld_source"][relative, index] = (
                0 if not meld.opened else (event["target"] - actor) % 4 + 1
            )
            packed["meld_call_age"][relative, index] = min(
                1, (len(state.history) - event["event_index"]) / 100
            )
            if "upgrade_index" in event:
                packed["meld_upgrade_age"][relative, index] = min(
                    1, (len(state.history) - event["upgrade_index"]) / 100
                )
        riichi_boundary = next(
            (
                i + 1
                for i, row in enumerate(state.rivers[seat])
                if row["riichi"]
            ),
            len(state.rivers[seat]),
        )
        last_tedashi = None
        call_indices = [
            event[key]
            for event in state.meld_events[seat]
            for key in ("event_index", "upgrade_index")
            if key in event
        ]
        last_hand_change = max(
            call_indices
            + [
                row["event_index"]
                for row in state.rivers[seat]
                if not row["tsumogiri"]
            ],
            default=-1,
        )
        # Passed tiles remain evidence through tsumogiri, but not a changed
        # hand. Keep this separate from genbutsu: ron can be declined.
        for other_seat, river in enumerate(state.rivers):
            if other_seat != seat:
                for row in river:
                    if (
                        row["resolution"]
                        and row["event_index"] > last_hand_change
                    ):
                        packed["passed_unchanged_to_seat"][
                            relative, shanten.base_id(row["pai"])
                        ] = 1
        if state.accepted[seat]:
            declared_index = next(
                i
                for i, event in enumerate(state.history)
                if event["type"] == "reach_accepted" and event["actor"] == seat
            )
            for other in state.rivers:
                for discard in other:
                    if (
                        discard["event_index"] > declared_index
                        and discard["resolution"]
                    ):
                        packed["genbutsu_to_seat"][
                            relative, shanten.base_id(discard["pai"])
                        ] = 1
        for river_index, row in enumerate(state.rivers[seat]):
            if last_tedashi is not None and any(
                last_tedashi[2] < call_index < row["event_index"]
                for call_index in call_indices
            ):
                last_tedashi = None
            tile = shanten.base_id(row["pai"])
            packed["genbutsu_to_seat"][relative, tile] = 1
            if river_index < 6 and not row["riichi"] and tile < 27:
                for outside in OUTSIDE[tile % 9]:
                    target = tile // 9 * 9 + outside
                    if not packed["sotogawa_earliest_turn_to_seat"][
                        relative, target
                    ]:
                        packed["sotogawa_earliest_turn_to_seat"][
                            relative, target
                        ] = (river_index + 1)
            if relative == 0:
                continue
            # Independently left-pad each chronological
            # segment. The declaration discard remains in the pre river.
            phase = "pre" if river_index < riichi_boundary else "post"
            end = riichi_boundary if phase == "pre" else len(state.rivers[seat])
            index = 32 - end + river_index
            if index < 0:
                continue
            values = {
                "tile_id": tile,
                "is_red": shanten.is_red(row["pai"]),
                "tsumogiri": row["tsumogiri"],
                "riichi": row["riichi"],
                "valid": 1,
                "origin": (
                    1
                    if seat == state.start["oya"] and river_index == 0
                    else (
                        4
                        if row["origin"] == "chi"
                        else (
                            5
                            if row["origin"] == "pon"
                            else 2 if row["tsumogiri"] else 3
                        )
                    )
                ),
                "call_resolution": row["resolution"] + 1,
                "caller": (
                    1
                    if row["resolution"] == 0
                    else (
                        0
                        if row["caller"] is None
                        else (row["caller"] - actor) % 4 + 2
                    )
                ),
                "event_age": min(
                    1, (len(state.history) - row["event_index"]) / 100
                ),
                "own_gap": min(
                    1, (len(state.rivers[actor]) - row["own_turns"][actor]) / 24
                ),
                "live_wall": row["live_wall"] / 70,
                "current_dora_multiplicity": packed["dora_multiplicity"][tile],
                "dora_multiplicity_at_discard": river_dora[
                    tuple(row["dora_markers"])
                ][tile],
            }
            if not row["tsumogiri"]:
                if last_tedashi is not None:
                    prior, turn, _ = last_tedashi
                    gap = abs(tile - prior)
                    values["tedashi_relation"] = (
                        1
                        if tile == prior
                        else (
                            gap + 1
                            if tile < 27
                            and tile // 9 == prior // 9
                            and gap <= 2
                            else 0
                        )
                    )
                    values["tedashi_gap"] = min(31, river_index - turn)
                last_tedashi = (tile, river_index, row["event_index"])
            for name, value in values.items():
                packed["river_" + phase + "_" + name][
                    relative - 1, index
                ] = value


def store_value_ranges(packed, action, values):
    packed["action_value_range"][action] = values["range"]
    packed["action_value_valid"][action] = values["valid"]
    packed["action_open_yaku_restricted"][action] = values["restricted"]
    packed["action_value_eligible"][action] = values["eligible"]


def candidate_features(packed, state, actor, legal_observation, *, stage="all"):
    own = legal_observation.hand
    counts = numpy.asarray(
        packed["candidate_hand_counts"][0], dtype=numpy.int64
    )
    reds = numpy.asarray(packed["candidate_hand_red"][0], dtype=numpy.int64)
    unavailable = packed["known_unavailable_counts"]
    for name in ("riichi_legal_mask", "response_legal_mask", "kan_legal_mask"):
        packed[name][0] = 1
    legal = legal_observation.legal_actions()
    for action in legal:
        if action.action_type == riichienv.ActionType.DISCARD:
            packed["discard_action_legal"][native_tile_code(action.tile)] = 1
        elif action.action_type == riichienv.ActionType.RIICHI:
            packed["riichi_legal_mask"][1] = 1
            declared = state.native_env.clone()
            declared.apply_event({"type": "reach", "actor": actor})
            for cut in declared.get_observation(actor).legal_actions():
                if cut.action_type == riichienv.ActionType.DISCARD:
                    packed["riichi_discard_legal_mask"][
                        native_tile_code(cut.tile)
                    ] = 1
    held = [native_tile_code(tile) for tile in own]
    if packed["current_draw_valid"]:
        held.remove(drawn_tile_code(packed))
    for code in set(held):
        packed["discard_tedashi_legal"][code] = (
            packed["discard_action_legal"][code] and not state.accepted[actor]
        )
    packed["candidate_melds"][0] = hand_melds(state.melds[actor])
    packed["candidate_yaku_shanten"][0] = shanten.calculate_yaku_shanten(
        counts,
        [[tile // 4 for tile in meld.tiles] for meld in state.melds[actor]],
    )
    packed["candidate_hand_valid"][0] = 1
    bank = {
        (
            counts.tobytes(),
            reds.tobytes(),
            packed["candidate_melds"][0].tobytes(),
        ): 0
    }

    def add_hand(hand, melds):
        candidate_counts = numpy.bincount(
            [tile // 4 for tile in hand], minlength=34
        )
        candidate_red = numpy.array([hand.count(tile) for tile in (16, 52, 88)])
        candidate_melds = hand_melds(melds)
        key = (
            candidate_counts.tobytes(),
            candidate_red.tobytes(),
            candidate_melds.tobytes(),
        )
        if key not in bank:
            index = len(bank)
            bank[key] = index
            packed["candidate_hand_counts"][index] = candidate_counts
            packed["candidate_hand_red"][index] = candidate_red
            packed["candidate_melds"][index] = candidate_melds
            packed["candidate_yaku_shanten"][index] = (
                shanten.calculate_yaku_shanten(
                    candidate_counts,
                    [[tile // 4 for tile in meld.tiles] for meld in melds],
                )
            )
            packed["candidate_hand_valid"][index] = 1
        return bank[key]

    context = {
        "own_discards": [
            shanten.base_id(row["pai"]) for row in state.rivers[actor]
        ],
        "missed_agari": state.missed_agari(actor),
        "unavailable_counts": unavailable,
        "unavailable_red": reds + packed["public_visible_red_counts"],
        "dora_indicators": legal_observation.dora_indicators,
        "player_wind": int(packed["seat_wind"][0]),
        "round_wind": int(packed["prevailing_wind"]),
    }
    cuts = numpy.flatnonzero(packed["discard_action_legal"])
    if len(cuts):
        decline = {
            "range": packed["action_value_range"][260],
            "valid": packed["action_value_valid"][260],
            "restricted": packed["action_open_yaku_restricted"][260],
            "eligible": 0,
        }
        if stage in {"all", "yaku"}:
            discard_yaku = yaku_flags(
                state.native_env.get_observation(actor), actor
            )
        if stage in {"all", "analysis"}:
            analysis = scoring_reference.analyze_actions(
                counts=counts,
                tile_ids=cuts,
                known_unavailable_counts=unavailable,
                calculate_upgrades=False,
                calculate_completion=False,
            )
            for mode, name in enumerate(("normal", "all")):
                for suffix in (
                    "shanten",
                    "ukeire_count",
                ):
                    packed[f"discard_action_{name}_{suffix}"][cuts] = analysis[
                        mode
                    ][suffix][shanten.BASE_IDS[cuts]] + (suffix == "shanten")
        for code in sorted(
            cuts, key=lambda code: (shanten.base_id(code), code < 34)
        ):
            base = shanten.base_id(code)
            hand = list(own)
            hand.remove(
                next(tile for tile in hand if native_tile_code(tile) == code)
            )
            packed["discard_candidate_hand"][code] = add_hand(
                hand, state.melds[actor]
            )
            if stage in {"all", "yaku"}:
                # The public yaku encoder uses melds and terminal/honor
                # visibility, not concealed tiles. Discarding a simple
                # leaves every input to its checks unchanged.
                if base < 27 and 1 <= base % 9 <= 7:
                    packed["discard_action_yaku_possibility"][
                        code
                    ] = discard_yaku
                else:
                    after_discard = state.native_env.clone()
                    after_discard.apply_event(
                        {
                            "type": "dahai",
                            "actor": actor,
                            "pai": shanten.TILE_NAMES[code],
                            "tsumogiri": bool(
                                packed["current_draw_valid"]
                                and code == drawn_tile_code(packed)
                            ),
                        }
                    )
                    packed["discard_action_yaku_possibility"][code] = (
                        yaku_flags(after_discard.get_observation(actor), actor)
                    )
            if stage in {"all", "scoring"}:
                values = scoring_reference.action_value_ranges(
                    hand, state.melds[actor], context, base
                )
                store_value_ranges(packed, code, values)
                store_value_ranges(packed, 37 + code, values)
                if packed["riichi_discard_legal_mask"][code]:
                    store_value_ranges(packed, 74 + code, values)
                scoring_reference.merge_value_ranges(decline, values)
            packed["discard_retained_genbutsu"][code] = packed[
                "genbutsu_to_seat"
            ] @ (counts - (numpy.arange(34) == base))
        if stage in {"all", "scoring"}:
            store_value_ranges(packed, 260, decline)
        if stage in {"all", "analysis"}:
            best = min(
                cuts,
                key=lambda code: (
                    analysis[1]["shanten"][shanten.base_id(code)],
                    -analysis[1]["ukeire_count"][shanten.base_id(code)],
                    -int(
                        analysis[1]["ukeire_mask"][shanten.base_id(code)]
                    ).bit_count(),
                ),
            )
            base = shanten.base_id(best)
            packed["self_efficiency"][:] = [
                analysis[1]["shanten"][base] / 8,
                int(analysis[1]["ukeire_mask"][base]).bit_count() / 34,
                analysis[1]["ukeire_count"][base] / 80,
            ]
    else:
        if stage in {"all", "analysis"}:
            s, improving_tiles, ukeire = shanten.improving(counts, unavailable)
            packed["self_efficiency"][:] = [
                s / 8,
                len(improving_tiles) / 34,
                ukeire / 80,
            ]
        if stage in {"all", "scoring"}:
            values = scoring_reference.action_value_ranges(
                own, state.melds[actor], context
            )
            store_value_ranges(packed, 111, values)
            store_value_ranges(packed, 260, values)
    call_features(
        packed=packed,
        state=state,
        actor=actor,
        context=context,
        add_hand=add_hand,
        stage=stage,
    )


def call_features(packed, state, actor, context, add_hand, *, stage="all"):
    legal = prefer_red_calls(state.legal_observation(actor).legal_actions())
    unavailable = packed["known_unavailable_counts"]
    for action in legal:
        match action.action_type:
            case (
                riichienv.ActionType.CHI
                | riichienv.ActionType.PON
                | riichienv.ActionType.DAIMINKAN
            ):
                # Native call values are the response-mask indices 1/2/3.
                response = int(action.action_type)
            case riichienv.ActionType.ANKAN | riichienv.ActionType.KAKAN:
                response = 0
            case _:
                continue
        event = {**action_event(action), "actor": actor}
        if response:
            event["target"] = state.pending_discard["actor"]
            packed["response_legal_mask"][response] = 1
        env = state.native_env.clone()
        env.apply_event(event)
        next_observation = env.get_observation(actor)
        hand = list(next_observation.hand)
        if response not in (1, 2):
            packed["kan_legal_mask"][1 + action.tile // 4] = 1
            packed["kan_candidate_hand"][action.tile // 4] = add_hand(
                hand, env.melds[actor]
            )
            if stage in {"all", "scoring"}:
                store_value_ranges(
                    packed,
                    261 + action.tile // 4,
                    scoring_reference.action_value_ranges(
                        hand, env.melds[actor], context
                    ),
                )
            if stage in {"all", "analysis"}:
                s, tiles, ukeire = shanten.improving(
                    numpy.bincount(numpy.array(hand) // 4, minlength=34),
                    unavailable,
                )
                packed["kan_efficiency"][action.tile // 4] = [
                    s / 8,
                    len(tiles) / 34,
                    ukeire / 80,
                ]
            if stage in {"all", "yaku"}:
                packed["kan_yaku_possibility"][action.tile // 4] = yaku_flags(
                    next_observation, actor
                )
            continue
        if response == 1:
            prefix = "chi"
            option = chii_pattern(action.tile, action.consume_tiles)
            packed["chi_pattern_legal_mask"][option] = 1
        else:
            prefix = "pon"
            option = 0
        call_cuts = sorted(
            set(
                native_tile_code(cut.tile)
                for cut in next_observation.legal_actions()
                if cut.action_type == riichienv.ActionType.DISCARD
            )
        )
        call_counts = numpy.bincount(numpy.array(hand) // 4, minlength=34)
        if stage in {"all", "analysis"}:
            analysis = scoring_reference.analyze_actions(
                call_counts, call_cuts, unavailable, modes=(1,)
            )[1]
        for code in sorted(
            call_cuts, key=lambda code: (shanten.base_id(code), code < 34)
        ):
            cut_hand = list(hand)
            cut_hand.remove(
                next(tile for tile in hand if native_tile_code(tile) == code)
            )
            packed[prefix + "_candidate_hand"][option, code] = add_hand(
                cut_hand, env.melds[actor]
            )
            packed[prefix + "_discard_legal"][option, code] = 1
            if stage in {"all", "analysis"}:
                packed[prefix + "_discard_shanten"][option, code] = (
                    analysis["shanten"][shanten.base_id(code)] + 1
                )
                packed[prefix + "_discard_ukeire_count"][option, code] = (
                    analysis["ukeire_count"][shanten.base_id(code)]
                )
            if stage in {"all", "scoring"}:
                store_value_ranges(
                    packed,
                    (112 if response == 1 else 223) + option * 37 + code,
                    scoring_reference.action_value_ranges(
                        cut_hand,
                        env.melds[actor],
                        context,
                        shanten.base_id(code),
                    ),
                )
        if call_cuts and stage in {"all", "analysis"}:
            best = min(
                (shanten.base_id(code) for code in call_cuts),
                key=lambda base: (
                    analysis["shanten"][base],
                    -analysis["ukeire_count"][base],
                    -int(analysis["ukeire_mask"][base]).bit_count(),
                ),
            )
            packed[prefix + "_call_efficiency"][option] = [
                analysis["shanten"][best] / 8,
                int(analysis["ukeire_mask"][best]).bit_count() / 34,
                analysis["ukeire_count"][best] / 80,
            ]
        if stage in {"all", "yaku"}:
            packed[prefix + "_call_yaku_possibility"][option] = yaku_flags(
                next_observation, actor
            )
