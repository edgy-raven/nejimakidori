"""Actor-relative public tensors and deterministic post-action hand banks."""

import json

import numpy
import riichienv

import feature_vector
from log_dataset import shanten

from . import outcomes, tenpai

PHASES = {"DISCARD": 0, "RESPONSE": 1, "KAN": 2, "RIICHI": 3, "WIN": 4}
OUTSIDE = ((), (0,), (0, 1), (1, 2), (2, 6), (6, 7), (7, 8), (8,), ())


def native_tile_code(tile):
    return 34 + (tile // 4) // 9 if tile in (16, 52, 88) else tile // 4


def action_event(action, drawn_tile=None):
    event = json.loads(action.to_mjai())
    if event["type"] == "dahai":
        event["tsumogiri"] = action.tile == drawn_tile
    return event


def single_legal_action(legal, drawn_tile):
    """Physical copies of the same held tile are one available action."""
    events = (action_event(action, drawn_tile) for action in legal)
    first = next(events, None)
    if first is not None and all(event == first for event in events):
        return legal[0]
    return None


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


def preferred_recorded_call(kind, consumed, legal):
    return any(
        event["type"] == kind and sorted(event["consumed"]) == sorted(consumed)
        for action in prefer_red_calls(legal)
        if action.action_type
        in (riichienv.ActionType.CHI, riichienv.ActionType.PON)
        for event in [action_event(action)]
    )


def discard_tile_code(action, packed):
    if action != 37:
        return int(action)
    tile = int(packed["current_draw_tile_id"])
    return 34 + tile // 9 if packed["current_draw_is_red"] else tile


def discard_legal(packed):
    return numpy.r_[
        packed["discard_tedashi_legal"],
        int(
            packed["current_draw_valid"]
            and packed["discard_action_legal"][discard_tile_code(37, packed)]
        ),
    ]


def chi_pattern(called, consumed):
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


def yaku_flags(observation, actor):
    flags = numpy.frombuffer(
        observation.encode_yaku_possibility(), numpy.float32
    )
    return flags.reshape(4, 21, 2)[actor, list(range(18)) + [19], 0].astype(
        numpy.int8
    )


def hand_melds(melds):
    result = numpy.zeros((34, 6), numpy.int8)
    for meld in melds:
        result[min(tile // 4 for tile in meld.tiles), int(meld.meld_type)] += 1
        for tile in meld.tiles:
            if tile in (16, 52, 88):
                result[tile // 4, 5] += 1
    return result


def pack(state, actor, phase, legal_observation, east_only=False):
    packed = feature_vector.FeatureVector().tensors
    own = legal_observation.hand
    counts = numpy.bincount([tile // 4 for tile in own], minlength=34)
    reds = numpy.array([own.count(tile) for tile in (16, 52, 88)])
    packed["candidate_hand_counts"][0] = counts
    packed["candidate_hand_red"][0] = reds
    packed["closed_hand"][...] = all(
        not meld.opened for meld in state.melds[actor]
    )
    for name, value in zip(
        ("normal", "chiitoitsu", "kokushi"), shanten.calculate_families(counts)
    ):
        packed[name + "_self_shanten"][...] = (
            5 if name != "normal" and state.melds[actor] else min(4, value + 1)
        )
    packed["self_yaku_possibility"][:] = yaku_flags(legal_observation, actor)
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
    hand_number = wind * 4 + state.start["kyoku"] - 1
    final_hand = 3 if east_only else 7
    packed["nominal_all_last"][...] = hand_number == final_hand
    packed["scheduled_hands_remaining"][...] = max(0, final_hand - hand_number)
    packed["dealer_currently_leads"][...] = (
        scores[state.start["oya"]] == scores.max()
    )
    packed["any_player_at_or_above_30000"][...] = scores.max() >= 30000
    table_features(packed, state, actor, legal_observation)
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
    candidate_features(packed, state, actor, legal_observation)
    return packed


def table_features(packed, state, actor, legal_observation):
    scores = numpy.array(legal_observation.scores)
    ranks = outcomes.score_ranks(
        scores,
        (
            (state.start["oya"] - state.start["kyoku"] + 1) % 4
            if state.hanchan is None
            else state.hanchan["first_dealer"]
        ),
    )
    packed["score_extrema"][:] = [scores.min() / 100000, scores.max() / 100000]
    for relative in range(4):
        seat = (actor + relative) % 4
        packed["seat_wind"][relative] = (seat - state.start["oya"]) % 4
        packed["seat_context"][relative] = [
            scores[seat] / 100000,
            ranks[seat] / 4,
        ]
        packed["riichi_state"][relative] = int(state.declared[seat]) + int(
            state.accepted[seat]
        )
        packed["ippatsu_alive"][relative] = state.ippatsu[seat]
        if relative:
            packed["opponent_called_tile_count"][relative] = sum(
                len(meld.tiles) for meld in state.melds[seat] if meld.opened
            )
            packed["opponent_discard_count"][relative] = len(state.rivers[seat])
            if any(meld.opened for meld in state.melds[seat]):
                packed["open_opponent_yaku_possibility"][relative] = yaku_flags(
                    legal_observation, seat
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
        last_tedashi = None
        call_indices = [
            event[key]
            for event in state.meld_events[seat]
            for key in ("event_index", "upgrade_index")
            if key in event
        ]
        for river_index, row in enumerate(state.rivers[seat]):
            if last_tedashi is not None and any(
                last_tedashi[2] < call_index < row["event_index"]
                for call_index in call_indices
            ):
                last_tedashi = None
            tile = shanten.base_id(row["pai"])
            packed["genbutsu_to_seat"][relative, tile] = 1
            if state.accepted[seat]:
                declared_index = next(
                    i
                    for i, event in enumerate(state.history)
                    if event["type"] == "reach_accepted"
                    and event["actor"] == seat
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
            if river_index < 6 and not row["riichi"] and tile < 27:
                for outside in OUTSIDE[tile % 9]:
                    target = tile // 9 * 9 + outside
                    if not packed["sotogawa_earliest_turn_to_seat"][
                        relative, target
                    ]:
                        packed["sotogawa_earliest_turn_to_seat"][
                            relative, target
                        ] = (river_index + 1)
            index = 32 - len(state.rivers[seat]) + river_index
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
                "dora_multiplicity_at_discard": dora_counts(
                    row["dora_markers"]
                )[tile],
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
                packed["river_" + name][relative, index] = value


def candidate_features(packed, state, actor, legal_observation):
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
        held.remove(discard_tile_code(37, packed))
    for code in set(held):
        packed["discard_tedashi_legal"][code] = (
            packed["discard_action_legal"][code] and not state.accepted[actor]
        )
    packed["candidate_melds"][0] = hand_melds(state.melds[actor])
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
        analysis = shanten.analyze_actions(
            counts,
            cuts,
            unavailable,
            calculate_upgrades=not state.accepted[actor],
            calculate_completion=True,
        )
        for code in sorted(
            cuts, key=lambda code: (shanten.base_id(code), code < 34)
        ):
            hand = list(own)
            hand.remove(
                next(tile for tile in hand if native_tile_code(tile) == code)
            )
            packed["discard_candidate_hand"][code] = add_hand(
                hand, state.melds[actor]
            )
            for mode, name in enumerate(("normal", "all")):
                for suffix in (
                    "shanten",
                    "ukeire_count",
                    "upgrade_type_count",
                    "upgrade_tile_count",
                    "upgrade_weighted_gain",
                ):
                    packed[f"discard_action_{name}_{suffix}"][code] = analysis[
                        mode
                    ][suffix][shanten.base_id(code)] + (suffix == "shanten")
                packed[f"discard_action_{name}_upgrade_valid"][code] = (
                    not state.accepted[actor]
                )
            packed["discard_completion_probability"][code] = analysis[1][
                "completion_probability"
            ][shanten.base_id(code)]
            packed["discard_completion_valid"][code] = analysis[1][
                "completion_valid"
            ][shanten.base_id(code)]
            after_discard = state.native_env.clone()
            after_discard.apply_event(
                {
                    "type": "dahai",
                    "actor": actor,
                    "pai": shanten.TILE_NAMES[code],
                    "tsumogiri": bool(
                        packed["current_draw_valid"]
                        and code == discard_tile_code(37, packed)
                    ),
                }
            )
            packed["discard_action_yaku_possibility"][code] = yaku_flags(
                after_discard.get_observation(actor), actor
            )
            if analysis[1]["shanten"][shanten.base_id(code)] == 0:
                packed["discard_tenpai_values"][code] = tenpai.score_waits(
                    hand,
                    state.melds[actor],
                    {
                        **context,
                        "own_discards": context["own_discards"]
                        + [shanten.base_id(code)],
                    },
                )[0]
            packed["discard_retained_genbutsu"][code] = packed[
                "genbutsu_to_seat"
            ] @ (counts - (numpy.arange(34) == shanten.base_id(code)))
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
        s, improving_tiles, ukeire = shanten.improving(counts, unavailable)
        packed["self_efficiency"][:] = [
            s / 8,
            len(improving_tiles) / 34,
            ukeire / 80,
        ]
        if s == 0:
            packed["self_tenpai_values"][:] = tenpai.score_waits(
                own, state.melds[actor], context
            )[0]
    call_features(
        packed=packed,
        state=state,
        actor=actor,
        context=context,
        add_hand=add_hand,
    )


def call_features(packed, state, actor, context, add_hand):
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
            s, tiles, ukeire = shanten.improving(
                numpy.bincount(numpy.array(hand) // 4, minlength=34),
                unavailable,
            )
            packed["kan_efficiency"][action.tile // 4] = [
                s / 8,
                len(tiles) / 34,
                ukeire / 80,
            ]
            packed["kan_yaku_possibility"][action.tile // 4] = yaku_flags(
                next_observation, actor
            )
            continue
        if response == 1:
            prefix = "chi"
            option = chi_pattern(action.tile, action.consume_tiles)
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
        analysis = shanten.analyze_actions(
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
            packed[prefix + "_discard_shanten"][option, code] = (
                analysis["shanten"][shanten.base_id(code)] + 1
            )
            packed[prefix + "_discard_ukeire_count"][option, code] = analysis[
                "ukeire_count"
            ][shanten.base_id(code)]
            if analysis["shanten"][shanten.base_id(code)] == 0:
                after_cut = {
                    **context,
                    "own_discards": context["own_discards"]
                    + [shanten.base_id(code)],
                }
                packed[prefix + "_call_tenpai_values"][option, code] = (
                    tenpai.score_waits(cut_hand, env.melds[actor], after_cut)[
                        0
                    ][:2]
                )
                if packed["closed_hand"]:
                    packed[prefix + "_call_closed_riichi_values"][
                        option, code
                    ] = tenpai.score_waits(
                        cut_hand + list(env.melds[actor][-1].tiles),
                        state.melds[actor],
                        after_cut,
                    )[
                        0
                    ][
                        2:
                    ]
        if call_cuts:
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
        packed[prefix + "_call_yaku_possibility"][option] = yaku_flags(
            next_observation, actor
        )
