"""Complete-round labels separated from decision-time visible inputs."""

import numpy
import riichienv

from log_dataset import shanten

from . import completion_yaku, features, outcomes, records, tenpai


def known_ron_returns(state, actor, payments):
    """Convert complete terminal transfers to the existing 2/1/0 utility."""
    result = numpy.zeros(37, numpy.float32)
    valid = numpy.zeros(37, numpy.int8)
    order = [(actor + offset) % 4 for offset in range(4)]
    dealerships = outcomes.dealerships_remaining(state.start, actor)
    first_dealer = (
        state.hanchan["first_dealer"]
        if state.hanchan is not None
        else (state.start["oya"] - state.start["kyoku"] + 1) % 4
    )
    for tile in numpy.flatnonzero(
        (payments >= 0).all(axis=0) & (payments.sum(axis=0) > 0)
    ):
        scores = numpy.array(state.legal_observation(actor).scores)
        scores[actor] -= payments[:, tile].sum()
        scores[order[1:]] += payments[:, tile]
        ranks = outcomes.score_ranks(scores, first_dealer)
        winners = [
            order[index + 1] for index in range(3) if payments[index, tile] > 0
        ]
        dealer_won = state.start["oya"] in winners
        hand = (
            4 * "ESWN".index(state.start["bakaze"]) + state.start["kyoku"] - 1
        )
        ended = scores.min() < 0 or (
            hand >= 7
            and scores.max() >= 30000
            and (not dealer_won or ranks[state.start["oya"]] == 1)
        )
        scores[winners[0]] += 1000 * state.native_env.riichi_sticks
        if dealerships >= 2 or ended:
            result[tile] = outcomes.dealership_utility(
                -int(payments[:, tile].sum()),
                outcomes.score_ranks(scores, first_dealer)[actor],
                dealerships,
            )
            valid[tile] = 1
    return result, valid


def abort_ron_labels(state, actor, phase, payments):
    """Safe discards can end four-riichi/four-kan hands with no transfers."""
    points = numpy.full((2, 37), -1, numpy.int32)
    owners = [
        seat
        for seat, melds in enumerate(state.melds)
        for meld in melds
        if meld.meld_type
        in (
            riichienv.MeldType.Ankan,
            riichienv.MeldType.Kakan,
            riichienv.MeldType.Daiminkan,
        )
    ]
    four_kans = len(owners) == 4 and len(set(owners)) > 1
    for declaration in range(2 if phase == "RIICHI" else 1):
        four_riichi = all(
            state.declared[seat] or (seat == actor and declaration)
            for seat in range(4)
        )
        if phase in ("DISCARD", "RIICHI") and (four_kans or four_riichi):
            points[declaration, (payments == 0).all(axis=0)] = -1000 * int(
                (declaration or state.declared[actor])
                and not state.accepted[actor]
            )
    return {
        "abort_points": points,
        "abort_return_valid": (
            (points != -1)
            & (outcomes.dealerships_remaining(state.start, actor) >= 2)
        ).astype(numpy.int8),
    }


def ron_labels(state, actor, phase):
    """Known ron and abortive-draw payments for one decision."""
    payments = numpy.full((3, 37), -1, numpy.int32)
    if phase in ("DISCARD", "RIICHI"):
        cuts = {
            features.native_tile_code(action.tile): action.tile
            for action in state.legal_observation(actor).legal_actions()
            if action.action_type == riichienv.ActionType.DISCARD
        }
        payments[:, list(cuts)] = 0
        ura = next(
            (
                event["ura_markers"]
                for event in state.events
                if event["type"] == "hora" and event["ura_markers"]
            ),
            [],
        )
        for relative in range(3):
            opponent = (actor + relative + 1) % 4
            hand = state.legal_observation(opponent).hand
            if len(hand) % 3 != 1:
                continue
            waits = riichienv.HandEvaluator(
                hand, state.melds[opponent]
            ).get_waits()
            if (
                not waits
                or state.missed_agari(opponent)
                or set(waits)
                & {
                    shanten.base_id(row["pai"])
                    for row in state.rivers[opponent]
                }
            ):
                continue
            for code, tile in cuts.items():
                if tile // 4 not in waits:
                    continue
                if (
                    state.accepted[opponent]
                    and len(ura) < len(state.dora_markers)
                ) or state.native_env.pao[opponent]:
                    payments[relative, code] = -1
                    continue
                score = riichienv.HandEvaluator(
                    hand + [tile], state.melds[opponent]
                ).calc(
                    win_tile=tile,
                    dora_indicators=state.legal_observation(
                        opponent
                    ).dora_indicators,
                    conditions=riichienv.Conditions(
                        riichi=state.accepted[opponent],
                        double_riichi=(
                            state.accepted[opponent]
                            and state.rivers[opponent][0]["riichi"]
                            and not any(
                                event["type"]
                                in {"chi", "pon", "daiminkan", "ankan", "kakan"}
                                for event in state.history[
                                    : state.rivers[opponent][0]["event_index"]
                                ]
                            )
                        ),
                        ippatsu=state.ippatsu[opponent],
                        houtei=state.live_wall == 0,
                        player_wind=(opponent - state.start["oya"]) % 4,
                        round_wind="ESWN".index(state.start["bakaze"]),
                    ),
                    ura_indicators=(
                        [
                            shanten.base_id(marker) * 4
                            for marker in ura[: len(state.dora_markers)]
                        ]
                        if state.accepted[opponent]
                        else []
                    ),
                )
                if score.is_win:
                    payments[relative, code] = (
                        score.ron_agari + 300 * state.start["honba"]
                    )
        payments[:, (payments > 0).sum(axis=0) == 3] = -1
    value, valid = known_ron_returns(state, actor, payments)
    return {
        **abort_ron_labels(state, actor, phase, payments),
        "ron_payments": payments,
        "ron_return": value,
        "ron_return_valid": valid,
    }


def round_targets(state, events):
    length = len(events)
    draws = numpy.zeros((length, 4), int)
    ready = numpy.zeros((length, 4), bool)
    tenpai_value = numpy.zeros(4, numpy.float32)
    hand_yaku_focus = numpy.zeros(4, numpy.int8)
    dangers = numpy.zeros((length, 4, 37), numpy.float32)
    completion = numpy.full((4, len(completion_yaku.NAMES)), -1, numpy.int8)
    opportunities = []
    for index, event in enumerate(events):
        if event["type"] == "dahai" and not state.accepted[event["actor"]]:
            actor = event["actor"]
            observation = state.legal_observation(actor)
            codes = sorted(
                set(
                    features.native_tile_code(action.tile)
                    for action in observation.legal_actions()
                    if action.action_type == riichienv.ActionType.DISCARD
                )
            )
            visible = state.public_tiles()
            counts = state.concealed_counts(actor)
            unavailable = counts + numpy.bincount(
                [shanten.base_id(tile) for tile in visible], minlength=34
            )
            opportunities.append(
                (
                    index,
                    actor,
                    counts,
                    unavailable,
                    codes,
                    shanten.mjai_tile_code(event["pai"]),
                )
            )
        if index:
            state.apply(event)
            draws[index] = draws[index - 1]
        if event["type"] == "tsumo":
            draws[index, event["actor"]] += 1
        for actor in range(4):
            counts = state.concealed_counts(actor)
            if counts.sum() % 3 != 1:
                continue
            ready[index, actor] = shanten.calculate_shanten(counts) == 0
            if index == 0 or (
                event.get("actor") == actor
                and event["type"] in {"dahai", "ankan", "kakan", "daiminkan"}
            ):
                observation = state.legal_observation(actor)
                flags = completion_yaku.completion_flags(
                    observation.hand,
                    state.melds[actor],
                    (actor - state.start["oya"]) % 4,
                    "ESWN".index(state.start["bakaze"]),
                )
                completion[actor] = numpy.maximum(completion[actor], flags)
                if ready[index, actor] and not hand_yaku_focus[actor]:
                    hand_yaku_focus[actor] = any(
                        flag > 0 and name in completion_yaku.FOCUSED_NAMES
                        for name, flag in zip(
                            completion_yaku.NAMES,
                            completion_yaku.completion_flags(
                                tiles=observation.hand,
                                melds=state.melds[actor],
                                player_wind=(actor - state.start["oya"]) % 4,
                                round_wind="ESWN".index(state.start["bakaze"]),
                                require_all=False,
                            ),
                        )
                    )
            if not ready[index, actor]:
                continue
            observation = state.legal_observation(actor)
            visible = state.public_tiles()
            _, waits = tenpai.score_waits(
                observation.hand,
                state.melds[actor],
                {
                    "own_discards": [
                        shanten.base_id(row["pai"])
                        for row in state.rivers[actor]
                    ],
                    "missed_agari": state.missed_agari(actor),
                    "unavailable_counts": counts
                    + numpy.bincount(
                        [shanten.base_id(tile) for tile in visible],
                        minlength=34,
                    ),
                    "unavailable_red": [
                        observation.hand.count(tile) + visible.count(name)
                        for tile, name in zip(
                            (16, 52, 88), ("5mr", "5pr", "5sr")
                        )
                    ],
                    "dora_indicators": observation.dora_indicators,
                    "player_wind": (actor - state.start["oya"]) % 4,
                    "round_wind": "ESWN".index(state.start["bakaze"]),
                },
            )
            if not ready[:index, actor].any():
                tenpai_value[actor] = (
                    sum(
                        wait["unseen"]
                        * wait["values"][2 if state.declared[actor] else 0][
                            "points"
                        ]
                        for wait in waits
                    )
                    / max(1, sum(wait["unseen"] for wait in waits))
                    / 10000
                )
            for wait in waits:
                dangers[index, actor, wait["tile"]] = (
                    wait["values"][2 if state.accepted[actor] else 0]["points"]
                    / 10000
                    if wait["unseen"] > 0
                    else 0
                )
        if event["type"] == "hora":
            ready[index, event["actor"]] = True
    cumulative = holding_costs(
        [row for row in opportunities if ready[row[0] :, row[1]].any()],
        dangers,
        draws,
        length,
    )
    return {
        "completion": completion,
        "hand_yaku_focus": hand_yaku_focus,
        "sakigiri": cumulative,
        "tenpai_reached": ready.any(axis=0).astype(numpy.int8),
        "tenpai_value": tenpai_value,
        "outcomes": outcomes.hand_outcomes(events),
        "danger": dangers,
        "payments": outcomes.terminal_payments(events),
        "scores": outcomes.terminal_scores(events),
    }


def attach(observation, state, index, action_index):
    actor = observation.actor
    order = [(actor + offset) % 4 for offset in range(4)]
    dealerships = outcomes.dealerships_remaining(state.start, actor)
    observation.dealership_return_valid = int(
        dealerships >= 2 or state.hanchan is not None
    )
    observation.dealership_return = (
        outcomes.dealership_utility(
            state.targets["scores"][actor]
            - state.legal_observation(actor).scores[actor],
            0 if dealerships >= 2 else state.hanchan["ranks"][actor],
            dealerships,
        )
        if observation.dealership_return_valid
        else 0
    )
    observation.tenpai_reached = state.targets["tenpai_reached"][actor]
    observation.tenpai_value = state.targets["tenpai_value"][actor]
    observation.specialist_outcomes = state.targets["outcomes"][actor].copy()
    observation.defense_ron = state.targets["danger"][index, order[1:]].sum(0)
    observation.terminal_payment = outcomes.payment_classes(
        outcomes.relative_payments(
            state.targets["payments"], actor, state.accepted
        )
    )
    observation.opponent_shanten = numpy.full(3, -1, numpy.int8)
    observation.opponent_ukeire = numpy.full((3, 34), -1, numpy.int8)
    observation.opponent_hand_counts = numpy.zeros((3, 37), numpy.int8)
    for relative, seat in enumerate(order[1:]):
        hand = state.legal_observation(seat).hand
        for tile in hand + [
            tile for meld in state.melds[seat] for tile in meld.tiles
        ]:
            observation.opponent_hand_counts[
                relative, features.native_tile_code(tile)
            ] += 1
        if len(hand) % 3 == 1:
            distance, improving, _ = shanten.improving(
                state.concealed_counts(seat)
            )
            observation.opponent_shanten[relative] = min(3, distance)
            observation.opponent_ukeire[relative] = 0
            observation.opponent_ukeire[relative, improving] = 1
    observation.round_placement = (
        numpy.array(
            outcomes.score_ranks(
                state.targets["scores"],
                (state.start["oya"] - state.start["kyoku"] + 1) % 4,
            )
        )[order]
        - 1
    )
    if state.hanchan is not None:
        observation.final_placement = state.hanchan["ranks"][actor] - 1
    observation.round_completion_yaku = state.targets["completion"][order]
    observation.hand_yaku_focus = state.targets["hand_yaku_focus"][actor]
    observation.sakigiri_cost = state.targets["sakigiri"][index, actor]

    for relative, seat in enumerate(order[1:]):
        observation.opponent_high_value[relative] = high_value(state, seat)
    if (
        observation.tenpai_reached
        and state.events[action_index]["type"] == "dahai"
        and state.draws[actor] is not None
        and not state.accepted[actor]
        and not any(state.declared[seat] for seat in order[1:])
        and not observation.opponent_high_value.any()
        and not observation.defense_ron.any()
    ):
        counts = observation.features["candidate_hand_counts"][0].astype(
            numpy.int32
        )
        distance, improving, _ = shanten.improving(
            counts - (numpy.arange(34) == shanten.base_id(state.draws[actor])),
            observation.features["known_unavailable_counts"],
        )
        retained_ukeire = shanten.improving(
            counts
            - (
                numpy.arange(34)
                == shanten.base_id(state.events[action_index]["pai"])
            ),
            observation.features["known_unavailable_counts"],
        )[1]
        missed_draws = []
        for event in state.events[action_index + 1 :]:
            if event.get("actor") != actor:
                continue
            if event["type"] in {
                "chi",
                "pon",
                "daiminkan",
                "ankan",
                "kakan",
            } or (event["type"] == "dahai" and not event["tsumogiri"]):
                break
            if event["type"] == "tsumo":
                draw = shanten.base_id(event["pai"])
                if draw in improving:
                    missed_draws.append(draw)
                # Include the draw that changes the retained hand's progress,
                # then end attribution to the old shape. Tsumogiri alone
                # does not end the window; passive visibility is not a change
                # in the hand's structural improving-tile set.
                if draw in retained_ukeire:
                    break
        if missed_draws:
            for cut in numpy.flatnonzero(
                features.discard_legal(observation.features)
            ):
                after = counts - (
                    numpy.arange(34)
                    == shanten.base_id(
                        features.discard_tile_code(cut, observation.features)
                    )
                )
                if shanten.calculate_shanten(after) == distance:
                    observation.discard_regret[cut] = numpy.mean(
                        ~numpy.isin(
                            missed_draws,
                            shanten.improving(
                                after,
                                observation.features[
                                    "known_unavailable_counts"
                                ],
                            )[1],
                        )
                    )
            if (
                numpy.unique(
                    observation.discard_regret[observation.discard_regret >= 0]
                ).size
                < 2
            ):
                observation.discard_regret[:] = -1


def high_value(state, actor):
    observation = state.legal_observation(actor)
    counts = state.concealed_counts(actor)
    if counts.sum() % 3 != 1 or shanten.calculate_shanten(counts) > 1:
        return False
    visible = state.public_tiles()
    unavailable = counts + numpy.bincount(
        [shanten.base_id(tile) for tile in visible], minlength=34
    )
    context = {
        "own_discards": [
            shanten.base_id(row["pai"]) for row in state.rivers[actor]
        ],
        "missed_agari": state.missed_agari(actor),
        "unavailable_counts": unavailable,
        "unavailable_red": [
            observation.hand.count(tile) + visible.count(name)
            for tile, name in zip((16, 52, 88), ("5mr", "5pr", "5sr"))
        ],
        "dora_indicators": observation.dora_indicators,
        "player_wind": (actor - state.start["oya"]) % 4,
        "round_wind": "ESWN".index(state.start["bakaze"]),
    }
    if shanten.calculate_shanten(counts) == 0:
        return any(
            row["unseen"] > 0 and row["values"][0]["points"] >= 7700
            for row in tenpai.score_waits(
                observation.hand, state.melds[actor], context
            )[1]
        )
    improving = shanten.improving(counts, unavailable)[1]
    occupied = {
        *observation.hand,
        *(tile for meld in state.melds[actor] for tile in meld.tiles),
    }
    for draw in range(37):
        base = shanten.base_id(draw)
        if base not in improving:
            continue
        unseen = (
            1 - context["unavailable_red"][draw - 34]
            if draw >= 34
            else 4
            - unavailable[base]
            - (
                1 - context["unavailable_red"][base // 9]
                if base in (4, 13, 22)
                else 0
            )
        )
        if unseen <= 0:
            continue
        hand = list(observation.hand) + [
            next(
                tile
                for tile in range(base * 4, base * 4 + 4)
                if tile not in occupied
                and features.native_tile_code(tile) == draw
            )
        ]
        after_draw = counts + (numpy.arange(34) == base)
        for cut in numpy.flatnonzero(after_draw):
            if shanten.calculate_shanten(
                after_draw - (numpy.arange(34) == cut)
            ):
                continue
            retained = list(hand)
            # A retained red copy cannot reduce the best reachable value.
            retained.remove(max(tile for tile in retained if tile // 4 == cut))
            _, waits = tenpai.score_waits(
                retained,
                state.melds[actor],
                {
                    **context,
                    "missed_agari": False,
                    "own_discards": context["own_discards"] + [int(cut)],
                    "unavailable_counts": unavailable
                    + (numpy.arange(34) == base),
                    "unavailable_red": numpy.asarray(context["unavailable_red"])
                    + (numpy.arange(34, 37) == draw),
                },
            )
            if any(
                row["unseen"] > 0 and row["values"][0]["points"] >= 7700
                for row in waits
            ):
                return True
    return False


def holding_costs(opportunities, dangers, draws, length):
    charges = numpy.zeros((length, 4), numpy.float32)
    for index, actor, counts, unavailable, codes, released in opportunities:
        other = [seat for seat in range(4) if seat != actor]
        future_danger = sum(
            numpy.max(
                (dangers[index:, seat] > 0)
                * numpy.power(
                    records.SAKIGIRI_TARGET["future_draw_discount"],
                    draws[index:, seat] - draws[index, seat],
                )[:, None],
                axis=0,
            )
            for seat in other
        )
        safe = dangers[index, other].sum(0) == 0
        charged = [
            code
            for code in codes
            if code != released and safe[code] and future_danger[code] > 0
        ]
        if not charged:
            continue
        participation = shanten.tenpai_participation(
            counts, unavailable, numpy.asarray(codes)
        )
        charges[index, actor] = (
            -1
            if (participation < 0).any()
            else sum(
                future_danger[code] * (1 - participation[shanten.base_id(code)])
                for code in charged
            )
        )
    cumulative = numpy.zeros_like(charges)
    for index in range(length - 1, -1, -1):
        if index + 1 < length:
            cumulative[index] = cumulative[index + 1]
        cumulative[index] = numpy.where(
            (charges[index] < 0) | (cumulative[index] < 0),
            -1,
            charges[index] + cumulative[index],
        )
    return cumulative
