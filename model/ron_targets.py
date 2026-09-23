"""Known ron and abortive-draw payments, used only as training labels."""

import numpy
import riichienv

from log_dataset import shanten

from . import features, outcomes


def known_returns(state, actor, payments):
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
        # Houou hanchan: bankruptcy, or all-last/extension completion.
        # Evaluate the 30k threshold before awarding outstanding sticks.
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


def abort_labels(state, actor, phase, payments):
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
            # Ron takes priority. Unknown ura/recipients cannot prove safety.
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


def labels(state, actor, phase):
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
                # Unrevealed ura and responsibility transfers are not known.
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
        # Tenhou triple ron is an abortive draw, not three payouts.
        payments[:, (payments > 0).sum(axis=0) == 3] = -1
    value, valid = known_returns(state, actor, payments)
    return {
        **abort_labels(state, actor, phase, payments),
        "ron_payments": payments,
        "ron_return": value,
        "ron_return_valid": valid,
    }
