"""Own-hand witnesses for fixed-continuation discard-order comparisons."""

import numpy
import riichienv

from log_dataset import replay, scoring, shanten
from model import features


def preserves_completion(steps, final_counts, late_cut):
    """Can the first discard and final discard exchange their order?

    Steps contain physical-code counts before/after each own discard,
    its cut, legal cuts and resulting shanten. final_counts is the hand
    before the last discard. The caller excludes intervening calls/kans
    and accepted own riichi, and checks endpoint ron exposures and legal
    live winning opportunities. Counts come from valid recorded hands.

    Require a continuously retained unique late tile and no worse
    intermediate shanten. The final concealed
    hand is identical. This does not predict opponents' responses or
    assert that the recorded draws survive the changed earlier discard.
    """
    if (
        steps[0]["cut"] == late_cut
        or late_cut not in steps[0]["legal"]
        or final_counts[late_cut] != 1
    ):
        return False
    delta = (
        numpy.eye(37, dtype=int)[steps[0]["cut"]]
        - numpy.eye(37, dtype=int)[late_cut]
    )
    for step in steps:
        if step["counts"][late_cut] != 1 or step["cut"] == late_cut:
            return False
        after = step["after"] + delta
        if (
            shanten.calculate_shanten(
                after[:34] + after[34:] @ numpy.eye(34, dtype=int)[[4, 13, 22]]
            )
            > step["distance"]
        ):
            return False
    return True


def ron_exposure(state, actor, codes):
    """Binary legal ron evidence, independent of payment magnitude."""
    result = dict.fromkeys(codes, 0)
    for opponent in range(4):
        if opponent == actor:
            continue
        hand = state.legal_observation(opponent).hand
        if len(hand) % 3 != 1:
            continue
        waits = riichienv.HandEvaluator(hand, state.melds[opponent]).get_waits()
        if state.missed_agari(opponent) or set(waits) & {
            shanten.base_id(row["pai"]) for row in state.rivers[opponent]
        }:
            continue
        for code in codes:
            base = shanten.base_id(code)
            if base not in waits:
                continue
            physical = next(
                tile
                for tile in range(base * 4, base * 4 + 4)
                if tile not in hand
            )
            result[code] += int(
                riichienv.HandEvaluator(
                    hand + [physical], state.melds[opponent]
                )
                .calc(
                    win_tile=physical,
                    conditions=riichienv.Conditions(
                        riichi=state.accepted[opponent],
                        houtei=state.live_wall == 0,
                        player_wind=(opponent - state.start["oya"]) % 4,
                        round_wind="ESWN".index(state.start["bakaze"]),
                    ),
                )
                .is_win
            )
    return result


def verified_exchange(game_id, events, actor, early, late):
    """Replay both orders: legal, safe early, same live winning opportunity."""
    states = [replay.Replay.from_start(game_id, events[0]) for _ in range(2)]
    risks = []
    for index, event in enumerate(events[1 : late + 1], 1):
        for alternate, state in enumerate(states):
            actual = event
            if alternate and index in (early, late):
                code = shanten.mjai_tile_code(
                    events[late if index == early else early]["pai"]
                )
                if code not in {
                    features.native_tile_code(action.tile)
                    for action in state.legal_observation(actor).legal_actions()
                    if action.action_type == riichienv.ActionType.DISCARD
                }:
                    return False
                actual = {
                    **event,
                    "pai": shanten.TILE_NAMES[code],
                    "tsumogiri": False,
                }
            state.apply(actual)
        if index in (early, late):
            risks.append(
                [
                    sum(
                        any(
                            action.action_type == riichienv.ActionType.RON
                            for action in state.native_env.get_observation(
                                opponent
                            ).legal_actions()
                        )
                        for opponent in range(4)
                        if opponent != actor
                    )
                    for state in states
                ]
            )
    if risks[0] != [0, 0] or 3 in risks[1]:
        return False
    if bool(risks[1][0]) == bool(risks[1][1]):
        return False
    for seat in range(4):
        assert sorted(
            features.native_tile_code(tile)
            for tile in states[0].native_env.hands[seat]
        ) == sorted(
            features.native_tile_code(tile)
            for tile in states[1].native_env.hands[seat]
        )
    assert sorted(row["pai"] for row in states[0].rivers[actor]) == sorted(
        row["pai"] for row in states[1].rivers[actor]
    )
    summaries = []
    for state in states:
        hand = state.legal_observation(actor).hand
        visible = state.public_tiles()
        summaries.append(
            scoring.score_waits(
                hand,
                state.melds[actor],
                {
                    "own_discards": [
                        shanten.base_id(row["pai"])
                        for row in state.rivers[actor]
                    ],
                    "missed_agari": state.missed_agari(actor),
                    "unavailable_counts": state.concealed_counts(actor)
                    + numpy.bincount(
                        [shanten.base_id(tile) for tile in visible],
                        minlength=34,
                    ),
                    "unavailable_red": [
                        hand.count(tile) + visible.count(name)
                        for tile, name in zip(
                            (16, 52, 88), ("5mr", "5pr", "5sr")
                        )
                    ],
                    "dora_indicators": state.legal_observation(
                        actor
                    ).dora_indicators,
                    "player_wind": (actor - state.start["oya"]) % 4,
                    "round_wind": "ESWN".index(state.start["bakaze"]),
                },
            )[0]
        )
    scenario = 2 if states[0].declared[actor] else 0
    return bool(
        numpy.array_equal(*summaries)
        and states[0].live_wall > 0
        and summaries[0][scenario : scenario + 2, 1].max() > 0
    )


def round_pairs(game_id, events):
    """Sparse preferred/other early cuts, normalized per first-tenpai event.

    Future hands are labels only. The recorded continuation is conditional
    evidence, not an estimate of counterfactual full-game value.
    """
    state = replay.Replay.from_start(game_id, events[0])
    history = [[] for _ in range(4)]
    seen_ready = [False] * 4
    pairs = {}
    for index, event in enumerate(events[1:], 1):
        if event["type"] in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
            history = [[] for _ in range(4)]
        if event["type"] == "reach_accepted":
            history[event["actor"]] = []
        if event["type"] == "dahai":
            actor = event["actor"]
            cut = shanten.mjai_tile_code(event["pai"])
            native = state.legal_observation(actor)
            counts = numpy.bincount(
                [features.native_tile_code(tile) for tile in native.hand],
                minlength=37,
            )
            after = counts - numpy.eye(37, dtype=int)[cut]
            distance = shanten.calculate_shanten(
                after[:34] + after[34:] @ numpy.eye(34, dtype=int)[[4, 13, 22]]
            )
            first_ready = distance == 0 and not seen_ready[actor]
            seen_ready[actor] |= distance == 0
            if not state.accepted[actor]:
                legal = sorted(
                    {
                        features.native_tile_code(action.tile)
                        for action in native.legal_actions()
                        if action.action_type == riichienv.ActionType.DISCARD
                    }
                )
                danger = ron_exposure(
                    state,
                    actor,
                    sorted(set(legal) | {row["cut"] for row in history[actor]}),
                )
                endpoint = {}
                if first_ready:
                    for earlier_ix, row in enumerate(history[actor]):
                        if (
                            cut not in row["legal"]
                            or row["danger"][cut]
                            or row["danger"][row["cut"]]
                            or 3 in (danger[cut], danger[row["cut"]])
                            or bool(danger[cut]) == bool(danger[row["cut"]])
                            or not preserves_completion(
                                history[actor][earlier_ix:], counts, cut
                            )
                        ):
                            continue
                        if verified_exchange(
                            game_id=game_id,
                            events=events,
                            actor=actor,
                            early=row["index"],
                            late=index,
                        ):
                            endpoint[row["index"]] = (
                                [cut, row["cut"]]
                                if danger[cut]
                                else [row["cut"], cut]
                            )
                    for early, pair in endpoint.items():
                        pairs[early] = (pair, 1 / len(endpoint))
                if not seen_ready[actor]:
                    history[actor].append(
                        {
                            "index": index,
                            "counts": counts,
                            "after": after,
                            "cut": cut,
                            "legal": legal,
                            "distance": distance,
                            "danger": danger,
                        }
                    )
        state.apply(event)
    return pairs
