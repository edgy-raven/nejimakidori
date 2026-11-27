"""Later danger nominates earlier legal discards for fresh rollouts."""

import collections
import types

import riichienv

from log_dataset import (
    _native_collection,
    features,
    focus_sampling,
    replay,
    shanten,
)

DEFINITION = {
    "endpoint": (
        "first historical visible-threat genbutsu/non-genbutsu dilemma "
        "at minimum legal shanten"
    ),
    "earlier": "both safe, equal immediate shanten to recorded cut",
    "endpoints_are_roots": False,
    "actions": "all qualifying alternatives plus recorded cut",
}


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
                _native_collection.score_hand(
                    types.SimpleNamespace(
                        tiles=hand + [physical],
                        melds=state.melds[opponent],
                        agari_tile=physical,
                        dora_indicators=[],
                    ),
                    conditions=riichienv.Conditions(
                        riichi=state.accepted[opponent],
                        houtei=state.live_wall == 0,
                        player_wind=(
                            riichienv.Wind.East,
                            riichienv.Wind.South,
                            riichienv.Wind.West,
                            riichienv.Wind.North,
                        )[(opponent - state.start["oya"]) % 4],
                        round_wind="ESWN".index(state.start["bakaze"]),
                    ),
                    ura=[],
                ).is_win
            )
    return result


def round_pairs(game_id, events):
    """Nominate safe early cuts before the first future defensive dilemma.

    At the endpoint, genbutsu and non-genbutsu alternatives against the same
    publicly threatening opponent must both achieve best immediate shanten.
    Only earlier safe, equal-shanten alternatives become roots; calls and
    changed later hands are allowed. All qualifying alternative tiles precede
    the recorded cut in each action list. Ordering is not an actor target.
    """
    state = replay.Replay.from_start(game_id, events[0])
    history = [[] for _ in range(4)]
    seen_endpoint = [False] * 4
    candidates = {}
    for index, event in enumerate(events[1:], 1):
        if event["type"] == "reach_accepted":
            history[event["actor"]] = []
        if (
            event["type"] == "dahai"
            and not seen_endpoint[event["actor"]]
            and not state.accepted[event["actor"]]
        ):
            actor = event["actor"]
            cut = shanten.mjai_tile_code(event["pai"])
            legal = {
                features.native_tile_code(action.tile)
                for action in state.legal_observation(actor).legal_actions()
                if action.action_type == riichienv.ActionType.DISCARD
            }
            packed = features.pack(state, actor, "DISCARD")
            threatening = [
                seat
                for seat in range(1, 4)
                if focus_sampling.visible_threat(packed, seat)
            ]
            non_genbutsu = set()
            if threatening:
                best = min(
                    packed["discard_action_all_shanten"][tile] for tile in legal
                )
                efficient = {
                    tile
                    for tile in legal
                    if packed["discard_action_all_shanten"][tile] == best
                }
                for opponent in threatening:
                    safe = {
                        tile
                        for tile in efficient
                        if packed["genbutsu_to_seat"][
                            opponent, shanten.base_id(tile)
                        ]
                    }
                    if safe:
                        non_genbutsu.update(efficient - safe)
            if non_genbutsu:
                seen_endpoint[actor] = True
                for row in history[actor]:
                    alternatives = sorted(
                        (non_genbutsu & row["legal"]) - {row["cut"]}
                    )
                    if alternatives:
                        candidates[row["index"]] = (
                            index,
                            [*alternatives, row["cut"]],
                        )
            else:
                history[actor].append(
                    {
                        "index": index,
                        "cut": cut,
                        "legal": {
                            tile
                            for tile in legal
                            if packed["discard_action_all_shanten"][tile]
                            == packed["discard_action_all_shanten"][cut]
                        },
                    }
                )
        state.apply(event)

    # Future threat selects roots only. Equal-shanten earlier actions must
    # also both be safe at the original decision.
    state = replay.Replay.from_start(game_id, events[0])
    retained = {}
    for index, (endpoint, actions) in sorted(candidates.items()):
        while state.event_index < index - 1:
            state.apply(events[state.event_index + 1])
        actor = events[index]["actor"]
        exposure = ron_exposure(state, actor, actions)
        if exposure[actions[-1]]:
            continue
        alternatives = [tile for tile in actions[:-1] if not exposure[tile]]
        if alternatives:
            retained[index] = (endpoint, [*alternatives, actions[-1]])
    counts = collections.Counter(endpoint for endpoint, _ in retained.values())
    return {
        index: (actions, 1 / counts[endpoint])
        for index, (endpoint, actions) in retained.items()
    }
