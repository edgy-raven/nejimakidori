"""Translate browser rounds to model decisions without shared game state."""

import numpy
import riichienv

from log_dataset import shanten
from model import actions, features, outcomes, replay


class NoDecision(ValueError):
    def __init__(self, context):
        super().__init__("The cursor does not identify a decision.")
        self.context = context


def tile_name(tile):
    if tile == "?":
        return tile
    if not isinstance(tile, str) or len(tile) != 2 or tile[1] not in "mpsz":
        raise ValueError("Invalid tile: " + str(tile))
    if tile[1] == "z":
        return ("E", "S", "W", "N", "P", "F", "C")[int(tile[0]) - 1]
    return "5" + tile[1] + "r" if tile[0] == "0" else tile


def round_events(round_data):
    if len(round_data["hands"]) != 4 or len(round_data["scores"]) != 4:
        raise ValueError("A round requires four players.")
    if not 1 <= round_data["kyoku"] <= 4:
        raise ValueError("Invalid round number.")
    events = [
        {
            "type": "start_kyoku",
            "bakaze": ("E", "S", "W", "N")[round_data["prevailingWind"]],
            "kyoku": round_data["kyoku"],
            "honba": round_data["honba"],
            "kyotaku": round_data["riichiSticks"],
            "oya": round_data["kyoku"] - 1,
            "scores": round_data["scores"],
            "dora_marker": tile_name(round_data["doraIndicators"][0]),
            "tehais": [
                [tile_name(tile) for tile in hand]
                for hand in round_data["hands"]
            ],
        }
    ]
    cursors = []
    doras = list(round_data["doraIndicators"])
    accepted = set()
    last_discard = None
    for event in round_data["events"]:
        if event.get("liqi") and event["liqi"]["seat"] not in accepted:
            actor = event["liqi"]["seat"]
            events.append({"type": "reach_accepted", "actor": actor})
            accepted.add(actor)
        cursors.append(len(events))
        kind = event["type"]
        if kind == "draw":
            events.append(
                {
                    "type": "tsumo",
                    "actor": event["seat"],
                    "pai": tile_name(event["tile"]),
                }
            )
        elif kind == "discard":
            if event["riichi"]:
                events.append({"type": "reach", "actor": event["seat"]})
            last_discard = event["seat"]
            events.append(
                {
                    "type": "dahai",
                    "actor": event["seat"],
                    "pai": tile_name(event["tile"]),
                    "tsumogiri": event["tsumogiri"],
                }
            )
        elif kind == "call":
            caller = event["seat"]
            called = next(
                index
                for index, seat in enumerate(event["froms"])
                if seat != caller
            )
            events.append(
                {
                    "type": (
                        "chi"
                        if event["callType"] == "chii"
                        else event["callType"]
                    ),
                    "actor": caller,
                    "target": event["froms"][called],
                    "pai": tile_name(event["tiles"][called]),
                    "consumed": [
                        tile_name(tile)
                        for index, tile in enumerate(event["tiles"])
                        if index != called
                    ],
                }
            )
        elif kind == "kan":
            converted = {
                "type": event["callType"],
                "actor": event["seat"],
                "consumed": [tile_name(tile) for tile in event["tiles"]],
            }
            if event["callType"] == "kakan":
                converted["pai"] = converted["consumed"].pop()
            events.append(converted)
        elif kind == "agari":
            for index, winner in enumerate(event["hules"]):
                events.append(
                    {
                        "type": "hora",
                        "actor": winner["seat"],
                        "target": (
                            winner["seat"] if winner["zimo"] else last_discard
                        ),
                        "deltas": event["deltas"] if index == 0 else [0] * 4,
                    }
                )
        elif kind == "ryuukyoku":
            events.append({"type": "ryukyoku", "deltas": event["deltas"]})
        else:
            raise ValueError("Unsupported round event: " + kind)
        for tile in event.get("doras", [])[len(doras) :]:
            events.append({"type": "dora", "dora_marker": tile_name(tile)})
            doras.append(tile)
    return events, cursors


def reconstruct(round_data, cursor):
    try:
        events, cursors = round_events(round_data)
    except (KeyError, TypeError, IndexError, StopIteration) as error:
        raise ValueError("Invalid round structure: " + str(error)) from error
    if type(cursor) is not int or not 0 <= cursor <= len(cursors):
        raise ValueError("Event cursor is outside this hand.")
    state = replay.Replay.from_start("browser", events[0])
    stop = cursors[cursor] if cursor < len(cursors) else len(events)
    for event in events[1:stop]:
        state.apply(event)
    return state, events


def discard_options(packed):
    result = []
    for action in numpy.flatnonzero(features.discard_legal(packed)):
        code = features.discard_tile_code(action, packed)
        result.append(
            {
                "action": int(action),
                "tile": shanten.TILE_NAMES[code],
                "tsumogiri": bool(action == 37),
                "all_shanten": int(packed["discard_action_all_shanten"][code])
                - 1,
                "all_ukeire_count": int(
                    packed["discard_action_all_ukeire_count"][code]
                ),
                "all_upgrade_count": int(
                    packed["discard_action_all_upgrade_tile_count"][code]
                ),
            }
        )
    return result


def predict_replay(predictor, round_data, event_count):
    state, events = reconstruct(round_data, event_count)
    context = {
        "actual_shanten": [
            int(shanten.calculate_shanten(state.concealed_counts(seat)))
            for seat in range(4)
        ]
    }
    if event_count == len(round_data["events"]) or round_data["events"][
        event_count
    ]["type"] not in {"discard", "call", "kan"}:
        raise NoDecision(context)
    event = round_data["events"][event_count]
    actor = event["seat"]
    observation = state.legal_observation(actor)
    phase = {"discard": "DISCARD", "call": "RESPONSE", "kan": "KAN"}[
        event["type"]
    ]
    if phase == "DISCARD" and any(
        action.action_type == riichienv.ActionType.RIICHI
        for action in observation.legal_actions()
    ):
        phase = "RIICHI"
    packed = features.pack(
        state=state, actor=actor, phase=phase, legal_observation=observation
    )
    expert = {}
    if event["type"] == "discard":
        code = (
            37
            if event["tsumogiri"]
            else shanten.mjai_tile_code(tile_name(event["tile"]))
        )
        expert.update(discard_policy=code, riichi_action=int(event["riichi"]))
        expert["riichi_policy"] = code + 38 * int(event["riichi"])
    elif event["type"] == "call":
        kind = event["callType"]
        expert["response_action"] = {
            "chii": 1,
            "chi": 1,
            "pon": 2,
            "daiminkan": 3,
        }[kind]
        if kind == "daiminkan":
            expert["response_policy"] = 149
        else:
            called = next(
                index
                for index, seat in enumerate(event["froms"])
                if seat != actor
            )
            consumed = [
                tile_name(tile)
                for index, tile in enumerate(event["tiles"])
                if index != called
            ]
            continuation = next(
                followup
                for followup in round_data["events"][event_count + 1 :]
                if followup["type"] == "discard" and followup["seat"] == actor
            )
            cut = shanten.mjai_tile_code(tile_name(continuation["tile"]))
            expert["continuation_discard"] = cut
            if kind in {"chi", "chii"}:
                pattern = features.chi_pattern(
                    tile_name(event["tiles"][called]), consumed
                )
                expert["chi_pattern"] = pattern
                expert["response_policy"] = 1 + pattern * 37 + cut
            else:
                expert["response_policy"] = 112 + cut
            if not features.preferred_recorded_call(
                "chi" if kind in {"chi", "chii"} else "pon",
                consumed,
                observation.legal_actions(),
            ):
                # Coarse call type is observed; its nonpreferred physical
                # continuation has no probability in the current policy.
                del expert["response_policy"]
    else:
        expert["kan_action"] = 1 + shanten.base_id(
            shanten.mjai_tile_code(tile_name(event["tiles"][0]))
        )
    inference = predictor.infer([packed])
    settlement = outcomes.settlement(
        outcomes.relative_payments(
            outcomes.terminal_payments(events), actor, state.accepted
        )
    )
    position = {
        **context,
        "actor": actor,
        "phase": phase,
        "actual_settlement": settlement.astype(float).tolist(),
        "current_riichi_sticks": int(state.native_env.riichi_sticks),
        "expert_decisions": expert,
        "discard_options": (
            discard_options(packed) if phase in {"DISCARD", "RIICHI"} else []
        ),
    }
    return {"position": position, "inference": inference.as_dict()}


def predict_live(predictor, round_data, actor):
    if type(actor) is not int or not 0 <= actor < 4:
        raise ValueError("Actor must be a seat from zero to three.")
    state, _ = reconstruct(round_data, len(round_data["events"]))
    observation = state.legal_observation(actor)
    legal = observation.legal_actions()
    forced = actions.forced_policy_action(state, actor, legal)
    phase = actions.phase_for_actions(legal)
    inference_result = None
    step = {"probability": 1.0, "candidate_count": 1}
    selected = forced
    if selected is None:
        phase, request = actions.policy_request(
            state, actor, observation, round_data.get("eastOnly", False)
        )
        latency = 0.0
        while selected is None:
            result = predictor.infer([request["features"]])
            latency += result.latency_ms
            selected, step = actions.select_policy_action(
                request, result.outputs, 0
            )
            phase = request["phase"]
            inference_result = result.as_dict()
            inference_result["latency_ms"] = latency
            if selected is None:
                if phase not in {"WIN", "KAN"}:
                    raise ValueError("No legal action at this position.")
                request = actions.continue_request(request)
                forced = actions.forced_policy_action(
                    state, actor, request["legal_actions"]
                )
                if forced is not None:
                    selected = forced
                    phase = request["phase"]
                    step = {"probability": 1.0, "candidate_count": 1}
    decision = {
        "actor": actor,
        "phase": phase,
        "possible_actions": [
            features.action_event(action, observation.drawn_tile)
            for action in legal
        ],
        "discard_options": (
            discard_options(request["features"])
            if forced is None and phase in {"DISCARD", "RIICHI"}
            else []
        ),
        "action": features.action_event(selected, observation.drawn_tile),
        "action_kind": "forced" if forced is not None else "recommended",
        "riichi_discard": None,
        "call_discard": None,
        "difficulty": (
            0.0
            if forced is not None
            else max(0.0, min(1.0, 1.0 - float(step["probability"])))
        ),
    }
    if "discard_action" in step and step["discard_action"] is not None:
        code = int(step["discard_action"])
        paired = {
            "actor": actor,
            "type": "dahai",
            "pai": shanten.TILE_NAMES[
                (
                    features.native_tile_code(observation.drawn_tile)
                    if code == 37
                    else code
                )
            ],
            "tsumogiri": code == 37,
        }
        if selected.action_type == riichienv.ActionType.RIICHI:
            decision["riichi_discard"] = paired
        elif selected.action_type in (
            riichienv.ActionType.CHI,
            riichienv.ActionType.PON,
        ):
            decision["call_discard"] = paired
    return {"decision": decision, "inference": inference_result}
