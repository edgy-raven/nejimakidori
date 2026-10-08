"""Translate browser rounds to model decisions without shared game state."""

import numpy
from log_dataset import replay, review, rollout_collection, shanten
from model import actions, features, outcomes

import riichienv


class NoDecision(ValueError):
    def __init__(self, context):
        super().__init__("The cursor does not identify a decision.")
        self.context = context


def replay_deal_ins(rounds, seat):
    """Legal ron tiles before recorded discards, without model inference."""
    if type(seat) is not int or not 0 <= seat < 4:
        raise ValueError("Actor must be a seat from zero to three.")
    result = []
    for round_data in rounds:
        events, cursors = review.round_events(round_data)
        state = replay.Replay.from_start("browser", events[0])
        applied = 1
        tiles = [[] for _ in round_data["events"]]
        for cursor, event in enumerate(round_data["events"]):
            for converted in events[applied : cursors[cursor]]:
                state.apply(converted)
            applied = cursors[cursor]
            if event["type"] != "discard" or event["seat"] != seat:
                continue
            opponents = [
                opponent
                for opponent in range(4)
                if opponent != seat and not state.missed_agari(opponent)
            ]
            cuts = {
                features.native_tile_code(action.tile): action
                for action in state.native_env.get_observation(
                    seat
                ).legal_actions()
                if action.action_type == riichienv.ActionType.DISCARD
            }
            for code, action in cuts.items():
                env = state.native_env.clone()
                env.apply_event(
                    {
                        **features.action_event(action),
                        "actor": seat,
                        "tsumogiri": shanten.TILE_NAMES[code]
                        == state.draws[seat],
                    }
                )
                if any(
                    response.action_type == riichienv.ActionType.RON
                    for opponent in opponents
                    for response in env.get_observation(
                        opponent
                    ).legal_actions()
                ):
                    tiles[cursor].append(shanten.TILE_NAMES[code])
        result.append(tiles)
    return {"deal_in_tiles": result}


def discard_options(packed):
    result = []
    for action in numpy.flatnonzero(features.discard_legal(packed)):
        code = int(action)
        result.append(
            {
                "action": int(action),
                "tile": shanten.TILE_NAMES[code],
                "tsumogiri": bool(
                    packed["current_draw_valid"]
                    and code == features.drawn_tile_code(packed)
                ),
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
    state, events = review.reconstruct(round_data, event_count)
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
        code = shanten.mjai_tile_code(review.tile_name(event["tile"]))
        expert.update(discard_policy=code, riichi_action=int(event["riichi"]))
        expert["riichi_policy"] = code + 37 * int(event["riichi"])
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
                review.tile_name(tile)
                for index, tile in enumerate(event["tiles"])
                if index != called
            ]
            continuation = next(
                followup
                for followup in round_data["events"][event_count + 1 :]
                if followup["type"] == "discard" and followup["seat"] == actor
            )
            cut = shanten.mjai_tile_code(review.tile_name(continuation["tile"]))
            expert["continuation_discard"] = cut
            if kind in {"chi", "chii"}:
                pattern = features.chi_pattern(
                    review.tile_name(event["tiles"][called]), consumed
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
            shanten.mjai_tile_code(review.tile_name(event["tiles"][0]))
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
        "rollout": rollout_collection.nominations(
            packed, inference.predictions[0], expert
        ),
        "discard_options": (
            discard_options(packed) if phase in {"DISCARD", "RIICHI"} else []
        ),
    }
    return {"position": position, "inference": inference.as_dict()}


def predict_live(predictor, round_data, actor):
    if type(actor) is not int or not 0 <= actor < 4:
        raise ValueError("Actor must be a seat from zero to three.")
    state, _ = review.reconstruct(round_data, len(round_data["events"]))
    observation = state.legal_observation(actor)
    legal = observation.legal_actions()
    forced = actions.forced_policy_action(state, actor, legal)
    phase = actions.phase_for_actions(legal)
    inference_result = None
    rollout_marks = []
    step = {"probability": 1.0, "candidate_count": 1}
    selected = forced
    if selected is None:
        phase, request = actions.policy_request(
            state, actor, observation, round_data.get("eastOnly", False)
        )
        latency = 0.0
        while selected is None:
            result = predictor.infer([request["features"]])
            rollout_marks.append(
                {
                    "phase": request["phase"],
                    **rollout_collection.nominations(
                        request["features"], result.predictions[0], {}
                    ),
                }
            )
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
        "rollout_marks": rollout_marks,
        "difficulty": (
            0.0
            if forced is not None
            else max(0.0, min(1.0, 1.0 - float(step["probability"])))
        ),
    }
    if "discard_action" in step:
        code = int(step["discard_action"])
        paired = {
            "actor": actor,
            "type": "dahai",
            "pai": shanten.TILE_NAMES[code],
            "tsumogiri": bool(
                selected.action_type == riichienv.ActionType.RIICHI
                and request["features"]["current_draw_valid"]
                and code == features.drawn_tile_code(request["features"])
            ),
        }
        if selected.action_type == riichienv.ActionType.RIICHI:
            decision["riichi_discard"] = paired
        elif selected.action_type in (
            riichienv.ActionType.CHI,
            riichienv.ActionType.PON,
        ):
            decision["call_discard"] = paired
    return {"decision": decision, "inference": inference_result}
