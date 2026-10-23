"""Translate browser rounds to model decisions without shared game state."""

import numpy
import riichienv

from log_dataset import features, replay, review, shanten
from model import actions, outcomes


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
    for action in numpy.flatnonzero(packed["discard_action_legal"]):
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


def replay_phase(round_data, event_count, actor, state):
    if event_count == len(round_data["events"]):
        return None
    event = round_data["events"][event_count]
    if event["type"] in {"discard", "call", "kan"}:
        if event["seat"] == actor:
            return {"discard": "DISCARD", "call": "RESPONSE", "kan": "KAN"}[
                event["type"]
            ]
    if event_count:
        previous = round_data["events"][event_count - 1]
        if previous["type"] == "discard" and previous["seat"] != actor:
            if any(
                action.action_type
                in {
                    riichienv.ActionType.CHI,
                    riichienv.ActionType.PON,
                    riichienv.ActionType.DAIMINKAN,
                }
                for action in state.legal_observation(actor).legal_actions()
            ):
                return "RESPONSE"
    return None


def replay_decisions(round_data, seat):
    """Index recorded choices and every native-legal call opportunity."""
    if type(seat) is not int or not 0 <= seat < 4:
        raise ValueError("Actor must be a seat from zero to three.")
    state, events = review.reconstruct(round_data, 0)
    _, cursors = review.round_events(round_data)
    applied = cursors[0] if cursors else len(events)
    result = []
    for cursor, stop in enumerate(cursors):
        for event in events[applied:stop]:
            state.apply(event)
        applied = stop
        if replay_phase(round_data, cursor, seat, state) is not None:
            result.append(cursor)
    return {"event_counts": result}


def replay_position(round_data, event_count, seat):
    state, events = review.reconstruct(round_data, event_count)
    context = {
        "actual_shanten": [
            int(shanten.calculate_shanten(state.concealed_counts(seat)))
            for seat in range(4)
        ]
    }
    if type(seat) is not int or not 0 <= seat < 4:
        raise ValueError("Actor must be a seat from zero to three.")
    if event_count == len(round_data["events"]):
        raise NoDecision(context)
    event = round_data["events"][event_count]
    actor = seat
    phase = replay_phase(round_data, event_count, actor, state)
    if phase is None:
        raise NoDecision(context)
    observation = state.legal_observation(actor)
    if phase == "DISCARD" and any(
        action.action_type == riichienv.ActionType.RIICHI
        for action in observation.legal_actions()
    ):
        phase = "RIICHI"
    packed = features.pack(state=state, actor=actor, phase=phase)
    expert = {}
    if phase == "RESPONSE" and (
        event["type"] != "call" or event["seat"] != actor
    ):
        if event["type"] == "draw":
            expert.update(response_action=0, response_policy=0)
    elif event["type"] == "discard":
        code = shanten.mjai_tile_code(review.tile_name(event["tile"]))
        expert.update(discard_policy=code, riichi_action=int(event["riichi"]))
        expert["riichi_policy"] = code + 37 * int(event["riichi"])
    elif event["type"] == "call" and event["seat"] == actor:
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
                pattern = features.chii_pattern(
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
    elif phase == "KAN":
        expert["kan_action"] = 1 + shanten.base_id(
            shanten.mjai_tile_code(review.tile_name(event["tiles"][0]))
        )
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
        "response_censored": phase == "RESPONSE" and not expert,
        "legal_response_actions": (
            numpy.flatnonzero(packed["response_legal_mask"]).tolist()
            if phase == "RESPONSE"
            else []
        ),
        "discard_options": (
            discard_options(packed) if phase in {"DISCARD", "RIICHI"} else []
        ),
    }
    return packed, position


def predict_replay(predictor, round_data, event_counts, seat):
    if not isinstance(event_counts, list) or not 1 <= len(event_counts) <= 8:
        raise ValueError("event_counts must contain 1 to 8 cursors.")
    packed, positions = zip(
        *(replay_position(round_data, cursor, seat) for cursor in event_counts)
    )
    inference = predictor.infer(list(packed))
    return {
        "annotations": [
            {
                **position,
                "predictions": [review_prediction(prediction)],
                "model_latency_ms": inference.latency_ms / len(positions),
            }
            for position, prediction in zip(positions, inference.predictions)
        ]
    }


def review_prediction(prediction):
    """Keep grading inputs and forecasts, without auxiliary training outputs."""
    return {
        **{
            head + "_probabilities": prediction[head + "_probabilities"]
            for head in (
                "discard_policy",
                "riichi_policy",
                "riichi_action",
                "response_policy",
                "response_action",
                "chi_pattern",
                "kan_action",
                "opponent_shanten",
                "final_placement",
                "terminal_payment",
                "review_outcome",
            )
        },
        "action_values": [
            {
                "action": value["action"],
                "expected_points": value["expected_points"],
                "projected_st3": value["projected_st3"],
                "immediate_ron_probabilities": value[
                    "immediate_ron_probabilities"
                ],
            }
            for value in prediction["action_values"]
        ],
        "structured_outcome": {
            name: prediction["structured_outcome"][name]
            for name in (
                "types",
                "values",
                "settlement_mean",
                "settlement_stddev",
            )
        },
    }


def predict_live(predictor, round_data, actor):
    if type(actor) is not int or not 0 <= actor < 4:
        raise ValueError("Actor must be a seat from zero to three.")
    state, _ = review.reconstruct(round_data, len(round_data["events"]))
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
                request,
                result.outputs,
                0,
                temperature=actions.opening_temperature(state, actor),
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
        "temperature": actions.opening_temperature(state, actor),
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
