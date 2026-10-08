"""Native legal action bridge, including paired calls and physical copies."""

import numpy
import riichienv

from model import features

KAN_ACTIONS = {riichienv.ActionType.ANKAN, riichienv.ActionType.KAKAN}
WIN_ACTIONS = {riichienv.ActionType.RON, riichienv.ActionType.TSUMO}


def phase_for_actions(legal_actions):
    kinds = {action.action_type for action in legal_actions}
    if kinds & WIN_ACTIONS:
        return "WIN"
    if kinds & KAN_ACTIONS:
        return "KAN"
    if riichienv.ActionType.RIICHI in kinds:
        return "RIICHI"
    if riichienv.ActionType.DISCARD in kinds:
        return "DISCARD"
    return "RESPONSE"


def forced_policy_action(state, actor, legal_actions):
    # Temporary policy: always accept native-legal agari and kyuushu offers.
    for action in legal_actions:
        if action.action_type in WIN_ACTIONS:
            return action
    for action in legal_actions:
        if action.action_type == riichienv.ActionType.KYUSHU_KYUHAI:
            return action
    if len(legal_actions) == 1:
        return legal_actions[0]
    selected = features.single_legal_action(
        legal_actions, state.legal_observation(actor).drawn_tile
    )
    if (
        selected is not None
        and selected.action_type == riichienv.ActionType.DISCARD
    ):
        return select_discard(
            legal_actions,
            features.native_tile_code(selected.tile),
            state.legal_observation(actor).drawn_tile,
        )
    return selected


def policy_request(state, actor, observation, east_only=False):
    legal = observation.legal_actions()
    phase = phase_for_actions(legal)
    packed = features.pack(
        state=state,
        actor=actor,
        phase=phase,
        legal_observation=observation,
        east_only=east_only,
    )
    return phase, {
        "features": packed,
        "actor": actor,
        "legal_actions": legal,
        "drawn_tile": observation.drawn_tile,
        "phase": phase,
    }


def continue_request(request):
    declined = WIN_ACTIONS if request["phase"] == "WIN" else KAN_ACTIONS
    legal = [
        action
        for action in request["legal_actions"]
        if action.action_type not in declined
    ]
    phase = phase_for_actions(legal)
    return {
        **request,
        "legal_actions": legal,
        "phase": phase,
        "features": {
            **request["features"],
            "decision_phase": numpy.array(features.PHASES[phase], numpy.int8),
            "win_kind": numpy.array(0, numpy.int8),
        },
    }


def select_discard(legal_actions, discard_action, drawn_tile):
    candidates = [
        action
        for action in legal_actions
        if action.action_type == riichienv.ActionType.DISCARD
        and features.native_tile_code(action.tile) == discard_action
    ]
    # Balance drawn/held origins when both are legal for this tile class.
    origins = [
        copies
        for drawn in (False, True)
        if (
            copies := [
                action
                for action in candidates
                if (action.tile == drawn_tile) == drawn
            ]
        )
    ]
    copies = origins[numpy.random.randint(len(origins))]
    return copies[numpy.random.randint(len(copies))]


def policy_candidates(request):
    if request["phase"] == "WIN":
        return (
            [("win_decision_logits", 0, None, {})]
            if any(
                action.action_type not in WIN_ACTIONS
                for action in request["legal_actions"]
            )
            else []
        ) + [
            (
                "win_decision_logits",
                1,
                next(
                    action
                    for action in request["legal_actions"]
                    if action.action_type in WIN_ACTIONS
                ),
                {},
            ),
        ]
    if request["phase"] == "DISCARD":
        return [
            (
                "discard_policy_logits",
                code,
                select_discard(
                    request["legal_actions"], code, request["drawn_tile"]
                ),
                {},
            )
            for code in numpy.flatnonzero(
                features.discard_legal(request["features"])
            )
        ]
    result = []
    if request["phase"] == "RIICHI":
        for code in numpy.flatnonzero(
            features.discard_legal(request["features"])
        ):
            action = select_discard(
                request["legal_actions"], code, request["drawn_tile"]
            )
            result.append(
                (
                    "riichi_policy_logits",
                    int(code),
                    action,
                    {"discard_action": int(code)},
                )
            )
            if request["features"]["riichi_discard_legal_mask"][code]:
                declaration = next(
                    value
                    for value in request["legal_actions"]
                    if value.action_type == riichienv.ActionType.RIICHI
                )
                result.append(
                    (
                        "riichi_policy_logits",
                        37 + int(code),
                        declaration,
                        {"discard_action": int(code)},
                    )
                )
    elif request["phase"] == "RESPONSE":
        for action in features.prefer_red_calls(request["legal_actions"]):
            if action.action_type == riichienv.ActionType.PASS:
                result.append(("response_policy_logits", 0, action, {}))
            elif action.action_type == riichienv.ActionType.DAIMINKAN:
                result.append(("response_policy_logits", 149, action, {}))
            elif action.action_type in (
                riichienv.ActionType.CHI,
                riichienv.ActionType.PON,
            ):
                chi = action.action_type == riichienv.ActionType.CHI
                option = (
                    features.chi_pattern(action.tile, action.consume_tiles)
                    if chi
                    else 0
                )
                for cut in numpy.flatnonzero(
                    request["features"][
                        "chi_discard_legal" if chi else "pon_discard_legal"
                    ][option]
                ):
                    result.append(
                        (
                            "response_policy_logits",
                            (1 if chi else 112) + option * 37 + int(cut),
                            action,
                            {
                                "discard_action": int(cut),
                                "continuation_discard": int(cut),
                            },
                        )
                    )
    else:
        result.append(("kan_action_logits", 0, None, {}))
        for action in request["legal_actions"]:
            if action.action_type in KAN_ACTIONS:
                result.append(
                    (
                        "kan_action_logits",
                        1 + action.tile // 4,
                        action,
                        {},
                    )
                )
    return list({(row[0], row[1]): row for row in result}.values())


def select_policy_action(request, outputs, batch_index):
    candidates = policy_candidates(request)
    if not candidates:
        return None, {"candidate_count": 0, "probability": 1.0}
    # A decision's candidates all use the head chosen by policy_candidates.
    scores = numpy.asarray(outputs[candidates[0][0]])[
        batch_index, [code for _, code, _, _ in candidates]
    ]
    weights = numpy.exp(scores - scores.max())
    probability = weights / weights.sum()
    # Select the immediate decision by its marginal mass before choosing
    # its continuation. Otherwise several good discards split a call's vote.
    groups = []
    if request["phase"] == "RIICHI":
        groups = [numpy.array([code // 37 for _, code, _, _ in candidates])]
    elif request["phase"] == "RESPONSE":
        groups = [
            numpy.array(
                [
                    (
                        0
                        if code == 0
                        else 1 if code < 112 else 2 if code < 149 else 3
                    )
                    for _, code, _, _ in candidates
                ]
            ),
            numpy.array(
                [
                    0 if code == 0 else (code - 1) // 37 + 1
                    for _, code, _, _ in candidates
                ]
            ),
        ]
    eligible = numpy.ones(len(candidates), dtype=bool)
    for group in groups:
        totals = numpy.bincount(group, weights=probability * eligible)
        branch = numpy.random.choice(numpy.flatnonzero(totals == totals.max()))
        eligible &= group == branch
    selected = int(
        numpy.random.choice(
            numpy.flatnonzero(
                eligible & (probability == probability[eligible].max())
            )
        )
    )
    return candidates[selected][2], {
        **candidates[selected][3],
        "candidate_count": len(candidates),
        "probability": float(probability[selected]),
    }
