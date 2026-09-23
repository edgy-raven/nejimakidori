"""Native legal action bridge, including paired calls and physical copies."""

import numpy
import riichienv

from . import features

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
    return features.single_legal_action(
        legal_actions, state.legal_observation(actor).drawn_tile
    )


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
    for action in legal_actions:
        if action.action_type != riichienv.ActionType.DISCARD:
            continue
        if discard_action == 37 and action.tile == drawn_tile:
            return action
        if (
            discard_action != 37
            and action.tile != drawn_tile
            and features.native_tile_code(action.tile) == discard_action
        ):
            return action
    raise ValueError("requested physical discard is not legal")


def policy_candidates(request):
    phase = request["phase"]
    legal = request["legal_actions"]
    if phase == "WIN":
        return (
            [("win_decision_logits", 0, None, {})]
            if any(action.action_type not in WIN_ACTIONS for action in legal)
            else []
        ) + [
            (
                "win_decision_logits",
                1,
                next(
                    action
                    for action in legal
                    if action.action_type in WIN_ACTIONS
                ),
                {},
            ),
        ]
    if phase == "DISCARD":
        return [
            (
                "discard_policy_logits",
                code,
                select_discard(legal, code, request["drawn_tile"]),
                {},
            )
            for code in numpy.flatnonzero(
                features.discard_legal(request["features"])
            )
        ]
    result = []
    if phase == "RIICHI":
        for code in numpy.flatnonzero(
            features.discard_legal(request["features"])
        ):
            action = select_discard(legal, code, request["drawn_tile"])
            result.append(
                (
                    "riichi_policy_logits",
                    int(code),
                    action,
                    {"discard_action": int(code)},
                )
            )
            tile_code = features.discard_tile_code(code, request["features"])
            if request["features"]["riichi_discard_legal_mask"][tile_code]:
                declaration = next(
                    value
                    for value in legal
                    if value.action_type == riichienv.ActionType.RIICHI
                )
                result.append(
                    (
                        "riichi_policy_logits",
                        38 + int(code),
                        declaration,
                        {"discard_action": int(code)},
                    )
                )
    elif phase == "RESPONSE":
        for action in features.prefer_red_calls(legal):
            kind = action.action_type
            if kind == riichienv.ActionType.PASS:
                result.append(("response_policy_logits", 0, action, {}))
            elif kind == riichienv.ActionType.DAIMINKAN:
                result.append(("response_policy_logits", 149, action, {}))
            elif kind in (riichienv.ActionType.CHI, riichienv.ActionType.PON):
                chi = kind == riichienv.ActionType.CHI
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
        for action in legal:
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
    normalized = {}
    for name in {candidate[0] for candidate in candidates}:
        row = numpy.asarray(outputs[name])[batch_index]
        shifted = row - numpy.max(row)
        normalized[name] = shifted - numpy.log(numpy.exp(shifted).sum())
    scores = numpy.array(
        [normalized[name][code] for name, code, action, meta in candidates]
    )
    probability = numpy.exp(scores - scores.max())
    probability /= probability.sum()
    groups = {}
    for index, (name, code, _, _) in enumerate(candidates):
        if name in {"discard_policy_logits", "riichi_policy_logits"}:
            key = (
                name,
                code // 38,
                features.discard_tile_code(code % 38, request["features"]),
            )
        else:
            key = (name, code)
        groups.setdefault(key, []).append(index)
    chosen = max(groups.values(), key=lambda group: probability[group].sum())
    selected = max(chosen, key=lambda index: probability[index])
    return candidates[selected][2], {
        **candidates[selected][3],
        "candidate_count": len(candidates),
        "probability": float(probability[chosen].sum()),
        "copy_probability": float(
            probability[selected] / probability[chosen].sum()
        ),
    }
