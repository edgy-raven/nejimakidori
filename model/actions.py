"""Legal actions, physical-copy selection and model action indexing."""

import numpy
import riichienv

from log_dataset import features
from model import outcomes

TRAIN_POLICY_HEADS = (
    "discard_policy_logits",
    "riichi_policy_logits",
    "response_policy_logits",
    "kan_action_logits",
    "win_decision_logits",
)

# All actor outputs use float32; only their shapes vary.
ACTOR_OUTPUTS = {
    "win_decision_logits": [2],
    "opponent_shanten_logits": [3, 4],
    "opponent_tenpai_logits": [3],
    "opponent_ukeire_by_shanten_logits": [3, 4, 34],
    "opponent_ukeire_logits": [3, 34],
    "opponent_wait_logits": [3, 34],
    "opponent_hand_count_logits": [3, 37, 5],
    "round_completion_yaku_logits": [4, 2, 23],
    "round_completion_score_logits": [4, 2, len(outcomes.SCORE_STATES)],
    "discard_policy_logits": [37],
    "riichi_policy_logits": [74],
    "response_policy_logits": [150],
    "riichi_action_logits": [2],
    "tenpai_decision_logits": [3],
    "response_action_logits": [4],
    "chi_pattern_logits": [3],
    "kan_action_logits": [35],
}


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
    # Kyuushu remains automatic; optional wins belong to the policy.
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
        state=state, actor=actor, phase=phase, east_only=east_only
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
                request["features"]["discard_action_legal"]
            )
        ]
    result = []
    if request["phase"] == "RIICHI":
        for code in numpy.flatnonzero(
            request["features"]["discard_action_legal"]
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
                chii = action.action_type == riichienv.ActionType.CHI
                option = (
                    features.chii_pattern(action.tile, action.consume_tiles)
                    if chii
                    else 0
                )
                for cut in numpy.flatnonzero(
                    request["features"][
                        "chi_discard_legal" if chii else "pon_discard_legal"
                    ][option]
                ):
                    result.append(
                        (
                            "response_policy_logits",
                            (1 if chii else 112) + option * 37 + int(cut),
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


def opening_temperature(state, actor):
    return max(0.0, 1.5 - 0.5 * len(state.rivers[actor]))


def select_policy_action(request, outputs, batch_index, temperature=0.0):
    candidates = policy_candidates(request)
    if not candidates:
        return None, {"candidate_count": 0, "probability": 1.0}
    # A decision's candidates all use the head chosen by policy_candidates.
    scores = numpy.asarray(outputs[candidates[0][0]])[
        batch_index, [code for _, code, _, _ in candidates]
    ]
    weights = numpy.exp(scores - scores.max())
    probability = weights / weights.sum()
    if temperature > 0:
        weights = numpy.exp((scores - scores.max()) / temperature)
        selected = int(
            numpy.random.choice(len(candidates), p=weights / weights.sum())
        )
        return candidates[selected][2], {
            **candidates[selected][3],
            "candidate_count": len(candidates),
            "probability": float(probability[selected]),
        }
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


def legal_choices(inputs):
    import tensorflow as tf

    discard = tf.cast(inputs["discard_action_legal"], tf.bool)
    declaration = tf.cast(inputs["riichi_discard_legal_mask"], tf.bool)
    return {
        "win_decision_logits": tf.stack(
            [
                tf.ones_like(inputs["win_kind"], dtype=tf.bool),
                inputs["win_kind"] > 0,
            ],
            axis=-1,
        ),
        "discard_policy_logits": discard,
        "riichi_policy_logits": tf.concat([discard, discard & declaration], 1),
        "response_policy_logits": tf.concat(
            [
                tf.cast(inputs["response_legal_mask"][:, :1], tf.bool),
                tf.reshape(
                    tf.cast(inputs["chi_discard_legal"], tf.bool), (-1, 111)
                ),
                tf.reshape(
                    tf.cast(inputs["pon_discard_legal"], tf.bool), (-1, 37)
                ),
                tf.cast(inputs["response_legal_mask"][:, 3:], tf.bool),
            ],
            axis=1,
        ),
        "kan_action_logits": tf.cast(inputs["kan_legal_mask"], tf.bool),
    }


def guaranteed_safe_tiles(inputs):
    """Publicly proven ron safety, by opponent and base tile."""
    import tensorflow as tf

    exhausted = inputs["known_unavailable_counts"] == 4
    # The offered honor itself cannot rule out a kokushi wait for that honor.
    kokushi_blocked = (
        tf.reduce_sum(
            tf.cast(
                tf.gather(
                    exhausted,
                    [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33],
                    axis=1,
                ),
                tf.int32,
            ),
            axis=1,
        )
        >= 2
    )
    return (inputs["genbutsu_to_seat"][:, 1:] > 0) | (
        exhausted[:, None, :]
        & (tf.range(34)[None, None, :] >= 27)
        & (
            kokushi_blocked[:, None, None]
            | tf.reduce_any(
                inputs["meld_valid"][:, 1:] > 0, axis=-1, keepdims=True
            )
        )
    )


def candidate_hand_indices(inputs):
    """Map the shared 295 action slots to their post-action hand banks."""
    import tensorflow as tf

    discard = inputs["discard_candidate_hand"]
    batch = tf.shape(discard)[0]
    return tf.concat(
        [
            discard,
            discard,
            discard,
            tf.zeros((batch, 1), discard.dtype),
            tf.reshape(inputs["chi_candidate_hand"], (-1, 111)),
            tf.reshape(inputs["pon_candidate_hand"], (-1, 37)),
            tf.zeros((batch, 1), discard.dtype),
            inputs["kan_candidate_hand"],
        ],
        axis=1,
    )


def discard_decisions(inputs, selected):
    """Actions ending in a discard, including chii/pon then discard."""
    import tensorflow as tf

    phase = tf.gather(inputs["decision_phase"], selected[:, 0])
    return (
        ((phase == 0) & (selected[:, 1] < 37))
        | ((phase == 3) & (selected[:, 1] >= 37) & (selected[:, 1] < 111))
        | ((phase == 1) & (selected[:, 1] >= 112) & (selected[:, 1] < 260))
    )


def immediate_decisions(inputs, selected):
    """Discard ron and robbery of an actor's concealed/added kan."""
    import tensorflow as tf

    return discard_decisions(inputs, selected) | (
        (tf.gather(inputs["decision_phase"], selected[:, 0]) == 2)
        & (selected[:, 1] >= 261)
    )
