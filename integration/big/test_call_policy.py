"""Accepted call states, sparse transport and joint policy factorization."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import _native_collection, call_states, features, replay
from model import actions, actor, beliefs, critic
from model import network as joint_network
from model import payment_scoring


def test_sparse_states_match_executed_chi_pon_and_preserve_transport():
    kinds = set()
    examples = []
    for filename in ("payment_dealer_pao", "payment_pao", "riichi_push"):
        path = (
            pathlib.Path(__file__).parents[1]
            / "fixtures/model"
            / (filename + ".json")
        )
        events = json.loads(path.read_text())
        state = None
        for event in events:
            if event["type"] == "start_kyoku":
                state = replay.Replay.from_start(path.stem, event)
                known = [
                    seat
                    for seat, hand in enumerate(event["tehais"])
                    if "?" not in hand
                ]
            elif state is None:
                continue
            else:
                state.apply(event)
            for seat in known:
                observation = state.legal_observation(seat)
                legal = observation.legal_actions()
                if not legal or actions.phase_for_actions(legal) != "RESPONSE":
                    continue
                packed = features.pack(state, seat, "RESPONSE")
                for action in features.prefer_red_calls(legal):
                    call = features.action_event(action)
                    if call["type"] not in ("chi", "pon"):
                        continue
                    option = (
                        features.chii_pattern(action.tile, action.consume_tiles)
                        if call["type"] == "chi"
                        else 3
                    )
                    index = numpy.flatnonzero(
                        packed["post_call_option"] == option
                    )
                    assert len(index) == 1
                    call["actor"] = seat
                    call["target"] = state.pending_discard["actor"]
                    executed = _native_collection.FeatureReplay()
                    executed.apply(json.dumps([*state.history, call]))
                    expected = executed.pack(seat, 0)
                    for name, spec in _native_collection.INPUTS.items():
                        numpy.testing.assert_array_equal(
                            packed["post_call_" + name][index[0]],
                            numpy.frombuffer(
                                expected[name], spec["dtype"]
                            ).reshape(spec["shape"]),
                        )
                    numpy.testing.assert_array_equal(
                        packed["post_call_discard_action_legal"][index[0]],
                        packed["call_discard_legal"][option],
                    )
                    kinds.add(call["type"])
                if len(packed["post_call_owner"]):
                    examples.append(packed)
                if kinds == {"chi", "pon"} and len(examples) >= 3:
                    break
            if kinds == {"chi", "pon"} and len(examples) >= 3:
                break
        if kinds == {"chi", "pon"} and len(examples) >= 3:
            break
    assert kinds == {"chi", "pon"}
    combined = call_states.batch(examples)
    decoded = call_states.decode(
        tf.constant([call_states.serialize(row) for row in examples])
    )
    for name, value in decoded.items():
        numpy.testing.assert_array_equal(value, combined[name])


def test_call_mass_normalization_shared_discard_gradients_and_sixway_greedy(
    monkeypatch,
):
    start = {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "9s",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000] * 4,
        "tehais": [["?"] * 13 for _ in range(4)],
    }
    start["tehais"][1] = "1m 2m 3m 4m 4m 5m 6m 7m 7p 8p 9p E E".split()
    state = replay.Replay.from_start("shared-call-critic", start)
    state.apply({"type": "tsumo", "actor": 0, "pai": "4m"})
    state.apply({"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": True})
    rows = tf.nest.map_structure(
        tf.constant, call_states.batch([features.pack(state, 1, "RESPONSE")])
    )
    numpy.testing.assert_array_equal(
        numpy.sort(rows["post_call_option"]), [0, 1, 2, 3]
    )
    base, calls = call_states.base(rows), call_states.post(rows)
    # Calling 4m forbids 4m itself; 456m also forbids 7m, while 234m
    # forbids 1m. The other call options must retain those discards.
    numpy.testing.assert_array_equal(
        base["call_discard_legal"][0, :, 3], [0] * 4
    )
    numpy.testing.assert_array_equal(
        tf.gather(base["call_discard_legal"][0], [0, 6], axis=1),
        [[1, 0], [1, 1], [0, 1], [1, 1]],
    )
    numpy.testing.assert_array_equal(
        calls["discard_action_legal"],
        tf.gather(base["call_discard_legal"][0], rows["post_call_option"]),
    )
    network = joint_network.JointPolicyPayment()
    policy = network.actor
    encoded = []
    candidate_heads = policy.candidate_heads

    def capture(features, outputs):
        candidate_heads(features, outputs)
        encoded.append(dict(outputs))

    monkeypatch.setattr(policy, "candidate_heads", capture)
    with tf.GradientTape() as tape:
        outputs = policy(rows)
        probabilities = tf.nn.softmax(outputs["response_policy_logits"])
        loss = probabilities[
            0, 1 + int(numpy.flatnonzero(base["call_discard_legal"][0, 0])[0])
        ]
        loss += probabilities[
            0, 112 + int(numpy.flatnonzero(base["call_discard_legal"][0, 3])[0])
        ]
    gradients = tape.gradient(
        loss,
        [
            policy.call_choice.bias,
            policy.policy.get_layer("final_scores").kernel,
            policy.policy.get_layer("board_block_3_contract").kernel,
            policy.policy.get_layer("decision_candidates").projection.kernel,
            policy.policy.get_layer("river_encoder").projection.kernel,
            policy.policy.get_layer("hand_bank").projection.layers[0].kernel,
            policy.policy.get_layer("neural_features").tile_projection.kernel,
            *network.beliefs.trainable_variables,
            policy.policy.get_layer("candidate_ron").kernel,
            policy.policy.get_layer("route_affinity").kernel,
        ],
    )
    assert all(numpy.any(value.numpy()) for value in gradients[:2])
    assert all(value is None for value in gradients[2:])
    branch = [
        tf.reduce_sum(part).numpy()
        for part in tf.split(probabilities, [1, 37, 37, 37, 37, 1], axis=-1)
    ]
    numpy.testing.assert_allclose(branch, [0.2] * 5 + [0], atol=1e-6)

    # Local encodings contain no call-discard bank rows. Their child discard
    # representations are reused exactly by the joint action-conditioned critic.
    local = policy.policy(base)
    assert not numpy.any(
        (local["selected"][:, 1] >= 112) & (local["selected"][:, 1] < 260)
    )
    called = encoded[-1]
    for name in (
        *beliefs.BELIEF_HEADS,
        "opponent_wait_logits",
        "ron_eligibility_logits",
    ):
        numpy.testing.assert_array_equal(
            called[name], tf.repeat(outputs[name], 4, axis=0)
        )
    numpy.testing.assert_array_equal(
        outputs["candidate_context"][:, 384 : 384 + critic.BELIEF_WIDTH],
        tf.repeat(outputs["belief_context"], len(outputs["selected"]), axis=0),
    )
    # Different accepted calls offering the same cut share immediate danger.
    selected = outputs["selected"].numpy()[:, 1]
    alternatives = numpy.flatnonzero((selected >= 112) & (selected < 260))
    _, first, inverse = numpy.unique(
        (selected[alternatives] - 112) % 37,
        return_index=True,
        return_inverse=True,
    )
    assert len(first) < len(alternatives)
    numpy.testing.assert_array_equal(
        tf.gather(outputs["candidate_ron_logits"], alternatives),
        tf.gather(
            outputs["candidate_ron_logits"], alternatives[first[inverse]]
        ),
    )
    child_rows = numpy.flatnonzero(called["selected"][:, 1] < 37)
    child_selected = tf.gather(called["selected"], child_rows).numpy()
    joint_codes = (
        112
        + 37 * rows["post_call_option"].numpy()[child_selected[:, 0]]
        + child_selected[:, 1]
    )
    joint_rows = numpy.searchsorted(outputs["selected"][:, 1], joint_codes)
    numpy.testing.assert_array_equal(
        outputs["selected"].numpy()[joint_rows, 1], joint_codes
    )
    for name in (
        "candidate",
        "structure",
        "route_affinity_logits",
        "candidate_ron_logits",
    ):
        numpy.testing.assert_array_equal(
            tf.gather(outputs[name], joint_rows),
            tf.gather(called[name], child_rows),
        )
    numpy.testing.assert_array_equal(
        tf.gather(outputs["candidate_completion"], joint_rows),
        tf.concat(
            [
                tf.reshape(
                    tf.gather(
                        tf.sigmoid(called["round_completion_yaku_logits"]),
                        child_selected[:, 0],
                    ),
                    [-1, 4 * 2 * 23],
                ),
                tf.reshape(
                    tf.gather(
                        tf.nn.softmax(called["round_completion_score_logits"]),
                        child_selected[:, 0],
                    ),
                    [-1, 2 * len(payment_scoring.SCORES)],
                ),
            ],
            axis=-1,
        ),
    )
    hand_context = network.context(
        [calls, called["selected"], called["structure"]]
    )
    numpy.testing.assert_array_equal(
        tf.gather(outputs["candidate_context"], joint_rows)[
            :, -hand_context.shape[1] :
        ],
        tf.gather(hand_context, child_rows),
    )
    numpy.testing.assert_array_equal(
        outputs["constraint_loss"], local["constraint_loss"]
    )
    assert numpy.all(calls["meld_tiles"][:, 0, 0, 0] < 37)
    numpy.testing.assert_array_equal(
        payment_scoring.fixed_meld_fu(base)[:, 0], [0]
    )
    numpy.testing.assert_array_equal(
        payment_scoring.fixed_meld_fu(calls)[:, 0],
        tf.where(rows["post_call_option"] == 3, 2, 0),
    )
    numpy.testing.assert_array_equal(
        critic.public_inputs(rows, outputs["selected"])["public_fu"][:, 0], 0
    )
    assert not numpy.any(
        critic.public_inputs(rows, outputs["selected"])["opened"][:, 0]
    )
    assert (
        network.candidate_context(rows, outputs).shape[0]
        == outputs["selected"].shape[0]
    )
    # Two chi-pattern masses must not combine to defeat the pass branch.
    scores = numpy.full((1, 150), -1e9, numpy.float32)
    scores[0, [0, 1, 38]] = numpy.log([0.4, 0.3, 0.3])
    assert (
        int(actor.greedy_choices(tf.constant(scores), tf.constant([2]))[0]) == 0
    )
    monkeypatch.setattr(
        actions,
        "policy_candidates",
        lambda request: [
            ("response_policy_logits", code, code, {}) for code in (0, 1, 38)
        ],
    )
    choice, _ = actions.select_policy_action(
        {"phase": "RESPONSE"}, {"response_policy_logits": scores}, 0
    )
    assert choice == 0
    empty_base = {
        **base,
        **{
            name: tf.zeros_like(base[name])
            for name in (
                "response_legal_mask",
                "call_discard_legal",
            )
        },
    }
    empty = call_states.combine(
        empty_base,
        {name: value[:0] for name, value in calls.items()},
        tf.zeros(0, tf.int32),
        tf.zeros(0, tf.int32),
    )
    numpy.testing.assert_array_equal(policy(empty)["response_policy_logits"], 0)


def test_declined_ron_retains_accepted_pon_discard_state():
    start = {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "5p",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000] * 4,
        "tehais": [
            ["?"] * 13,
            ["1m"] * 3
            + ["2p", "3p", "4p", "7s", "8s", "9s", "E", "E", "S", "S"],
            ["?"] * 13,
            ["?"] * 13,
        ],
    }
    state = replay.Replay.from_start("declined-ron-call", start)
    state.apply({"type": "tsumo", "actor": 0, "pai": "E"})
    state.apply({"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": True})
    legal = state.legal_observation(1).legal_actions()
    assert actions.phase_for_actions(legal) == "WIN"
    packed = features.pack(state, 1, "WIN")
    numpy.testing.assert_array_equal(packed["post_call_option"], [3])
    numpy.testing.assert_array_equal(packed["post_call_decision_phase"], [0])
    numpy.testing.assert_array_equal(
        packed["post_call_discard_action_legal"][0],
        packed["call_discard_legal"][3],
    )
