"""Candidate transitions and immediate endings share one serving contract."""

import json
import pathlib

import numpy
import riichienv
import tensorflow as tf

from log_dataset import payment_bonus, replay, training_targets
from model import critic, feature_vector, features, immediate


def test_calls_kans_and_declarations_change_scoring_before_prediction():
    packed = {
        name: numpy.zeros((3, *spec["shape"]), spec["dtype"])
        for name, spec in feature_vector.INPUTS.items()
    }
    packed["seat_context"][:, :, 0] = 0.25
    packed["live_wall_count"][:] = 20
    packed["decision_phase"][:] = [1, 2, 3]
    packed["riichi_state"][:, 1] = 2
    packed["ippatsu_alive"][:, 1] = 1
    packed["pon_candidate_hand"][0, 0, 0] = 1
    packed["candidate_melds"][0, 1, 27, 1] = 1  # Open east triplet.
    packed["kan_candidate_hand"][1, 0] = 1
    packed["candidate_melds"][1, 1, 0, 3] = 1  # Concealed terminal kan.
    packed = {name: tf.constant(value) for name, value in packed.items()}
    selected = tf.constant([[0, 111], [0, 223], [1, 261], [2, 74]])
    public = critic.public_inputs(packed, selected)
    numpy.testing.assert_array_equal(public["opened"][:, 0], [0, 1, 0, 0])
    numpy.testing.assert_array_equal(public["public_fu"][:, 0], [0, 4, 32, 0])
    numpy.testing.assert_array_equal(
        public["deposit_allowed"][:, 0], [1, 0, 1, 1]
    )
    numpy.testing.assert_array_equal(
        public["deposit_required"][:, 0], [0, 0, 0, 1]
    )
    numpy.testing.assert_array_equal(
        public["ippatsu_allowed"][:, 1], [1, 0, 0, 1]
    )
    # Robbing a kan precedes its successful resolution and preserves ippatsu.
    numpy.testing.assert_array_equal(
        public["immediate_ippatsu"][:, 1], [1, 0, 1, 1]
    )
    numpy.testing.assert_array_equal(
        critic.immediate_decisions(packed, selected), [0, 1, 1, 1]
    )
    model = critic.PaymentCritic()
    result = model.distributions(
        {
            **public,
            "state": tf.zeros((4, 64)),
            "immediate_allowed": critic.immediate_decisions(packed, selected),
            "immediate_safe": tf.zeros((4, 3), tf.bool),
            "immediate_wait": tf.zeros((4, 3)),
        }
    )
    own_wins = [
        i for i, cell in enumerate(payment_bonus.CELLS[:24]) if cell[2] == 0
    ]
    closed = [
        i
        for i, scenario in enumerate(payment_bonus.SCENARIOS)
        if scenario[0] != 0
    ]
    opened = [
        i
        for i, scenario in enumerate(payment_bonus.SCENARIOS)
        if scenario[0] == 0
    ]
    probability = tf.nn.softmax(result["bonus_logits"]).numpy()
    assert not probability[1][numpy.ix_(own_wins, closed)].any()
    assert not probability[3][numpy.ix_(own_wins, opened)].any()
    assert probability[1, own_wins].sum() > 0


def test_post_call_ron_labels_match_native_post_call_discard_state():
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    state = replay.Replay("call-ron", events)
    checked = 0
    for _, actor, phase, _, _ in state.decisions():
        if phase != "RESPONSE":
            continue
        labels = training_targets.ron_labels(state, actor, phase)
        for action in features.prefer_red_calls(
            state.legal_observation(actor).legal_actions()
        ):
            if action.action_type not in (
                riichienv.ActionType.CHI,
                riichienv.ActionType.PON,
            ):
                continue
            after = replay.Replay("call-ron-after", events)
            for event in state.history:
                after.apply(event)
            for seat in range(4):
                if after.declared[seat] and not after.accepted[seat]:
                    after.apply({"type": "reach_accepted", "actor": seat})
            after.apply(
                {
                    **features.action_event(action),
                    "actor": actor,
                    "target": state.pending_discard["actor"],
                }
            )
            expected = training_targets.ron_labels(after, actor, "DISCARD")
            valid = expected["ron_legal"] >= 0
            numpy.testing.assert_array_equal(
                labels["ron_legal"][valid], expected["ron_legal"][valid]
            )
            numpy.testing.assert_array_equal(
                labels["ron_payments"][valid], expected["ron_payments"][valid]
            )
            checked += 1
    assert checked > 0


def test_immediate_draws_respect_known_readiness():
    model = immediate.ImmediateOutcome()
    public = {
        "state": tf.zeros((3, 64)),
        "dealer": tf.constant([0, 0, 0]),
        "opened": tf.zeros((3, 4), tf.bool),
        "has_meld": tf.zeros((3, 4), tf.bool),
        "riichi_required": tf.zeros((3, 4), tf.bool),
        "immediate_ippatsu": tf.zeros((3, 4), tf.bool),
        "immediate_ankan": tf.zeros(3, tf.bool),
        "public_fu": tf.zeros((3, 4), tf.int32),
        "immediate_safe": tf.ones((3, 3), tf.bool),
        "immediate_wait": tf.ones((3, 3)),
        "immediate_allowed": tf.constant([True] * 3),
        "immediate_draw_kind": tf.constant([0, 1, 2]),
        "immediate_tenpai": tf.constant([[1, 1, -1, -1]] * 3),
    }
    result = model(public)
    # Three separate ron events preserve their marginals, including pairs.
    claims = model(
        {
            **public,
            "immediate_safe": tf.zeros((3, 3), tf.bool),
            "immediate_draw_kind": tf.zeros(3, tf.int32),
        }
    )
    masks = numpy.array(
        [[bool(mask & (1 << seat)) for seat in range(3)] for mask in range(8)],
        numpy.float32,
    )
    numpy.testing.assert_allclose(
        claims["immediate_claim_probability"] @ masks,
        claims["immediate_ron_probability"],
        atol=1e-6,
    )
    assert numpy.all(claims["immediate_probability"][:, 22:] > 0)
    kan = model(
        {
            **public,
            "immediate_safe": tf.zeros((3, 3), tf.bool),
            "immediate_ankan": tf.ones(3, tf.bool),
            "has_meld": tf.constant([[False, True, False, True]] * 3),
        }
    )
    numpy.testing.assert_array_equal(
        tf.gather(kan["immediate_ron_probability"], [0, 2], axis=1), 0
    )
    assert numpy.all(kan["immediate_ron_probability"][:, 1] > 0)
    probability = result["immediate_probability"].numpy()
    numpy.testing.assert_allclose(probability.sum(-1), 1, atol=1e-6)
    assert probability[0, 0] == 1
    assert probability[2, 20] == 1
    assert not probability[1, :4].any()
    masks = numpy.arange(16)
    assert not probability[1, 4:20][(masks & 3) != 3].any()
    with tf.GradientTape() as tape:
        result = model(public)
        loss = -tf.math.log(result["immediate_probability"][1, 19])
    assert (
        tf.linalg.global_norm(tape.gradient(loss, model.trainable_variables))
        > 0
    )


def test_irreversible_melds_remove_impossible_yaku_and_attack_routes():
    from model import yaku_constraints

    melds = numpy.zeros((3, 34, 6), numpy.int32)
    melds[0, 27, 1] = 1  # Open east pon.
    melds[1, 1, 0] = 1  # Fixed 234m sequence.
    melds[1, 10, 0] = 1  # A second suit also rules out a flush.
    melds[2, 0, 3] = 1  # Ankan stays closed but precludes seven pairs.
    allowed = yaku_constraints.support(tf.constant(melds)).numpy()
    assert not allowed[0, [0, 1, 2, 16, 19, 20, 21]].any()
    assert allowed[0, [3, 6, 10, 12, 18]].all()
    assert not allowed[1, [10, 11, 12, 14, 15, 16, 18, 19, 21]].any()
    assert allowed[1, 1]  # Tanyao is still possible.
    assert not allowed[2, [0, 1, 16, 20]].any()
    assert allowed[2, 2]  # Concealed kan does not rule out iipeikou.


def test_completion_score_support_respects_ron_tsumo_and_fixed_meld_fu():
    from model import payment_scoring

    public = {
        "meld_valid": tf.constant([[[1, 0, 0, 0]] * 4]),
        "meld_type": tf.constant([[[4, 0, 0, 0]] * 4]),
        "meld_tile_ids": tf.zeros((1, 4, 4, 4), tf.int32),
        "riichi_state": tf.zeros((1, 4), tf.int32),
    }
    logits = payment_scoring.CompletionScoreMask()(
        [tf.zeros((1, 4, 2, len(payment_scoring.SCORES))), public]
    )
    probabilities = tf.nn.softmax(logits).numpy()
    for score in ((1, 20), (2, 25), (2, 50)):
        assert not probabilities[..., payment_scoring.SCORES.index(score)].any()
    assert probabilities[..., payment_scoring.SCORES.index((5, 0))].min() > 0

    public["riichi_state"] = tf.ones((1, 4), tf.int32)
    public["meld_valid"] = tf.zeros((1, 4, 4), tf.int32)
    probability = tf.nn.softmax(
        payment_scoring.CompletionScoreMask()([tf.zeros_like(logits), public])
    ).numpy()
    one_han = payment_scoring.SCORE_HAN == 1
    assert probability[:, :, 0, one_han].sum() > 0
    assert not probability[:, :, 1, one_han].any()
    scenarios = ((2, 0, 0, 0), (3, 3, 1, 0), (2, 5, 0, 0), (0, 0, 0, 0))
    probability = tf.nn.softmax(
        payment_scoring.masked_score_logits(
            tf.zeros((1, 24, 4, len(payment_scoring.SCORES))),
            tf.constant([0]),
            tf.zeros((1, 4), tf.int32),
            scenarios,
        )
    ).numpy()[0]
    # Riichi ron can be one han; closed tsumo adds another guaranteed han.
    assert probability[:12, 0, one_han].sum() > 0
    assert not probability[12:, 0, one_han].any()
    assert not probability[
        :,
        1:3,
        (payment_scoring.SCORE_HAN > 0) & (payment_scoring.SCORE_HAN < 6),
    ].any()
    # A seven-han floor must retain the shared 6/7 haneman tier.
    assert probability[12:, 1:3, payment_scoring.SCORES.index((6, 0))].min() > 0

    assert not probability[12:, 0, payment_scoring.SCORES.index((2, 20))].any()
    assert not probability[:12, 0, payment_scoring.SCORES.index((2, 25))].any()
    assert not probability[12:, 0, payment_scoring.SCORES.index((3, 25))].any()
    assert probability[12:, 0, payment_scoring.SCORES.index((3, 20))].min() > 0
    assert probability[:12, 0, payment_scoring.SCORES.index((3, 25))].min() > 0
