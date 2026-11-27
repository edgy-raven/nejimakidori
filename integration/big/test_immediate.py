"""Candidate transitions and terminal boundaries constrain settlement."""

import json
import pathlib

import numpy
import riichienv
import tensorflow as tf

from integration import feature_reference
from log_dataset import (
    _native_collection,
    features,
    replay,
    scoring,
    training_targets,
)
from model import critic, payment_scoring


def test_fixed_meld_fu_uses_candidate_self_and_public_opponent_groups():
    packed = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    packed["meld_tiles"][...] = 37
    packed["candidate_meld_tiles"][...] = 34
    packed["last_tedashi_tile_id"][...] = 37
    packed["candidate_fu"][0, 0] = 8
    # Simple/terminal pons; simple/terminal open kans; closed honor/added kan.
    packed["meld_type"][0, 1:, :2] = [[2, 2], [3, 3], [4, 5]]
    packed["meld_tiles"][0, 1:, :2] = numpy.array([[1, 0], [1, 0], [27, 1]])[
        ..., None
    ]
    packed["meld_tiles"][0, 1, :2, 3] = 37
    numpy.testing.assert_array_equal(
        payment_scoring.fixed_meld_fu(
            {name: tf.constant(value) for name, value in packed.items()}
        ),
        [[8, 6, 24, 40]],
    )


def test_attempted_calls_preserve_possible_preemption_outcomes():
    packed = {
        name: numpy.zeros((3, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    packed["meld_tiles"][...] = 37
    packed["candidate_meld_tiles"][...] = 34
    packed["last_tedashi_tile_id"][...] = 37
    packed["scores"][:] = 25000
    packed["live_wall_count"][:] = 20
    packed["decision_phase"][:] = [1, 2, 3]
    packed["riichi_state"][:, 1] = 2
    packed["ippatsu_alive"][:, 1] = 1
    packed["kan_candidate_hand"][1, 0] = 1
    packed["candidate_meld_tiles"][1, 1, 0] = 0
    packed["candidate_meld_types"][1, 1, 0] = 3  # Concealed terminal kan.
    packed["dora_multiplicity"][1, 0] = 2
    feature_reference.public_scoring_features(packed)
    packed = {name: tf.constant(value) for name, value in packed.items()}
    selected = tf.constant([[0, 111], [0, 223], [1, 261], [2, 74]])
    public = critic.public_inputs(packed, selected)
    numpy.testing.assert_array_equal(public["opened"][:, 0], [0, 0, 0, 0])
    numpy.testing.assert_array_equal(public["public_fu"][:, 0], [0, 0, 32, 0])
    numpy.testing.assert_array_equal(public["public_bonus"][:, 0], [0, 0, 8, 0])
    numpy.testing.assert_array_equal(
        public["deposit_allowed"][:, 0], [1, 1, 1, 1]
    )
    numpy.testing.assert_array_equal(
        public["deposit_required"][:, 0], [0, 0, 0, 1]
    )
    model = critic.PaymentCritic(
        belief_offset=0, shanten_offset=critic.BELIEF_WIDTH + 4, player_width=8
    )
    result = model.distributions(
        {
            **public,
            "state": tf.zeros(
                (4, critic.BELIEF_WIDTH + critic.PREDICTED_TAIL_WIDTH + 64)
            ),
        }
    )
    own_wins = [i for i, cell in enumerate(scoring.CELLS[:24]) if cell[2] == 0]
    probability = result["score_probability"].numpy()
    assert probability.shape == (4, 24, len(payment_scoring.SCORES))
    # Future high-value outcomes remain learnable without a bonus axis.
    assert (
        probability[1, own_wins, payment_scoring.SCORES.index((8, 0))].sum() > 0
    )
    assert not probability[2, own_wins][
        ..., (payment_scoring.SCORE_HAN > 0) & (payment_scoring.SCORE_HAN < 8)
    ].any()
    # Fixed wind-order fields must reach the corresponding actor-relative
    # player, including the riichi deposit threshold and critic descriptors.
    packed["seat_wind"] = tf.constant([[2, 3, 0, 1]] * 3, tf.int8)
    packed["scores"] = tf.constant([[500, 125000, -50000, 24500]] * 3)
    packed["ranks"] = tf.constant([[3, 1, 4, 2]] * 3, tf.int8)
    numpy.testing.assert_array_equal(
        critic.public_inputs(packed, selected)["deposit_allowed"],
        [[False, False, False, True]] * 4,
    )
    descriptors = critic.PlayerContext()(
        [packed, tf.zeros((3, 3, 256)), selected]
    )
    numpy.testing.assert_allclose(
        descriptors[:, :, 257:259],
        [[[-0.5, 1.0], [0.245, 0.5], [0.005, 0.75], [1.25, 0.25]]] * 4,
    )


def test_response_pass_and_calls_wait_for_priority_on_last_discard():
    packed = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    packed["meld_tiles"][...] = 37
    packed["candidate_meld_tiles"][...] = 34
    packed["last_tedashi_tile_id"][...] = 37
    packed["scores"][:] = 25000
    packed["decision_phase"][:] = 1
    packed["riichi_state"][:] = [0, 1, 2, 0]
    packed["discard_abort"][:] = 1
    feature_reference.public_scoring_features(packed)
    packed = {name: tf.constant(value) for name, value in packed.items()}
    # Pass, chii, pon and open kan all await simultaneous opponent claims.
    selected = tf.constant([[0, 111], [0, 112], [0, 223], [0, 261]])
    public = critic.public_inputs(packed, selected)
    numpy.testing.assert_array_equal(public["ending_draw_kind"], [0] * 4)
    assert not numpy.any(public["deposit_required"])
    numpy.testing.assert_array_equal(
        public["riichi_required"], [[False, False, True, False]] * 4
    )
    # The pending stick remains possible; an already paid stick cannot recur.
    numpy.testing.assert_array_equal(
        public["deposit_allowed"], [[True, True, False, True]] * 4
    )


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


def test_irreversible_melds_remove_impossible_yaku_and_attack_routes():
    melds = numpy.zeros((3, 34, 6), numpy.int32)
    melds[0, 27, 1] = 1  # Open east pon.
    melds[1, 1, 0] = 1  # Fixed 234m sequence.
    melds[1, 10, 0] = 1  # A second suit also rules out a flush.
    melds[2, 0, 3] = 1  # Ankan stays closed but precludes seven pairs.
    allowed = payment_scoring.support(tf.constant(melds)).numpy()
    assert not allowed[0, [0, 1, 2, 16, 19, 20, 21]].any()
    assert allowed[0, [3, 6, 10, 12, 18]].all()
    assert not allowed[1, [10, 11, 12, 14, 15, 16, 18, 19, 21]].any()
    assert allowed[1, 1]  # Tanyao is still possible.
    assert not allowed[2, [0, 1, 16, 20]].any()
    assert allowed[2, 2]  # Concealed kan does not rule out iipeikou.


def test_completion_score_support_respects_ron_tsumo_and_fixed_meld_fu():
    public = {
        "meld_type": tf.constant([[[4, 0, 0, 0]] * 4]),
        "meld_tiles": tf.constant([[[[0, 0, 0, 0]] + [[37] * 4] * 3] * 4]),
        "candidate_fu": tf.fill((1, 16), 32),
        "public_bonus": tf.zeros((1, 4), tf.int32),
        "riichi_state": tf.zeros((1, 4), tf.int32),
        "dora_multiplicity": tf.zeros((1, 34), tf.int32),
    }
    logits = payment_scoring.CompletionScoreMask()(
        [tf.zeros((1, 4, 2, len(payment_scoring.SCORES))), public]
    )
    probabilities = tf.nn.softmax(logits).numpy()
    assert (1, 20) not in payment_scoring.SCORES
    for score in ((2, 25), (2, 50)):
        assert not probabilities[..., payment_scoring.SCORES.index(score)].any()
    assert probabilities[..., payment_scoring.SCORES.index((5, 0))].min() > 0

    public["riichi_state"] = tf.ones((1, 4), tf.int32)
    public["meld_tiles"] = tf.fill((1, 4, 4, 4), 37)
    public["candidate_fu"] = tf.zeros((1, 16), tf.int32)
    probability = tf.nn.softmax(
        payment_scoring.CompletionScoreMask()([tf.zeros_like(logits), public])
    ).numpy()
    one_han = payment_scoring.SCORE_HAN == 1
    assert probability[:, :, 0, one_han].sum() > 0
    assert not probability[:, :, 1, one_han].any()
    probability = tf.nn.softmax(
        payment_scoring.masked_score_logits(
            logits=tf.zeros((2, 24, len(payment_scoring.SCORES))),
            dealer=tf.constant([0, 0]),
            meld_fu=tf.zeros((2, 4), tf.int32),
            opened=tf.zeros((2, 4), tf.bool),
            riichi_required=tf.ones((2, 4), tf.bool),
            bonus=tf.constant([[0] * 4, [5] * 4]),
            yakuman=tf.ones((2, 4), tf.bool),
        )
    ).numpy()
    # Riichi ron can be one han; closed tsumo adds another guaranteed han.
    assert probability[0, :12, one_han].sum() > 0
    assert not probability[0, 12:, one_han].any()
    assert not probability[
        1,
        :,
        ((payment_scoring.SCORE_HAN > 0) & (payment_scoring.SCORE_HAN < 6)),
    ].any()
    # Seven guaranteed han retains the shared 6/7 haneman tier.
    assert probability[1, 12:, payment_scoring.SCORES.index((6, 0))].min() > 0
    assert not probability[0, 12:, payment_scoring.SCORES.index((2, 20))].any()
    assert not probability[0, :12, payment_scoring.SCORES.index((2, 25))].any()
    assert not probability[0, 12:, payment_scoring.SCORES.index((3, 25))].any()
    assert probability[0, 12:, payment_scoring.SCORES.index((3, 20))].min() > 0
    assert probability[0, :12, payment_scoring.SCORES.index((3, 25))].min() > 0


def test_flush_threat_constraints_and_exclusive_yaku_have_gradients():
    public = {
        name: tf.constant(numpy.zeros((1, *spec["shape"]), spec["dtype"]))
        for name, spec in _native_collection.INPUTS.items()
    }
    public["meld_tiles"] = tf.fill((1, 4, 4, 4), tf.constant(37, tf.int8))
    raw = tf.Variable(tf.fill((1, 4, 2, 23), 8.0))
    with tf.GradientTape() as tape:
        logits = payment_scoring.CompletionMask()([raw, public])
        loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(logits), logits=logits
            )
        )
    probability = tf.sigmoid(logits).numpy()
    for first, second in numpy.argwhere(payment_scoring.EXCLUSIVE):
        assert numpy.all(
            probability[..., first] + probability[..., second] <= 1.000001
        )
    gradient = tape.gradient(loss, raw)
    assert numpy.isfinite(gradient).all()
    assert tf.linalg.global_norm([gradient]) > 0

    # The reported 123m / 345m / 999m threat, viewed from another player.
    public = {name: value.numpy() for name, value in public.items()}
    public["meld_type"][0, 2, :3] = [1, 1, 2]
    public["meld_tiles"][0, 2, :3, :3] = [[0, 1, 2], [2, 3, 4], [8, 8, 8]]
    public["dora_multiplicity"][0, 8] = 1
    feature_reference.public_scoring_features(public)
    public = {name: tf.constant(value) for name, value in public.items()}
    probability = tf.sigmoid(
        payment_scoring.CompletionMask()([raw, public])
    ).numpy()[0, 2]
    for name in (
        "yakuman",
        "ittsuu",
        "sanshoku_doujun",
        "sanshoku_doukou",
        "shousangen",
    ):
        assert not probability[:, scoring.NAMES.index(name)].any()
    assert probability[:, scoring.NAMES.index("chinitsu")].min() > 0
    numpy.testing.assert_array_equal(
        tf.cast(public["public_bonus"], tf.int32), [[0, 0, 3, 0]]
    )
    scores = tf.nn.softmax(
        payment_scoring.CompletionScoreMask()(
            [tf.zeros((1, 4, 2, len(payment_scoring.SCORES))), public]
        )
    ).numpy()
    assert not scores[0, 2, :, payment_scoring.SCORE_HAN < 4].any()
    assert scores[0, 2, :, payment_scoring.SCORES.index((8, 0))].min() > 0

    continuation = tf.nn.softmax(
        payment_scoring.masked_score_logits(
            logits=tf.zeros((1, 24, len(payment_scoring.SCORES))),
            dealer=tf.constant([0]),
            meld_fu=payment_scoring.fixed_meld_fu(public),
            opened=critic.public_inputs(public)["opened"],
            riichi_required=critic.public_inputs(public)["riichi_required"],
            bonus=tf.cast(public["public_bonus"], tf.int32),
            yakuman=payment_scoring.support(
                payment_scoring.public_melds(public)
            )[..., -1],
        )
    ).numpy()
    assert not continuation[0, payment_scoring.CELLS[:, 2] == 2][
        ..., payment_scoring.SCORE_HAN < 4
    ].any()

    # Do not erase legal exceptions: green sequences can make ryuuiisou;
    # one fixed non-dragon sequence still leaves three dragon triplets.
    melds = numpy.zeros((3, 34, 6), numpy.float32)
    melds[0, 19, 0] = 1
    melds[0, 23, 1] = 1
    melds[1, 0, 0] = 1
    melds[2, 0, 1] = 1
    assert payment_scoring.support(tf.constant(melds)).numpy()[:, -1].all()


def test_scored_yaku_exclusions_preserve_native_compatible_combinations():
    # Include three-triplet outside hands: sanankou can coexist with both
    # chanta and junchan. Sequence families can coexist with iipeikou.
    for hand in (
        "223344m34555p456s",
        "111123m999p111s11z",
        "111123m999p11199s",
        "112233456789m55p",
        "11223344556677m",
        "1199m1199p1199s11z",
        "111m123p55566677z",
        "11122233344455z",
    ):
        tiles = riichienv.parse_hand(hand)[0]
        score = riichienv.HandEvaluator(tiles, []).calc(
            tiles[-1], [], riichienv.Conditions(tsumo=True)
        )
        assert score.is_win, hand
        ids = {yaku.id for yaku in score.yaku_list()}
        names = [
            scoring.NAMES.index(name)
            for name, value in scoring.YAKU_IDS.items()
            if value in ids
        ]
        if score.yakuman:
            names.append(scoring.NAMES.index("yakuman"))
        assert not payment_scoring.EXCLUSIVE[
            numpy.ix_(names, names)
        ].any(), hand
