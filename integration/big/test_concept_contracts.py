"""Structural families, scored yaku and guaranteed value stay distinct."""

import numpy
import riichienv
import tensorflow as tf

from integration import feature_reference
from log_dataset import _native_collection, scoring
from model import actor as actor_layers
from model import critic, payment_scoring


def test_route_families_include_yakuman_shapes_and_kan_completions():
    for tiles, family in (
        ([0, 0, 0, 8, 8, 8, 9, 9, 9, 17, 17, 17, 18], "chanta"),
        ([0, 0, 0, 8, 8, 8, 9, 9, 9, 27, 27, 27, 18], "chanta"),
        ([1, 1, 1, 3, 3, 3, 10, 10, 10, 12, 12, 12, 23], "tanyao"),
        ([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8], "honitsu"),
    ):
        counts = numpy.bincount(tiles, minlength=34)
        values = scoring.hand_yaku_fractions(
            counts=counts,
            unavailable=counts,
            melds=[],
            player_wind=1,
            round_wind=0,
            names=("tanyao", "honitsu", "chanta"),
        )
        assert values[("tanyao", "honitsu", "chanta").index(family)] == 1
    # Two kans give 16 physical tiles at completion; the convenience scorer
    # would append the winning tile again because 16 % 3 == 1.
    flags = scoring.completion_flags(
        tiles=[36, 37, 38, 72, 73, 74, 108],
        melds=[
            riichienv.Meld(riichienv.MeldType.Ankan, [0, 1, 2, 3], False),
            riichienv.Meld(riichienv.MeldType.Ankan, [32, 33, 34, 35], False),
        ],
        player_wind=1,
        round_wind=0,
        require_all=False,
    )
    assert flags[scoring.NAMES.index("yakuman")] == 1
    assert flags[scoring.NAMES.index("honroutou")] == 0


def test_named_outside_requires_sequence_but_structural_route_does_not():
    melds = numpy.zeros((2, 34, 6), numpy.int8)
    melds[0, [0, 8, 9, 17], 1] = 1
    melds[1, 0, 0] = 2  # Two identical chii still occupy two group slots.
    allowed = payment_scoring.support(tf.constant(melds)).numpy()
    assert not allowed[0, [10, 19]].any()
    assert allowed[0, 11]
    assert not allowed[1, 14]  # Cannot fit three kans beside two chii.
    inputs = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["candidate_melds"][0, 0] = melds[0]
    feature_reference.public_scoring_features(inputs)
    inputs = {name: tf.constant(value) for name, value in inputs.items()}
    route = actor_layers.RouteMask()(
        [tf.zeros((1, 3)), inputs, tf.constant([[0, 0]])]
    )
    assert route.numpy()[0, 2] == 0


def test_guaranteed_han_adds_public_dora_to_situational_yaku():
    inputs = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["meld_valid"][0, 1, 0] = 1
    inputs["meld_type"][0, 1, 0] = 4
    inputs["meld_tile_valid"][0, 1, 0] = 1
    inputs["meld_tile_ids"][0, 1, 0] = 8
    inputs["dora_multiplicity"][0, 8] = 1
    inputs["riichi_state"][0, 1] = 2
    feature_reference.public_scoring_features(inputs)
    inputs = {name: tf.constant(value) for name, value in inputs.items()}
    ordinary_below_haneman = (payment_scoring.SCORE_HAN > 0) & (
        payment_scoring.SCORE_HAN < 6
    )
    completion = tf.nn.softmax(
        payment_scoring.CompletionScoreMask()(
            [tf.zeros((1, 4, 2, len(payment_scoring.SCORES))), inputs]
        )
    ).numpy()
    # Four dora plus riichi plus menzen tsumo: at least six han.
    assert not completion[0, 1, 1, ordinary_below_haneman].any()
    immediate_scores = critic.ImmediateOutcome()(
        {
            **critic.public_inputs(inputs),
            "state": tf.zeros((1, 64)),
            "immediate_wait": tf.ones((1, 3)),
            "immediate_safe": tf.zeros((1, 3), tf.bool),
            "immediate_allowed": tf.ones(1, tf.bool),
            "immediate_ankan": tf.zeros(1, tf.bool),
            "immediate_ippatsu": tf.ones((1, 4), tf.bool),
            "immediate_draw_kind": tf.zeros(1, tf.int32),
            "immediate_tenpai": tf.fill((1, 4), -1),
        }
    )["immediate_score_probability"].numpy()
    # Four dora plus riichi plus ippatsu: at least six han.
    assert not immediate_scores[0, 0, ordinary_below_haneman].any()
    terminal = tf.nn.softmax(
        payment_scoring.masked_score_logits(
            logits=tf.zeros((1, 24, 1, len(payment_scoring.SCORES))),
            dealer=tf.constant([0]),
            meld_fu=tf.cast(inputs["public_fu"], tf.int32),
            bonus_han=(2,),
            opened=tf.zeros((1, 4), tf.bool),
            riichi_required=tf.ones((1, 4), tf.bool),
            bonus=tf.cast(inputs["public_bonus"], tf.int32),
            yakuman=tf.ones((1, 4), tf.bool),
        )
    ).numpy()
    # Four dora plus riichi plus one ura: at least six han.
    assert not terminal[0, payment_scoring.CELLS[:, 2] == 1][
        ..., ordinary_below_haneman
    ].any()
    # A riichi-only ron remains legal: dora is not itself a yaku.
    assert payment_scoring.minimum_han_mask(tf.constant(1)).numpy()[
        payment_scoring.SCORES.index((1, 30))
    ]


def test_bonus_prediction_uses_counts_without_bonus_type_heads():
    inputs = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    feature_reference.public_scoring_features(inputs)
    inputs = {name: tf.constant(value) for name, value in inputs.items()}
    public = critic.public_inputs(inputs, tf.constant([[0, 0]]))
    predictor = critic.PaymentCritic()
    result = predictor.distributions(
        {
            **public,
            "state": tf.zeros((1, 64)),
            "immediate_wait": tf.zeros((1, 3)),
            "immediate_safe": tf.zeros((1, 3), tf.bool),
            "immediate_allowed": tf.zeros(1, tf.bool),
        }
    )
    assert result["bonus_logits"].shape == (1, 24, 14)
    assert result["score_logits"].shape[2] == 14
    numpy.testing.assert_allclose(
        tf.reduce_sum(result["score_probability"], -1), 1, atol=1e-6
    )
