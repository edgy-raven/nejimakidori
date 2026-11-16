"""First-riichi expert preferences are soft, conditional, and score-aware."""

import numpy
import tensorflow as tf
from log_dataset import amend_riichi_prior, features, replay
from model import actor


def test_expert_prior_thresholds_and_threat_gate():
    # wait count, dama points, extra han, prior discards, safe, riichi, prefer
    cases = [
        (5, 0, 0, 0, True, False, False),
        (6, 0, 0, 12, True, False, True),
        (5, 0, 1, 0, True, False, True),
        (5, 5200, 3, 0, True, False, False),
        (6, 5200, 3, 12, True, False, True),
        (6, 7700, 3, 5, True, False, True),
        (9, 7700, 3, 6, True, False, False),
        (10, 7700, 3, 6, True, False, True),
        (10, 8000, 4, 11, True, False, True),
        (10, 7700, 3, 12, True, False, False),
        (12, 12000, 6, 0, True, False, False),
        (12, 32000, 13, 0, True, False, False),
        (6, 0, 0, 0, False, False, False),
        (6, 0, 0, 0, True, True, False),
    ]
    inputs = {
        "discard_action_all_ukeire_count": tf.constant(
            [[row[0]] * 37 for row in cases]
        ),
        "riichi_dama_points": tf.constant([[row[1]] * 37 for row in cases]),
        "riichi_extra_han": tf.constant([[row[2]] * 37 for row in cases]),
        "opponent_discard_count": tf.constant(
            [[row[3], 0, 0, 0] for row in cases]
        ),
        "policy_prior_no_threat": tf.constant([row[4] for row in cases]),
        "riichi_state": tf.constant([[0, int(row[5]), 0, 0] for row in cases]),
        "riichi_discard_legal_mask": tf.constant(
            [[True] + [False] * 36] * len(cases)
        ),
    }
    numpy.testing.assert_array_equal(
        actor.riichi_prior(inputs)[:, 0], [row[6] for row in cases]
    )
    assert not bool(tf.reduce_any(actor.riichi_prior(inputs)[:, 1:]))


def test_positive_prior_has_gradient_at_tiny_mass_and_is_neutral_elsewhere():
    logits = tf.Variable([[-20.0, 0.0, 0.0]] * 3)
    preference = tf.constant(
        [[True, False, False]] * 2 + [[False, False, False]]
    )
    with tf.GradientTape() as tape:
        loss = 0.2 * actor.expert_prior_loss(
            tf.nn.log_softmax(logits), preference, tf.constant([1, 2, 1])
        )
    gradient = tape.gradient(loss, logits).numpy()
    numpy.testing.assert_allclose(gradient[0], [-1 / 3, 1 / 6, 1 / 6])
    numpy.testing.assert_allclose(gradient[1], gradient[0] / 5)
    numpy.testing.assert_array_equal(gradient[2], 0)


def test_native_prior_and_record_amendment_share_exact_scoring():
    for seat, marker, five, expected in (
        (1, "9s", "5m", (1000, 1)),
        (0, "9s", "5m", (1000, 1)),
        (1, "1p", "5m", (3900, 3)),
        (1, "9s", "5mr", (2000, 2)),
    ):
        start = dict(
            type="start_kyoku",
            bakaze="E",
            dora_marker=marker,
            honba=3,
            kyoku=1,
            kyotaku=2,
            oya=0,
            scores=[25000] * 4,
            tehais=[["?"] * 13 for _ in range(4)],
        )
        start["tehais"][
            seat
        ] = f"1m 2m 3m 4m {five} 6m 7p 8p 9p 6s E 2p 2p".split()
        state = replay.Replay.from_start("expert-prior", start)
        state.apply(dict(type="tsumo", actor=seat, pai="7s"))
        packed = features.pack(state, seat, "RIICHI")
        assert packed["policy_prior_no_threat"] == 1
        assert (
            packed["riichi_dama_points"][27],
            packed["riichi_extra_han"][27],
        ) == expected
        amended = amend_riichi_prior.prior_values(packed, False)
        for name in amend_riichi_prior.FIELDS:
            numpy.testing.assert_array_equal(amended[name], packed[name])
        # Reuse the same public threat detector as focused sampling.
        state.apply(dict(type="reach", actor=(seat + 1) % 4))
        threatened = features.pack(state, seat, "RIICHI")
        assert threatened["policy_prior_no_threat"] == 0


def test_calls_prefer_post_call_value_or_open_shanten_advance():
    from log_dataset import _native_collection

    inputs = {
        name: tf.zeros([8, *spec["shape"]], tf.as_dtype(spec["dtype"]))
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["policy_prior_no_threat"] = tf.constant([1] * 7 + [0], tf.int8)
    inputs["normal_self_shanten"] = tf.constant([2] * 8, tf.int8)
    # Closed 3900+, cheap riichi, expensive riichi, no yaku; open advance,
    # open unchanged, non-response, outside threat.
    inputs["meld_valid"] = tf.constant(
        [[[int(row in (4, 5)), 0, 0, 0]] + [[0] * 4] * 3 for row in range(8)],
        tf.int8,
    )
    inputs["chi_discard_legal"] = tf.constant(
        [[[1] + [0] * 36] + [[0] * 37] * 2] * 8, tf.int8
    )
    inputs["chi_discard_shanten"] = tf.constant(
        [[[2 if row == 5 else 1] * 37] * 3 for row in range(8)], tf.int8
    )
    inputs["call_open_points"] = tf.constant(
        [
            [[value] * 37] * 4
            for value in (3900, 1000, 1000, 0, 0, 0, 3900, 3900)
        ],
        tf.float32,
    )
    inputs["call_riichi_points"] = tf.constant(
        [
            [[value] * 37] * 4
            for value in (8000, 5200, 7700, 2000, 0, 0, 8000, 8000)
        ],
        tf.float32,
    )
    result = actor.expert_prior(inputs, tf.constant([2] * 6 + [0, 2]))
    numpy.testing.assert_array_equal(
        result[:, 1], [True, True, False, False, True, False, False, False]
    )
    assert not bool(tf.reduce_any(result[:, 0]))
    assert not bool(tf.reduce_any(result[:, 2:]))


def test_native_call_prior_matches_amended_post_call_hand():
    start = dict(
        type="start_kyoku",
        bakaze="E",
        dora_marker="9s",
        honba=0,
        kyoku=1,
        kyotaku=0,
        oya=0,
        scores=[25000] * 4,
        tehais=[["?"] * 13 for _ in range(4)],
    )
    start["tehais"][1] = "2m 3m 4p 5p 6p 6s 7s 8s 2p 2p 5s 6s E".split()
    state = replay.Replay.from_start("call-prior", start)
    state.apply(dict(type="dahai", actor=0, pai="4m", tsumogiri=True))
    packed = features.pack(state, 1, "RESPONSE")
    assert packed["policy_prior_no_threat"] == 1
    assert numpy.any(packed["call_open_points"] > 0)
    amended = amend_riichi_prior.prior_values(packed, False)
    for name in amend_riichi_prior.FIELDS:
        numpy.testing.assert_array_equal(amended[name], packed[name])
