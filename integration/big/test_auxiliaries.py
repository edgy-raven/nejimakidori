"""Direct route and ron supervision stays separate from policy gradients."""

import json
import pathlib

import numpy
import riichienv
import tensorflow as tf

from log_dataset import records, replay, scoring
from model import actor as actor_layers
from model import candidate_predictions, losses
from model import network as joint_network
from training import restoration


def test_specialist_targets_drive_tanyao_and_early_liability_gradients():
    # Tanyao survives every winning tile of these simple-only shortest routes.
    counts = numpy.bincount(
        [1, 2, 3, 11, 12, 13, 21, 22, 23, 23, 24, 25, 10], minlength=34
    )
    affinity = scoring.hand_yaku_fractions(
        counts=counts,
        unavailable=counts,
        melds=[],
        player_wind=0,
        round_wind=0,
    )
    assert affinity[scoring.NAMES.index("tanyao")] == 1
    for hand in (
        counts,
        numpy.bincount(
            [0, 1, 2, 9, 10, 11, 18, 19, 20, 3, 4, 5, 27], minlength=34
        ),
    ):
        full = scoring.hand_yaku_fractions(
            counts=hand,
            unavailable=hand + numpy.eye(34, dtype=int)[6],
            melds=[],
            player_wind=0,
            round_wind=0,
        )
        selected = scoring.hand_yaku_fractions(
            counts=hand,
            unavailable=hand + numpy.eye(34, dtype=int)[6],
            melds=[],
            player_wind=0,
            round_wind=0,
            names=("tanyao", "honitsu", "chanta"),
        )
        numpy.testing.assert_array_equal(
            selected,
            full[
                [
                    scoring.NAMES.index(name)
                    for name in ("tanyao", "honitsu", "chanta")
                ]
            ],
        )
        numpy.testing.assert_array_equal(
            selected,
            scoring.hand_yaku_fractions(
                counts=hand + numpy.eye(34, dtype=int)[6],
                unavailable=hand + numpy.eye(34, dtype=int)[6],
                melds=[],
                player_wind=0,
                round_wind=0,
                cuts=(6,),
                names=("tanyao", "honitsu", "chanta"),
            ),
        )
    open_counts = numpy.bincount(
        [1, 2, 3, 11, 12, 13, 21, 22, 23, 10], minlength=34
    )
    blocked = scoring.hand_yaku_fractions(
        counts=open_counts,
        unavailable=open_counts + numpy.eye(34, dtype=int)[27] * 3,
        melds=[riichienv.Meld(riichienv.MeldType.Pon, [108, 109, 110], True)],
        player_wind=0,
        round_wind=0,
    )
    assert blocked[scoring.NAMES.index("tanyao")] == 0
    # Shared completion evidence must retain meld and wind context.
    for player_wind, round_wind in ((0, 0), (1, 0), (0, 1), (0, 0)):
        wind = scoring.hand_yaku_fractions(
            counts=open_counts,
            unavailable=open_counts + numpy.eye(34, dtype=int)[28] * 3,
            melds=[
                riichienv.Meld(riichienv.MeldType.Pon, [112, 113, 114], True)
            ],
            player_wind=player_wind,
            round_wind=round_wind,
        )
        assert wind[scoring.NAMES.index("seat_wind")] == (player_wind == 1)
        assert wind[scoring.NAMES.index("round_wind")] == (round_wind == 1)
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/sakigiri_swap.json")
        .read_text()
    )
    rows = list(replay.Replay("swap", events).observations())
    features, labels = records.parse_batch(
        [records.serialize_observation(row) for row in rows]
    )
    network = joint_network.JointPolicyPayment()
    outputs = network.actor(features)
    terms = candidate_predictions.loss_terms(features, labels, outputs)
    assert all(numpy.isfinite(value) for value in terms.values())
    assert "sakigiri_order_loss" not in terms
    assert terms["route_affinity_loss"] > 0
    with tf.GradientTape(persistent=True) as tape:
        outputs = network.actor(features)
        terms = candidate_predictions.loss_terms(features, labels, outputs)
        head, chosen, legal = restoration.policy_choices(features, labels)
        policy_loss = -tf.reduce_mean(
            tf.gather(
                actor_layers.distribution(outputs, head, legal),
                chosen,
                batch_dims=1,
            )
        )
    for term, layer in (
        ("route_affinity_loss", "route_affinity"),
        ("candidate_ron_loss", "candidate_ron"),
    ):
        gradient = tape.gradient(
            terms[term],
            network.actor.policy.get_layer(layer).trainable_variables,
        )
        assert all(value is not None for value in gradient)
        assert numpy.isfinite(tf.linalg.global_norm(gradient))
        assert tf.linalg.global_norm(gradient) > 0
    assert tape.gradient(
        terms["candidate_ron_loss"],
        network.actor.fusion.trainable_variables,
    ) == [None] * len(network.actor.fusion.trainable_variables)
    assert (
        tape.gradient(
            terms["candidate_ron_loss"],
            network.actor.policy.get_layer(
                "decision_candidates"
            ).projection.kernel,
        )
        is None
    )
    policy_gradients = tape.gradient(
        policy_loss,
        [
            network.actor.policy.get_layer("policy_evidence").kernel,
            network.actor.policy.get_layer("board_block_3_contract").kernel,
            network.actor.policy.get_layer(
                "decision_candidates"
            ).projection.kernel,
            network.actor.policy.get_layer("river_encoder").projection.kernel,
            network.actor.policy.get_layer("hand_bank")
            .projection.layers[0]
            .kernel,
            *network.beliefs.trainable_variables,
            network.actor.policy.get_layer("candidate_ron").kernel,
            network.actor.policy.get_layer("route_affinity").kernel,
        ],
    )
    assert tf.linalg.norm(policy_gradients[0]) > 0
    assert all(value is None for value in policy_gradients[1:])
    del tape
    unknown = {
        **labels,
        "candidate_yaku_affinity": -tf.ones_like(
            labels["candidate_yaku_affinity"]
        ),
        "ron_legal": -tf.ones_like(labels["ron_legal"]),
    }
    unknown_terms = candidate_predictions.loss_terms(features, unknown, outputs)
    assert unknown_terms["route_affinity_loss"] == 0
    assert unknown_terms["candidate_ron_loss"] == 0


def test_partial_route_labels_keep_equal_decision_weight():
    # Decision 0 has one known and one unknown candidate; decision 1 has
    # one known candidate; decision 2 is wholly unknown and must not dilute.
    loss_values = tf.Variable(
        [[1.0, 1.0, 1.0], [9.0, 9.0, 9.0], [3.0, 3.0, 3.0], [7.0, 7.0, 7.0]]
    )
    valid = tf.constant([[True] * 3, [False] * 3, [True] * 3, [False] * 3])
    rows = tf.constant([0, 0, 1, 2])
    with tf.GradientTape() as tape:
        loss = losses.candidate_mean(loss_values, valid, rows)
    numpy.testing.assert_allclose(loss, 2)
    numpy.testing.assert_allclose(
        tf.reduce_sum(tape.gradient(loss, loss_values), axis=-1),
        [0.5, 0, 0.5, 0],
    )
    # Duplicating a known candidate changes neither the decision's weight
    # nor the mean.
    duplicated = tf.constant([0, 0, 1, 2, 3])
    numpy.testing.assert_allclose(
        losses.candidate_mean(
            tf.gather(loss_values, duplicated),
            tf.gather(valid, duplicated),
            tf.gather(rows, duplicated),
        ),
        2,
    )
    assert losses.candidate_mean(loss_values, tf.zeros_like(valid), rows) == 0


def test_takame_yaku_and_han_fu_share_a_completion_without_a_win():
    from log_dataset import features, shanten, training_targets
    from model import payment_scoring

    # 1s completes sanshoku; 4s only pinfu. The pair of han/fu differs
    # between ron and tsumo, and neither is an observed winning outcome.
    hand = riichienv.parse_hand("123m123p55p23s789s")[0]
    remaining = numpy.array([tile for tile in range(136) if tile not in hand])
    numpy.random.default_rng(4).shuffle(remaining)
    event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 3,
        "scores": [25000] * 4,
        "dora_marker": "N",
        "tehais": [
            [
                shanten.TILE_NAMES[features.native_tile_code(int(tile))]
                for tile in tiles
            ]
            for tiles in [hand, *remaining[:39].reshape(3, 13)]
        ],
    }
    targets = training_targets.round_targets(
        replay.Replay.from_start("takame", event), [event]
    )
    numpy.testing.assert_array_equal(
        targets["completion_score"][0, 0],
        [
            payment_scoring.SCORES.index((3, 30)),
            payment_scoring.SCORES.index((4, 20)),
        ],
    )
    expected = numpy.zeros(23, numpy.int8)
    expected[
        [scoring.NAMES.index("pinfu"), scoring.NAMES.index("sanshoku_doujun")]
    ] = 1
    numpy.testing.assert_array_equal(
        targets["completion"][0, 0], numpy.stack([expected, expected])
    )
    numpy.testing.assert_array_equal(targets["completion_score"][0, 1:], -1)

    # Triple yakuman is a hand-value class, including dealer ron at 144000.
    hand = riichienv.parse_hand("111z555666777z2z")[0]
    remaining = numpy.array([tile for tile in range(136) if tile not in hand])
    numpy.random.default_rng(4).shuffle(remaining)
    event.update(
        {
            "oya": 0,
            "dora_marker": "9m",
            "tehais": [
                [
                    shanten.TILE_NAMES[features.native_tile_code(int(tile))]
                    for tile in tiles
                ]
                for tiles in [hand, *remaining[:39].reshape(3, 13)]
            ],
        }
    )
    targets = training_targets.round_targets(
        replay.Replay.from_start("triple", event), [event]
    )
    numpy.testing.assert_array_equal(
        targets["completion_score"][0, 0], payment_scoring.SCORES.index((0, 3))
    )
    assert (
        payment_scoring.PAYMENTS_BY_FACTOR[
            3, payment_scoring.SCORES.index((0, 3))
        ]
        == 144000
    )


def test_ron_risk_factors_wait_and_eligibility_and_masks_safe_discards():
    # Red/regular five and dama/riichi query the same tile-wise posterior.
    # Opponent two discarded it. A response pass or kan offers no discard.
    wait = tf.Variable(tf.zeros((4, 3, 34)))
    eligibility = tf.Variable(tf.zeros((4, 3, 34)))
    inputs = {
        "decision_phase": tf.constant([0, 3, 1, 2]),
        "known_unavailable_counts": tf.zeros((4, 34), tf.int8),
        "meld_tiles": tf.fill((4, 4, 4, 4), tf.constant(37, tf.int8)),
        "genbutsu_to_seat": tf.scatter_nd(
            [[0, 2, 4], [1, 2, 4]], [1, 1], (4, 4, 34)
        ),
    }
    selected = tf.constant(
        [[0, 4], [0, 34], [1, 41], [1, 78], [2, 111], [3, 261]]
    )
    with tf.GradientTape() as tape:
        logits = candidate_predictions.RonRisk()(
            [eligibility, wait, inputs, selected]
        )
        probability = tf.sigmoid(logits)
        total = tf.reduce_sum(probability)
    numpy.testing.assert_allclose(
        probability, [[0.25, 0, 0.25]] * 4 + [[0, 0, 0]] * 2, atol=1e-7
    )
    eligibility_grad, wait_grad = tape.gradient(total, [eligibility, wait])
    numpy.testing.assert_allclose(
        eligibility_grad,
        tf.scatter_nd(
            [[0, 0, 4], [0, 2, 4], [1, 0, 4], [1, 2, 4]],
            [0.25] * 4,
            (4, 3, 34),
        ),
        atol=1e-7,
    )
    assert wait_grad is None
