"""Visible attack routes and verified early defense orders train the actor."""

import json
import pathlib

import numpy
import riichienv
import tensorflow as tf

from log_dataset import records, replay, scoring
from model import critic, objectives, specialists


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
    features, labels, _ = records.parse_batch(
        [records.serialize_observation(row, False) for row in rows]
    )
    network = critic.JointPolicyPayment()
    outputs = network.actor(features)
    terms = specialists.loss_terms((features, labels, None), outputs)
    assert all(numpy.isfinite(value) for value in terms.values())
    assert terms["sakigiri_defense_loss"] > 0
    assert terms["attack_affinity_loss"] > 0
    with tf.GradientTape(persistent=True) as tape:
        outputs = network.actor(features)
        terms = specialists.loss_terms((features, labels, None), outputs)
        policy_loss = objectives.policy_loss(labels, outputs)
    for term, layer in (
        ("attack_affinity_loss", "attack_predictions"),
        ("defense_ron_loss", "defense_predictions"),
        ("sakigiri_defense_loss", "defense_decisions"),
    ):
        gradient = tape.gradient(
            terms[term], network.actor.get_layer(layer).trainable_variables
        )
        assert all(value is not None for value in gradient)
        assert numpy.isfinite(tf.linalg.global_norm(gradient))
        assert tf.linalg.global_norm(gradient) > 0
    # AWR/final-policy gradients can alter count estimates through projection.
    gradient = tape.gradient(
        policy_loss,
        network.beliefs.heads["opponent_hand_count_logits"].trainable_variables,
    )
    assert tf.linalg.global_norm(gradient) > 0
    del tape
    scores = tf.Variable(tf.zeros((len(rows), 37)))
    with tf.GradientTape() as tape:
        loss = specialists.order_loss(labels, scores)
    gradient = tape.gradient(loss, scores).numpy()
    for row_ix, row in enumerate(rows):
        if row.sakigiri_weight > 0:
            assert gradient[row_ix, row.sakigiri_pair[0]] < 0
            assert gradient[row_ix, row.sakigiri_pair[1]] > 0
        else:
            numpy.testing.assert_array_equal(gradient[row_ix], 0)
    # Entirely unlabeled batches contribute neither loss nor a gradient.
    unlabeled = {
        **labels,
        "sakigiri_pair": tf.fill((len(rows), 2), -1),
        "sakigiri_weight": tf.zeros(len(rows)),
    }
    with tf.GradientTape() as tape:
        loss = specialists.order_loss(unlabeled, scores)
    assert loss == 0
    numpy.testing.assert_array_equal(tape.gradient(loss, scores), 0)
    unknown = {
        **labels,
        "candidate_yaku_affinity": -tf.ones_like(
            labels["candidate_yaku_affinity"]
        ),
    }
    assert (
        specialists.loss_terms((features, unknown, None), outputs)[
            "attack_affinity_loss"
        ]
        == 0
    )


def test_partial_route_labels_keep_equal_decision_weight():
    # Decision 0 has one known and one unknown candidate; decision 1 has
    # one known candidate; decision 2 is wholly unknown and must not dilute.
    losses = tf.Variable(
        [[1.0, 1.0, 1.0], [9.0, 9.0, 9.0], [3.0, 3.0, 3.0], [7.0, 7.0, 7.0]]
    )
    valid = tf.constant([[True] * 3, [False] * 3, [True] * 3, [False] * 3])
    rows = tf.constant([0, 0, 1, 2])
    with tf.GradientTape() as tape:
        loss = objectives.candidate_mean(losses, valid, rows)
    numpy.testing.assert_allclose(loss, 2)
    numpy.testing.assert_allclose(
        tf.reduce_sum(tape.gradient(loss, losses), axis=-1), [0.5, 0, 0.5, 0]
    )
    # Duplicating a known candidate changes neither the decision's weight
    # nor the mean. Packed replica statistics must use the same definition.
    duplicated = tf.constant([0, 0, 1, 2, 3])
    means = objectives.MaskedMeans()
    objectives.candidate_mean(
        tf.gather(losses, duplicated),
        tf.gather(valid, duplicated),
        tf.gather(rows, duplicated),
        mean=means.collect,
    )
    means.reduce()
    numpy.testing.assert_allclose(
        objectives.candidate_mean(
            tf.gather(losses, duplicated),
            tf.gather(valid, duplicated),
            tf.gather(rows, duplicated),
            mean=means.read,
        ),
        2,
    )
    assert objectives.candidate_mean(losses, tf.zeros_like(valid), rows) == 0


def test_takame_yaku_and_han_fu_share_a_completion_without_a_win():
    from log_dataset import shanten, training_targets
    from model import features, payment_scoring

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
    numpy.testing.assert_array_equal(targets["outcomes"], 0)
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
