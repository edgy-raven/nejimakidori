"""Correction collection keeps paired roots, not ordinary game decisions."""

import numpy
import tensorflow as tf
from log_dataset import _native_collection
from model import actions
from model import actor as actor_layers
from model import outcomes
from training import self_play
from training.corrections import collection


def test_public_categories_compare_policy_choice_with_a_real_alternative():
    features = numpy.zeros((), self_play.FEATURE_DTYPE)
    features["chiitoitsu_self_shanten"] = 4
    features["kokushi_self_shanten"] = 4
    features["normal_self_shanten"] = 1
    features["live_wall_count"] = 12
    features["discard_action_all_shanten"][1] = 2
    legal = numpy.zeros(150, bool)
    legal[[0, 1, 37]] = True
    policy = numpy.zeros(150, numpy.float32)
    policy[0] = 2
    policy[1] = 1
    for category in ("tenpai", "keiten"):
        assert collection.action_pair(
            features=features,
            head=1,
            legal=legal,
            policy=policy,
            category=category,
            preferred=0,
        ) == (0, 1)
    assert collection.action_pair(
        features=features,
        head=1,
        legal=legal,
        policy=policy,
        category="riichi",
        preferred=0,
    ) == (0, 37)
    features["riichi_state"][1] = 2
    features["genbutsu_to_seat"][1, 1] = 1
    assert collection.action_pair(
        features=features,
        head=1,
        legal=legal,
        policy=policy,
        category="threat",
        preferred=0,
    ) == (0, 1)
    policy[1] = 3
    assert collection.action_pair(
        features=features,
        head=1,
        legal=legal,
        policy=policy,
        category="threat",
        preferred=1,
    ) == (1, 0)
    features["self_efficiency"][0] = 2 / 8
    features["chi_discard_legal"][0, 0] = 1
    features["chi_discard_shanten"][0, 0] = 2
    legal[37] = False
    policy[1] = 1
    assert collection.action_pair(
        features=features,
        head=2,
        legal=legal,
        policy=policy,
        category="calls",
        preferred=0,
    ) == (0, 1)
    features["chi_discard_shanten"][0, 0] = 3
    assert (
        collection.action_pair(
            features=features,
            head=2,
            legal=legal,
            policy=policy,
            category="calls",
            preferred=0,
        )
        is None
    )

    # Production selects declaration by branch mass, not flat argmax.
    legal[[0, 37, 38]] = True
    policy[:] = -1e9
    policy[0] = numpy.log(0.4)
    policy[[37, 38]] = numpy.log(0.3)
    with tf.device("/CPU:0"):
        preferred = int(
            actor_layers.greedy_choices(policy[None], tf.constant([1]))[0]
        )
    assert preferred == 37
    assert collection.action_pair(
        features=features,
        head=1,
        legal=legal,
        policy=policy,
        category="riichi",
        preferred=preferred,
    ) == (37, 0)


def test_seeded_collection_retains_only_paired_root_settlements():
    def predict(features):
        discard = (
            -100 * features["discard_action_all_shanten"].astype(numpy.float32)
            + features["discard_action_all_ukeire_count"]
        )
        outputs = {
            name: numpy.zeros(
                (
                    len(features["decision_phase"]),
                    actions.ACTOR_OUTPUTS[name][0],
                ),
                numpy.float32,
            )
            for name in actions.TRAIN_POLICY_HEADS
        }
        outputs["discard_policy_logits"] = discard
        outputs["riichi_policy_logits"] = numpy.concatenate(
            [discard, discard + 10], axis=1
        )
        outputs["win_decision_logits"][:, 1] = 10
        outputs["response_policy_logits"][:, 0] = 10
        return outputs

    def evaluate(features):
        legal = features["discard_action_legal"].astype(bool)
        preferred = numpy.where(
            legal, predict(features)["discard_policy_logits"], -1e9
        ).argmax(1)
        alternatives = numpy.where(
            legal & (numpy.arange(37) != preferred[:, None]),
            numpy.arange(37),
            -1,
        ).max(1)
        values = numpy.where(legal, 0.0, -numpy.inf)
        values[numpy.arange(len(preferred)), alternatives] = 1.0
        # An agreeing proposal must not hide another qualifying root.
        values[0, preferred[0]] = 2.0
        return {
            "q_values": values,
            # A second actor evaluation may rank a near tie differently;
            # it must not replace the captured policy action or its Q gap.
            "actions": numpy.stack([preferred, alternatives], axis=1),
            "priority": numpy.zeros(len(preferred)),
        }

    batch, statistics = collection.collect(predict, evaluate, 6, 941)
    count = len(batch["actions"])
    assert 0 < count <= 6
    assert statistics["selected_roots"] == count
    assert statistics["root_continuations"] == 2 * count
    assert statistics["disagreement_scored"] == 8
    assert statistics["disagreement_qualifying"] == 7
    assert statistics["selected/disagreement"] == 1
    assert len(numpy.unique(batch["source"][:, 0])) == count
    assert len(numpy.unique(batch["category"])) == count
    assert not batch["source_kind"].any()
    assert batch["settlement"].shape == (count, 2, 12)
    assert batch["payment_points"].shape == (
        count,
        2,
        outcomes.PAYMENT_CELLS.sum(),
    )
    assert numpy.isfinite(batch["settlement"]).all()
    assert (batch["actions"][:, 0] != batch["actions"][:, 1]).all()
    assert numpy.take_along_axis(batch["legal"], batch["actions"], axis=1).all()
    assert "chosen" not in batch
    assert "behavior" not in batch
    assert all(len(value) == count for value in batch["features"].values())
    assert (
        sum(
            statistics[f"assigned/{category}"]
            for category in collection.CATEGORIES
        )
        == 6
    )


def test_correction_continuations_are_greedy_except_the_forced_root():
    native = _native_collection.NativeCollector(1, 941)
    request = native.next()
    legal = numpy.frombuffer(request["legal"], bool).reshape(-1, 150)
    codes = numpy.flatnonzero(legal[0])
    assert request["head"][0] == 0
    assert len(codes) > 1
    root = native.snapshot(0)

    def predict(features):
        return {
            name: numpy.broadcast_to(
                numpy.arange(actions.ACTOR_OUTPUTS[name][0]) * 0.01,
                (len(features), actions.ACTOR_OUTPUTS[name][0]),
            ).astype(numpy.float32)
            for name in actions.TRAIN_POLICY_HEADS
        }

    batch, _ = self_play.collect_streams(
        predict,
        [
            self_play.collection_steps(
                1,
                42,
                collector=root.fork(1, 42, until_hand_end=True),
                forced_root=(root.decision, int(codes[0]), 1),
                greedy=True,
            )
        ],
    )[0]
    forced = batch["paired_root"]
    assert forced.sum() == 1
    assert batch["chosen"][forced][0] == codes[0]
    with tf.device("/CPU:0"):
        expected = actor_layers.greedy_choices(
            batch["behavior"][~forced], batch["head"][~forced]
        ).numpy()
    assert len(expected) > 1
    numpy.testing.assert_array_equal(batch["chosen"][~forced], expected)
    # The forced alternative was not the policy's maximum-logit discard.
    assert codes[0] != codes[-1]
