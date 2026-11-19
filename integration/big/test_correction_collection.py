"""Balanced starts retain roots and every descendant for the same MMD loss."""

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
    features["riichi_state"][1] = 2
    features["genbutsu_to_seat"][1, 1] = 1
    for category, preferred, expected in (
        ("tenpai", 0, (0, 1)),
        ("keiten", 0, (0, 1)),
        ("riichi", 0, (0, 37)),
        ("threat", 0, (0, 1)),
        ("threat", 1, (1, 0)),
    ):
        policy[:] = 0
        policy[preferred] = 2
        assert (
            collection.action_pair(
                features=features,
                head=1,
                legal=legal,
                policy=policy,
                category=category,
                preferred=preferred,
            )
            == expected
        )
    features["self_efficiency"][0] = 2 / 8
    features["chi_discard_legal"][0, 0] = 1
    features["chi_discard_shanten"][0, 0] = 2
    legal[37] = False
    features["kan_legal_mask"][1] = 1
    features["kan_efficiency"][0, 0] = 1 / 8
    for head, preferred, expected in (
        (2, 0, (0, 1)),
        (2, 1, (1, 0)),
        (3, 0, (0, 1)),
        (3, 1, (1, 0)),
    ):
        policy[:] = 0
        policy[preferred] = 2
        assert (
            collection.action_pair(
                features=features,
                head=head,
                legal=legal,
                policy=policy,
                category="calls",
                preferred=preferred,
            )
            == expected
        )
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


def test_seeded_collection_balances_starts_and_retains_all_descendants(
    monkeypatch,
):
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
        values[1, alternatives[1]] = 0.00249
        values[2, alternatives[2]] = 0.0025
        return values

    retained = []
    seeds = []
    collection_steps = self_play.collection_steps

    def continuation(*args, **kwargs):
        if "forced_root" in kwargs:
            seeds.append(kwargs["seed"])
        return collection_steps(*args, **kwargs)

    def discover(capture):
        def visit(collector, batch):
            capture(
                collector,
                {
                    **batch,
                    "sakigiri_pair": numpy.full((len(batch["head"]), 2), -1),
                },
            )

        return self_play.collect(
            predict=predict, games=7, seed=941, root_callback=visit, greedy=True
        )[1]

    monkeypatch.setattr(self_play, "collection_steps", continuation)
    batch, statistics = collection.collect(
        predict=predict,
        root_evaluate=evaluate,
        games=7,
        seed=941,
        retain=lambda rows, source: retained.append((rows, source)),
        discover=discover,
    )
    count = len(batch["actions"])
    assert 0 < count <= 14
    assert (
        statistics["targeted_roots"] == statistics["normal_roots"] == count // 2
    )
    assert statistics["selected_roots"] == count
    assert statistics["root_continuations"] == 2 * count
    assert statistics["disagreement_scored"] == 8
    assert statistics["disagreement_qualifying"] == 6
    assert statistics["selected/disagreement"] == 1
    assert len(numpy.unique(batch["source"][:, 0])) == count // 2
    assert (batch["category"] == -1).sum() == count // 2
    assert (batch["source"][:, 2] != 0).any()
    assert "settlement" not in batch
    assert batch["payment_points"].shape == (
        count,
        2,
        outcomes.PAYMENT_CELLS.sum(),
    )
    targeted = batch["category"] >= 0
    numpy.testing.assert_array_equal(
        batch["source"][targeted, 0], batch["source"][~targeted, 0]
    )
    seeds = numpy.array(seeds).reshape(count, 2)
    assert (seeds[targeted, 0] == seeds[targeted, 1]).all()
    assert (seeds[~targeted, 0] != seeds[~targeted, 1]).all()
    assert (
        batch["actions"][targeted, 0] != batch["actions"][targeted, 1]
    ).all()
    assert numpy.take_along_axis(batch["legal"], batch["actions"], axis=1).all()
    assert "chosen" not in batch
    assert "behavior" not in batch
    assert all(len(value) == count for value in batch["features"].values())
    assert all(
        statistics[f"assigned/{category}"] == 1
        for category in collection.CATEGORIES
    )

    assert len(retained) == 2 * count
    assert sum(source == "normal" for _, source in retained) == count
    assert all(rows["paired_root"].sum() == 1 for rows, _ in retained)
    assert sum(len(rows["chosen"]) for rows, _ in retained) > 2 * count


def test_correction_continuations_sample_descendants_after_the_forced_root():
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
                games=1,
                seed=42,
                collector=root.fork(1, 42, until_hand_end=True),
                forced_root=(root.decision, int(codes[0])),
                greedy=False,
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
    assert (batch["chosen"][~forced] != expected).any()
    # The forced alternative was not the policy's maximum-logit discard.
    assert codes[0] != codes[-1]
