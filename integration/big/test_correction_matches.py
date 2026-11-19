"""Recorded discovery preserves states without treating actions as labels."""

import gzip
import json

import numpy

from integration.big import test_selected_roots
from log_dataset import _native_collection
from model import actions
from training import self_play
from training.corrections import collection, matches


def test_recorded_discovery_matches_native_features_and_forks(tmp_path):
    games = []
    original = {}
    for game in range(2):
        collector = _native_collection.NativeCollector(1, 941 + game)
        while (request := collector.next()) is not None:
            features = numpy.frombuffer(
                request["features"], self_play.FEATURE_DTYPE
            )
            for row, head in enumerate(request["head"]):
                original[
                    game, request["event"][row], request["seat"][row], head
                ] = features[row].tobytes()
            collector.submit(test_selected_roots.greedy(request))
        events = json.loads(collector.finish()[0][0])
        path = tmp_path / f"{game}.jsonl.gz"
        with gzip.open(path, "wt") as stream:
            stream.writelines(json.dumps(event) + "\n" for event in events)
        games.append(
            {
                "path": path,
                "seed": 941 + game,
                "new_seats": [0],
            }
        )

    snapshots = []
    seen = set()

    def predict(features):
        return {
            name: numpy.zeros(
                (len(features), actions.ACTOR_OUTPUTS[name][0]), numpy.float32
            )
            for name in actions.TRAIN_POLICY_HEADS
        }

    def capture(collector, batch):
        for row, (game, hand, seat, event) in enumerate(batch["indices"]):
            key = game, event, seat, batch["head"][row]
            assert batch["features"][row].tobytes() == original[key]
            seen.add(key)
            numpy.testing.assert_allclose(
                numpy.exp(batch["behavior"][row, batch["legal"][row]]),
                1 / batch["legal"][row].sum(),
                rtol=1e-6,
            )
            if not snapshots:
                snapshots.append(collector.snapshot(row))

    statistics = matches.discover(predict, games, capture)
    assert seen == original.keys()
    assert statistics["games"] == statistics["logged_games"] == 2
    # Forks leave replay mode and can legally choose a counterfactual move.
    branch = snapshots[0].fork(1, 123, until_hand_end=True)
    request = branch.next()
    choices = test_selected_roots.greedy(request)
    legal = numpy.frombuffer(request["legal"], numpy.bool_).reshape(-1, 150)
    choices[0] = int(numpy.flatnonzero(legal[0])[0])
    branch.submit(choices)
    test_selected_roots.finish(branch)
    assert branch.finish()[1]["hands"] == 1


def test_sakigiri_witnesses_select_seeded_match_roots_without_order_targets(
    tmp_path,
):
    # Closed-hand play on this wall supplies two verified order witnesses.
    collector = _native_collection.NativeCollector(1, 946)
    while (request := collector.next()) is not None:
        features = numpy.frombuffer(
            request["features"], self_play.FEATURE_DTYPE
        )
        legal = numpy.frombuffer(request["legal"], bool).reshape(-1, 150)
        choices = []
        for row, head in enumerate(request["head"]):
            if head in (0, 1):
                scores = (
                    -100
                    * features["discard_action_all_shanten"][row].astype(float)
                    + features["discard_action_all_ukeire_count"][row]
                )
                choices.append(
                    int(numpy.where(legal[row, :37], scores, -1e9).argmax())
                )
            else:
                choices.append(1 if head == 4 else 0)
        collector.submit(choices)
    events = json.loads(collector.finish()[0][0])
    path = tmp_path / "match.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.writelines(json.dumps(event) + "\n" for event in events)

    def predict(features):
        return {
            name: numpy.zeros(
                (len(features), actions.ACTOR_OUTPUTS[name][0]), numpy.float32
            )
            for name in actions.TRAIN_POLICY_HEADS
        }

    def evaluate(features):
        raise AssertionError("Sakigiri roots do not require Q disagreement")

    retained = []
    batch, statistics = collection.collect(
        predict=predict,
        root_evaluate=evaluate,
        games=1,
        seed=6,
        retain=lambda rows, source: retained.append((rows, source)),
        discover=lambda capture: matches.discover(
            predict=predict,
            games=[
                {
                    "path": path,
                    "seed": 946,
                    "new_seats": [0],
                }
            ],
            capture=capture,
        ),
    )
    assert statistics["eligible/sakigiri"] == 2
    assert statistics["selected/sakigiri"] == 1
    assert statistics["targeted_roots"] == statistics["normal_roots"] == 1
    targeted = numpy.flatnonzero(batch["category"] == 6)[0]
    assert batch["source"][targeted, 2] == 2
    assert batch["source"][targeted, 3] in (781, 789)
    numpy.testing.assert_array_equal(batch["actions"][targeted], [12, 4])
    assert len(retained) == 4
    assert sum(len(rows["chosen"]) for rows, _ in retained) > 4
