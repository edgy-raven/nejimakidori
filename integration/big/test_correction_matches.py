"""Recorded discovery preserves states without treating actions as labels."""

import gzip
import json

import numpy

from integration import reference_policy
from log_dataset import _native_collection, call_states
from training.corrections import collection, matches


def test_recorded_discovery_matches_native_features_and_forks(tmp_path):
    games = []
    original = {}
    for game in range(2):
        collector = _native_collection.NativeCollector(1, 941 + game)
        while (request := collector.next()) is not None:
            features = numpy.frombuffer(
                request["features"], call_states.ROW_DTYPE
            )
            for row, head in enumerate(request["head"]):
                original[
                    game, request["event"][row], request["seat"][row], head
                ] = features[row].tobytes()
            collector.submit(reference_policy.greedy(request))
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

    def capture(collector, batch):
        for row, (game, hand, seat, event) in enumerate(batch["indices"]):
            key = game, event, seat, batch["head"][row]
            expected = numpy.frombuffer(original[key], call_states.ROW_DTYPE)[0]
            for name in call_states.ROW_DTYPE.names:
                numpy.testing.assert_array_equal(
                    batch["features"][name][row], expected[name]
                )
            seen.add(key)
            numpy.testing.assert_allclose(
                numpy.exp(batch["behavior"][row, batch["legal"][row]]),
                1 / batch["legal"][row].sum(),
                rtol=1e-6,
            )
            if not snapshots:
                snapshots.append(collector.snapshot(row))

    statistics = matches.discover(
        reference_policy.uniform_logits, games, capture
    )
    assert seen == original.keys()
    assert statistics["games"] == statistics["logged_games"] == 2
    # Forks leave replay mode and can legally choose a counterfactual move.
    branch = snapshots[0].fork(1, 123, until_hand_end=True)
    request = branch.next()
    choices = reference_policy.greedy(request)
    legal = numpy.frombuffer(request["legal"], numpy.bool_).reshape(-1, 150)
    choices[0] = int(numpy.flatnonzero(legal[0])[0])
    branch.submit(choices)
    reference_policy.finish(branch)
    assert branch.finish()[1]["hands"] == 1


def test_sakigiri_witnesses_select_seeded_match_roots_without_order_targets(
    tmp_path,
):
    # This policy declares riichi, yielding future defensive dilemmas with
    # several equally efficient earlier alternatives on the recorded wall.
    collector = _native_collection.NativeCollector(1, 941)
    while (request := collector.next()) is not None:
        features = numpy.frombuffer(request["features"], call_states.ROW_DTYPE)
        legal = numpy.frombuffer(request["legal"], bool).reshape(-1, 150)
        choices = []
        for row, head in enumerate(request["head"]):
            if head in (0, 1):
                scores = (
                    -100
                    * features["discard_action_all_shanten"][row].astype(float)
                    + features["discard_action_all_ukeire_count"][row]
                )
                offset = 37 if head == 1 else 0
                choices.append(
                    offset
                    + int(
                        numpy.where(
                            legal[row, offset : offset + 37], scores, -1e9
                        ).argmax()
                    )
                )
            else:
                choices.append(1 if head == 4 else 0)
        collector.submit(choices)
    events = json.loads(collector.finish()[0][0])
    path = tmp_path / "match.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.writelines(json.dumps(event) + "\n" for event in events)

    def evaluate(features):
        raise AssertionError("Sakigiri roots do not require Q disagreement")

    retained = []
    discovered = {}

    def discover(capture):
        def inspect(collector, batch):
            for row, actions in enumerate(batch["sakigiri_actions"]):
                if batch["head"][row] == 0 and len(actions):
                    discovered[int(batch["indices"][row, 3])] = actions
            capture(collector, batch)

        return matches.discover(
            predict=reference_policy.uniform_logits,
            games=[{"path": path, "seed": 941, "new_seats": [0]}],
            capture=inspect,
        )

    batch, statistics = collection.collect(
        predict=reference_policy.uniform_logits,
        root_evaluate=evaluate,
        games=1,
        seed=5,
        retain=lambda rows, source: retained.append((rows, source)),
        reference_policy=reference_policy.Policy(),
        discover=discover,
    )
    assert statistics["eligible/sakigiri"] == 116
    numpy.testing.assert_array_equal(discovered[7], [4, 12, 34, 32])
    assert statistics["selected/sakigiri"] == 1
    assert statistics["targeted_roots"] == statistics["normal_roots"] == 1
    targeted = numpy.flatnonzero(batch["category"] == 5)[0]
    witness = discovered[int(batch["source"][targeted, 3])]
    assert batch["actions"][targeted, 0] in witness[:-1]
    assert batch["actions"][targeted, 1] == witness[-1]
    assert len(retained) == 4
    assert sum(len(rows["chosen"]) for rows, _ in retained) > 4
