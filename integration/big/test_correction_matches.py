"""Recorded discovery preserves states without treating actions as labels."""

import gzip
import json
import pathlib
import zipfile

import numpy

from integration.big import test_selected_roots
from log_dataset import _native_collection, features, replay
from model import actions
from training import self_play
from training.corrections import matches


def test_recorded_discovery_matches_native_features_and_forks(tmp_path):
    games = []
    original = {}
    logged_events = []
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
        logged_events.append(events)
        path = tmp_path / f"{game}.jsonl.gz"
        with gzip.open(path, "wt") as stream:
            stream.writelines(json.dumps(event) + "\n" for event in events)
        games.append({"path": path, "seed": 941 + game, "duplicate": False})

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

    # Expert archives need no simulator seed and use public replay features.
    archive = tmp_path / "expert.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for game, events in enumerate(logged_events):
            output.writestr(
                f"{game}.mjson",
                "\n".join(json.dumps(event) for event in events),
            )
    expert_games = matches.expert_sources([archive])
    expert_snapshots = []
    expert_seen = set()

    def capture_expert(collector, batch):
        for row, (game, hand, seat, event) in enumerate(batch["indices"]):
            key = game, event, seat, batch["head"][row]
            root = collector.snapshot(row)
            # Corpus replay exposes dora logged after a rinshan draw before
            # the discard decision; seeded matches defer that reveal.
            if key in original:
                assert batch["features"][row].tobytes() == original[key]
            else:
                assert logged_events[game][event - 1]["type"] == "dora"
            pending = root.fork(
                1, 123, shuffle_remaining=False, until_hand_end=True
            ).next()
            assert pending["head"][root.decision] == batch["head"][row]
            numpy.testing.assert_array_equal(
                numpy.frombuffer(pending["legal"], numpy.bool_).reshape(
                    -1, 150
                )[root.decision],
                batch["legal"][row],
            )
            expert_seen.add(key)
            if not expert_snapshots:
                expert_snapshots.append(root)

    statistics = matches.discover(predict, expert_games, capture_expert)
    assert statistics["games"] == statistics["logged_games"] == 2
    assert statistics["expert_games"] == 2
    assert statistics["expert_hands"] > 0
    assert statistics["hands"] == statistics["wins"] == 0
    assert len(expert_seen) > len(original) // 2
    branch = expert_snapshots[0].fork(1, 123, until_hand_end=True)
    test_selected_roots.finish(branch)
    assert branch.finish()[1]["hands"] == 1

    # The same prefix with a different unobserved suffix gives the same root.
    altered = json.loads(json.dumps(logged_events[0]))
    last_draw = max(
        index for index, event in enumerate(altered) if event["type"] == "tsumo"
    )
    altered[last_draw]["pai"] = "C"
    replay = matches.ExpertReplay("altered", altered)
    root = replay.snapshot(0)
    assert root.prefix == expert_snapshots[0].prefix
    assert root.row.tobytes() == expert_snapshots[0].row.tobytes()
    branch = root.fork(1, 123, until_hand_end=True)
    test_selected_roots.finish(branch)
    original_branch = expert_snapshots[0].fork(1, 123, until_hand_end=True)
    test_selected_roots.finish(original_branch)
    assert branch.finish()[0] == original_branch.finish()[0]


def test_expert_rinshan_only_win_is_not_a_discard_correction():
    fixture = json.loads(
        (
            pathlib.Path(__file__).parents[1]
            / "fixtures/model/expert_rinshan.json"
        ).read_text()
    )
    state = replay.Replay("rinshan", fixture["events"])
    for event in fixture["events"]:
        state.apply(event)
    assert (
        actions.phase_for_actions(
            state.legal_observation(fixture["actor"]).legal_actions()
        )
        == "WIN"
    )
    root = _native_collection.NativeRoot.from_expert_prefix(
        json.dumps(fixture["events"]), fixture["actor"], 4, 123
    )
    actual = numpy.frombuffer(root.features, self_play.FEATURE_DTYPE)[0]
    expected = features.pack(state, fixture["actor"], "WIN")
    assert actual["win_kind"] > 0
    for name in expected:
        numpy.testing.assert_array_equal(actual[name], expected[name])
    state.apply(
        {
            "type": "dahai",
            "actor": fixture["actor"],
            "pai": fixture["events"][-1]["pai"],
            "tsumogiri": True,
        }
    )
    assert not state.native_env.is_rinshan_flag
