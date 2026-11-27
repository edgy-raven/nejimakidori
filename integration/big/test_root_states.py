"""Prepared human simulator states replace online archive reconstruction."""

import json
import pathlib
import types
import zipfile

import numpy

from log_dataset import (
    _native_collection,
    afk,
    call_states,
    records,
    root_states,
)


def test_prepared_states_preserve_features_and_need_no_source_archive(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    expected = {}
    games = []
    archive_path = tmp_path / "human.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in ("sakigiri_calls", "recorded_nonred_pon"):
            game_id = next(
                f"{name}-{index}"
                for index in range(100)
                if records.game_partition(f"{name}-{index}") < 7
            )
            events = json.loads(
                pathlib.Path(
                    "integration/fixtures/model", name + ".json"
                ).read_text()
            )
            events[0]["player_ranks"] = {"retained_seats": [0, 1, 2, 3]}
            archive.writestr(
                game_id + ".mjson",
                "\n".join(json.dumps(event) for event in events),
            )
            games.append(
                {
                    "game_id": game_id,
                    "archive": str(archive_path),
                    "member": game_id + ".mjson",
                }
            )
            for hand in afk.retained_hands(events):
                kyoku_id = (
                    f"{hand[0]['bakaze']}{hand[0]['kyoku']}-"
                    f"{hand[0]['honba']}"
                )
                expected[game_id, kyoku_id] = root_states.prepare_hand(
                    game_id, hand
                )
    (source / "games.jsonl").write_text(
        "\n".join(json.dumps(game) for game in games)
    )
    (source / "dataset_summary.json").write_text("{}\n")
    args = types.SimpleNamespace(
        data=source, output=tmp_path / "states", workers=1, limit_games=None
    )
    root_states.run(args)
    archive_path.unlink()
    # Completed shards can be resumed without opening original archives.
    root_states.run(args)
    store = root_states.Store(args.output)
    witnesses = 0
    for (game_id, kyoku_id), prepared in expected.items():
        assert set(store.hand(game_id, kyoku_id)) == set(prepared)
        for (decision_index, actor, phase), entry in prepared.items():
            identity = {
                "game_id": game_id,
                "kyoku_id": kyoku_id,
                "decision_index": decision_index,
                "actor": actor,
                "phase": phase,
            }
            root = store.root(**identity)
            original = _native_collection.NativeRoot.loads(
                entry["root"].encode()
            )
            assert root.decision == original.decision
            actual = call_states.from_request(root.fork(1, 941, False).next())
            expected_features = call_states.from_request(
                original.fork(1, 941, False).next()
            )
            for name in expected_features:
                numpy.testing.assert_array_equal(
                    actual[name], expected_features[name]
                )
            assert store.witness(**identity) == (
                None
                if entry["witness"] is None
                else json.loads(json.dumps(entry["witness"]))
            )
            witnesses += entry["witness"] is not None
    assert witnesses > 0
    store.close()
