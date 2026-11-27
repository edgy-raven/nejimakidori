"""Host partitions form one live, complete seat-rotated evaluation."""

import gzip
import json
import types

import pytest

from training import evaluate


def test_distributed_match_publishes_progress_and_verified_replays(tmp_path):
    parts = [tmp_path / "local", tmp_path / "remote"]
    output = tmp_path / "merged"
    events = []
    writer = types.SimpleNamespace(add_event=events.append, flush=lambda: None)
    rows = []
    for seed, part in zip((100, 101), parts, strict=True):
        (part / "replays").mkdir(parents=True)
        (part / "evaluation.json").write_text(
            json.dumps(
                {
                    "format": "riichienv_standard_south",
                    "seed": seed,
                    "games": 4,
                    "frozen_policy_randomness": "wall_actor_event_phase",
                    "saved_models": {"new": {"saved_model.pb": "same-policy"}},
                    "mortal": {"checkpoint": "same-opponent"},
                }
            )
        )
        for seat in range(4):
            row = {
                "format": "riichienv_standard_south",
                "seed": seed,
                "new_seats": [seat],
                "ranks": [1, 2, 3, 4],
                "scores": [40000, 30000, 20000, 10000],
                "replay": f"replays/{seed}-{seat}.jsonl.gz",
            }
            rows.append(row)
            with gzip.open(part / row["replay"], "wt") as stream:
                stream.write('{"type": "end_game"}\n')
    (parts[0] / "games.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:4])
    )
    (parts[1] / "games.jsonl").write_text('{"unfinished":')
    assert (
        evaluate.merge_parts(
            parts=parts, output=output, seed=100, games=8, writer=writer
        )
        == 4
    )
    assert json.loads((output / "summary.json").read_text())["seed_groups"] == 1
    with pytest.raises(ValueError, match="missing requested"):
        evaluate.merge_parts(
            parts=parts,
            output=output,
            seed=100,
            games=8,
            writer=writer,
            complete=True,
        )
    (parts[1] / "games.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows[4:])
    )
    assert (
        evaluate.merge_parts(
            parts=parts,
            output=output,
            seed=100,
            games=8,
            writer=writer,
            complete=True,
        )
        == 8
    )
    assert json.loads((output / "summary.json").read_text()) == (
        evaluate.summarize(rows, 100)
    )
    assert len(list((output / "replays").iterdir())) == 8
    assert len((output / "games.jsonl").read_text().splitlines()) == 8
    assert events[-1].step == 8
    manifest = json.loads((parts[1] / "evaluation.json").read_text())
    manifest["mortal"]["checkpoint"] = "different-opponent"
    (parts[1] / "evaluation.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different policies"):
        evaluate.merge_parts(
            parts=parts, output=output, seed=100, games=8, writer=writer
        )
