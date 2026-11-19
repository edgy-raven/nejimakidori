"""A frozen serving bundle completes duplicate games and resumes exactly."""

import argparse
import gzip
import json
import os
import pathlib
import shutil
import types

import numpy
import pytest
import riichienv
import tensorboard.backend.event_processing.event_accumulator as events
import tensorflow as tf

from log_dataset import _native_collection
from model import actions
from training import evaluate, frozen_policy, mortal_policy


@pytest.mark.parametrize("opponent_kind", ["savedmodel", "mortal"])
def test_frozen_bundle_plays_all_seats_and_resumes_without_duplicates(
    tmp_path, opponent_kind
):
    if opponent_kind == "mortal" and "MORTAL_SOURCE" not in os.environ:
        pytest.skip(
            "Set MORTAL_SOURCE and MORTAL_CHECKPOINT for Mortal policy games"
        )
    root = pathlib.Path(__file__).parents[2]
    bundle = tmp_path / "policy"
    for name in ("model", "training", "log_dataset"):
        frozen_policy.freeze_source(root / name, bundle / "source" / name)
    policy = tf.Module()

    @tf.function
    def serve(**features):
        return {
            name: tf.zeros([tf.shape(features["decision_phase"])[0], *shape])
            for name, shape in actions.ACTOR_OUTPUTS.items()
            if name in actions.TRAIN_POLICY_HEADS
        }

    tf.saved_model.save(
        policy,
        str(bundle / "saved_model"),
        signatures={
            "policy": serve.get_concrete_function(
                **{
                    name: tf.TensorSpec(
                        [None, *spec["shape"]], spec["dtype"], name=name
                    )
                    for name, spec in _native_collection.INPUTS.items()
                }
            )
        },
    )
    opponent = tmp_path / "opponent"
    shutil.copytree(bundle, opponent)
    for path in (opponent / "source").rglob("*.py"):
        source = path.read_text().replace(
            "native_tile_code", "frozen_tile_code"
        )
        path.write_text(source)
    # Retained production workers use their model directory as cwd and
    # import dataset helpers stored beside, rather than inside, source/.
    shutil.move(opponent / "source/log_dataset", opponent / "log_dataset")
    shutil.move(
        opponent / "source/training/frozen_policy.py",
        opponent / "source/model/frozen_policy.py",
    )
    args = argparse.Namespace(
        new=str(bundle / "saved_model"),
        old=str(opponent / "saved_model"),
        output=str(tmp_path / "games"),
        seed=14,
        games=4,
        workers=1,
        devices="-1",
        batch_games=4,
        mortal_source=None,
        tensorboard=None,
    )
    if opponent_kind == "mortal":
        args.old = os.environ["MORTAL_CHECKPOINT"]
        args.mortal_source = os.environ["MORTAL_SOURCE"]
        args.batch_games = 2
    evaluate.run(args)
    result = tmp_path / "games/games.jsonl"
    before = result.read_bytes()
    rows = [json.loads(line) for line in before.splitlines()]
    assert len(rows) == 4
    assert {tuple(row["new_seats"]) for row in rows} == {(0,), (1,), (2,), (3,)}
    assert all(row["seed"] == 14 for row in rows)
    summary = json.loads((tmp_path / "games/summary.json").read_text())
    manifest = json.loads((tmp_path / "games/evaluation.json").read_text())
    if opponent_kind == "mortal":
        assert "adapter/training/mortal_policy.py" in manifest["mortal"]
    assert summary["compared_games"] == 4
    assert summary["seed_groups"] == 1
    assert summary["rank_point_offsets"] == [125, 60, -5, -255]
    new_points, old_points = [], []
    for row in rows:
        points = numpy.ceil(
            (numpy.array(row["scores"]) - 25000) / 1000
            + numpy.array([125, 60, -5, -255])[numpy.array(row["ranks"]) - 1]
        )
        numpy.testing.assert_array_equal(row["rank_points"], points)
        new_points.extend(points[row["new_seats"]])
        old_points.extend(
            [
                value
                for seat, value in enumerate(points)
                if seat not in row["new_seats"]
            ]
        )
    expected_st3 = numpy.mean(new_points) - numpy.mean(old_points)
    assert summary["new_minus_old"]["rank_points"] == pytest.approx(
        expected_st3
    )
    assert summary["format"] == (
        "riichienv_standard_south"
        if opponent_kind == "mortal"
        else "riichienv_duplicate_south"
    )
    assert evaluate.summarize(rows[:3]) is None
    evaluate.run(args)
    assert result.read_bytes() == before
    summary_path = pathlib.Path(args.output) / "summary.json"
    published_at = summary_path.stat().st_mtime_ns
    for name in ("new", "old") if opponent_kind == "savedmodel" else ("new",):
        original = getattr(args, name)
        changed = tmp_path / (name + "-changed")
        shutil.copytree(original, changed)
        with (changed / "saved_model.pb").open("ab") as stream:
            stream.write(b"changed")
        setattr(args, name, str(changed))
        with pytest.raises(ValueError, match="Evaluation settings changed"):
            evaluate.run(args)
        assert summary_path.stat().st_mtime_ns == published_at
        assert result.read_bytes() == before
        setattr(args, name, original)
    history = events.EventAccumulator(str(tmp_path / "games/tensorboard"))
    history.Reload()
    assert history.Scalars("progress/completed_games")[-1].value == 4
    assert history.Scalars("progress/percent")[-1].value == 100
    assert history.Scalars("difference/rank_points")[-1].step == 4
    assert history.Scalars("difference/rank_points")[-1].value == pytest.approx(
        expected_st3
    )
    if opponent_kind == "savedmodel":
        replays = {
            row["replay"]: gzip.decompress(
                (tmp_path / "games" / row["replay"]).read_bytes()
            )
            for row in rows
        }
        assert len(set(replays.values())) == 1
        assert (
            json.loads((tmp_path / "games/evaluation.json").read_text())[
                "frozen_policy_randomness"
            ]
            == "wall_actor_event_phase"
        )
        args.output = str(tmp_path / "other-workers")
        args.workers = 2
        args.batch_games = 1
        evaluate.run(args)
        assert {
            name: gzip.decompress(
                (pathlib.Path(args.output) / name).read_bytes()
            )
            for name in replays
        } == replays
    if opponent_kind == "mortal":
        fixture = json.loads(
            (
                root / "integration/fixtures/model/" "mortal_delayed_dora.json"
            ).read_text()
        )
        observation = riichienv.Observation.deserialize_from_base64(
            fixture["observation"]
        )
        worker = mortal_policy.MortalPolicy(tmp_path / "games/old", capacity=1)
        try:
            worker.process.stdin.write(
                json.dumps(
                    {
                        "reset": True,
                        "items": [
                            {
                                "stream": 0,
                                "seat": fixture["seat"],
                                "events": [
                                    json.dumps(event)
                                    for event in fixture["events"]
                                ],
                            }
                        ],
                    }
                )
                + "\n"
            )
            worker.process.stdin.flush()
            action = worker.receive_actions(
                [
                    (
                        0,
                        types.SimpleNamespace(mjai_log=fixture["events"]),
                        observation,
                    )
                ]
            )[0]
            assert action.action_type == riichienv.ActionType.RON
        finally:
            worker.close()
        changed = tmp_path / "changed.zip"
        changed.write_bytes(pathlib.Path(args.old).read_bytes() + b"changed")
        args.old = str(changed)
        with pytest.raises(ValueError, match="Evaluation settings changed"):
            evaluate.run(args)
