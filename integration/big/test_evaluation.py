"""A frozen serving bundle completes duplicate games and resumes exactly."""

import argparse
import gzip
import json
import pathlib
import shutil

import numpy
import pytest
import tensorboard.backend.event_processing.event_accumulator as events
import tensorflow as tf

from log_dataset import _native_collection
from model import actions
from training import evaluate, frozen_policy


def test_frozen_bundle_plays_all_seats_and_resumes_without_duplicates(tmp_path):
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
    # The opponent owns its package layout and feature implementation.
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
    evaluate.run(args)
    result = tmp_path / "games/games.jsonl"
    before = result.read_bytes()
    rows = [json.loads(line) for line in before.splitlines()]
    assert len(rows) == 4
    assert {tuple(row["new_seats"]) for row in rows} == {(0,), (1,), (2,), (3,)}
    assert all(row["seed"] == 14 for row in rows)
    summary = json.loads((tmp_path / "games/summary.json").read_text())
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
    assert summary["format"] == "riichienv_duplicate_south"
    assert evaluate.summarize(rows[:3]) is None
    evaluate.run(args)
    assert result.read_bytes() == before
    summary_path = pathlib.Path(args.output) / "summary.json"
    published_at = summary_path.stat().st_mtime_ns
    for name in ("new", "old"):
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
        name: gzip.decompress((pathlib.Path(args.output) / name).read_bytes())
        for name in replays
    } == replays
