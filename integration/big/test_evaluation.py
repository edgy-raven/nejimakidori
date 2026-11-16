"""A frozen serving bundle completes duplicate games and resumes exactly."""

import argparse
import json
import os
import pathlib
import shutil
import types

import pytest
import tensorboard.backend.event_processing.event_accumulator as events
import tensorflow as tf
from log_dataset import _native_collection
from model import actions
from training import distill_policy, evaluate, frozen_policy

import riichienv


@pytest.mark.parametrize("opponent_kind", ["savedmodel", "distill"])
def test_frozen_bundle_plays_all_seats_and_resumes_without_duplicates(
    tmp_path, opponent_kind
):
    if opponent_kind == "distill" and "DISTILL_SOURCE" not in os.environ:
        pytest.skip(
            "Set DISTILL_SOURCE and DISTILL_CHECKPOINT for distill policy games"
        )
    root = pathlib.Path(__file__).parents[2]
    bundle = tmp_path / "policy"
    for name in ("model", "training", "log_dataset"):
        frozen_policy.freeze_source(
            root / name, bundle / "source" / name, include_native=True
        )
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
    worker = opponent / "source/model/frozen_policy.py"
    shutil.move(opponent / "source/training/frozen_policy.py", worker)
    worker.write_text(
        worker.read_text()
        .replace(
            "request = json.loads(line)",
            'request = json.loads(line)\n    assert request["batch_size"] == 1',
        )
        .replace(
            "from log_dataset import features",
            'import pathlib\nassert pathlib.Path("frozen_policy.py").is_file()\n'
            "from log_dataset import features",
        )
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
        distill_source=None,
        tensorboard=None,
    )
    if opponent_kind == "distill":
        args.old = os.environ["DISTILL_CHECKPOINT"]
        args.distill_source = os.environ["DISTILL_SOURCE"]
        args.batch_games = 2
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
    assert summary["format"] == (
        "riichienv_standard_south"
        if opponent_kind == "distill"
        else "riichienv_duplicate_south"
    )
    assert evaluate.summarize(rows[:3]) is None
    evaluate.run(args)
    assert result.read_bytes() == before
    history = events.EventAccumulator(str(tmp_path / "games/tensorboard"))
    history.Reload()
    assert history.Scalars("progress/completed_games")[-1].value == 4
    assert history.Scalars("progress/percent")[-1].value == 100
    assert history.Scalars("difference/rank_points")[-1].step == 4
    if opponent_kind == "distill":
        fixture = json.loads(
            (
                root / "integration/fixtures/model/" "distill_delayed_dora.json"
            ).read_text()
        )
        observation = riichienv.Observation.deserialize_from_base64(
            fixture["observation"]
        )
        worker = distill_policy.DistillPolicy(
            tmp_path / "games/old", capacity=1
        )
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
