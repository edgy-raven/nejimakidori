"""A frozen serving bundle completes duplicate games and resumes exactly."""

import argparse
import json
import pathlib
import shutil

import tensorflow as tf

from log_dataset import _native_collection
from model import actions, evaluate, frozen_policy


def test_frozen_bundle_plays_all_seats_and_resumes_without_duplicates(tmp_path):
    root = pathlib.Path(__file__).parents[2]
    bundle = tmp_path / "policy"
    for name in ("model", "log_dataset"):
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
    assert evaluate.summarize(rows[:3]) is None
    evaluate.run(args)
    assert result.read_bytes() == before
