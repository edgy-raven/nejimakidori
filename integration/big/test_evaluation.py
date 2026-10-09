"""A frozen serving bundle completes duplicate games and resumes exactly."""

import argparse
import json
import pathlib
import shutil

import tensorflow as tf
from model import evaluate, feature_vector, source_snapshot


def test_frozen_bundle_plays_all_seats_and_resumes_without_duplicates(tmp_path):
    root = pathlib.Path(__file__).parents[2]
    bundle = tmp_path / "policy"
    for name in ("model", "log_dataset"):
        source_snapshot.freeze(
            root / name, bundle / "source" / name, include_native=True
        )
    policy = tf.Module()

    @tf.function
    def serve(**features):
        return {
            name: tf.zeros([tf.shape(features["decision_phase"])[0], *shape])
            for name, shape in feature_vector.ACTOR_OUTPUTS.items()
            if name in feature_vector.TRAIN_POLICY_HEADS
        }

    tf.saved_model.save(
        policy,
        str(bundle / "saved_model"),
        signatures={
            "serving_default": serve.get_concrete_function(
                **{
                    name: tf.TensorSpec(
                        [None, *spec["shape"]], spec["dtype"], name=name
                    )
                    for name, spec in feature_vector.INPUTS.items()
                }
            )
        },
    )
    opponent = tmp_path / "opponent"
    shutil.copytree(bundle, opponent)
    schema = opponent / "source/model/feature_vector.py"
    schema.rename(opponent / "source/model/frozen_features.py")
    for path in (opponent / "source").rglob("*.py"):
        source = path.read_text().replace("feature_vector", "frozen_features")
        path.write_text(source)
    # Retained production workers use their model directory as cwd and
    # import dataset helpers stored beside, rather than inside, source/.
    shutil.move(opponent / "source/log_dataset", opponent / "log_dataset")
    worker = opponent / "source/model/frozen_policy.py"
    worker.write_text(
        worker.read_text().replace(
            "from model import actions, features",
            'import pathlib\nassert pathlib.Path("features.py").is_file()\n'
            "from model import actions, features",
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
