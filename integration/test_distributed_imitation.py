"""Masked imitation has the same gradient across replica layouts."""

import os
import pathlib
import subprocess
import sys


def test_imitation_keeps_its_weight_with_uneven_replica_labels():
    # Labeled rows can all land on the last replica. Their policy
    # gradient must match a single-device mean over the same valid labels.
    subprocess.run(
        args=[
            sys.executable,
            "-c",
            """
import tensorflow as tf
cpu = tf.config.list_physical_devices("CPU")[0]
tf.config.set_logical_device_configuration(cpu, [
    tf.config.LogicalDeviceConfiguration(),
    tf.config.LogicalDeviceConfiguration(),
])
import feature_vector
from model import objectives
for replicas in (1, 2):
    strategy = tf.distribute.MirroredStrategy(
        devices=[f"/cpu:{index}" for index in range(replicas)])
    for packed in (False, True):
        with strategy.scope():
            value = tf.Variable(0.0)

        def update():
            active = (
                tf.distribute.get_replica_context().replica_id_in_sync_group
                == replicas - 1)
            labels = {name: tf.constant([-1]) for name in feature_vector.TRAIN_POLICY_HEADS}
            labels["discard_policy_logits"] = tf.reshape(
                tf.where(active, 0, -1), (1,))
            with tf.GradientTape(persistent=True) as tape:
                outputs = {
                    name: tf.reshape(tf.stack([value, 0.0]), (1, 2))
                    for name in feature_vector.TRAIN_POLICY_HEADS
                }
                def terms(mean):
                    return (
                        objectives.policy_loss(labels, outputs, mean=mean),
                        objectives.awr_policy_terms(
                            objectives.policy_nll(labels, outputs)[0],
                            tf.ones((1,)), tf.reshape(active, (1,)),
                            mean=mean)["loss"],
                    )
                if packed:
                    means = objectives.MaskedMeans()
                    terms(means.collect)
                    means.reduce()
                    imitation, loss = terms(means.read)
                else:
                    imitation, loss = terms(objectives.mean_valid)
                scaled_imitation = imitation / replicas
                scaled = loss / replicas
            return (loss, tape.gradient(scaled, value),
                    tape.gradient(scaled_imitation, value))

        losses, gradients, imitation_gradients = strategy.run(update)
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, losses, None)
        gradient = strategy.reduce(tf.distribute.ReduceOp.SUM, gradients, None)
        tf.debugging.assert_near(loss, tf.math.log(2.0))
        tf.debugging.assert_near(gradient, -0.5)
        tf.debugging.assert_near(gradient, strategy.reduce(
            tf.distribute.ReduceOp.SUM, imitation_gradients, None))
""",
        ],
        cwd=pathlib.Path(__file__).parents[1],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
        timeout=60,
    )


def test_training_handover_keeps_metrics_local_to_each_replica(tmp_path):
    # Exercise the actual trainer, including optimizer synchronization and
    # SUM reductions of accuracy counters, on two logical CPU replicas.
    subprocess.run(
        args=[
            sys.executable,
            "-c",
            """
import csv
import json
import pathlib
import sys
import tensorflow as tf
cpu = tf.config.list_physical_devices("CPU")[0]
tf.config.set_logical_device_configuration(cpu, [
    tf.config.LogicalDeviceConfiguration(),
    tf.config.LogicalDeviceConfiguration(),
])
from log_dataset import rebuild
import feature_vector
from model import records, replay, train
root = pathlib.Path(sys.argv[1])
data = root / "data"
data.mkdir()
events = json.loads(pathlib.Path(
    "integration/fixtures/model/pass_speed_round.json").read_text())
count = 0
with tf.io.TFRecordWriter(
    str(data / "train-00000.tfrecord.gz"), options="GZIP"
) as writer:
    for observation in replay.Replay("replica-test", events).observations():
        writer.write(rebuild.serialize_observation(observation, True))
        count += 1
(data / "dataset_summary.json").write_text(json.dumps(
    {**records.DATA_CONTRACT, "examples": count, "focused_examples": count}))
strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0", "/cpu:1"])
tf.distribute.MirroredStrategy = lambda: strategy
train.run(data=data, output=root / "candidate", experiment=True)
for phase in ("natural_start", "focused", "natural_finish"):
    with (root / "candidate" / (phase + ".csv")).open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row["imitation_count"]) == 4, row
    assert 0 <= float(row["imitation_correct"]) <= 4, row
assert (root / "candidate/model.keras").is_file()
assert (root / "candidate/saved_model/saved_model.pb").is_file()
""",
            str(tmp_path),
        ],
        cwd=pathlib.Path(__file__).parents[1],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_NUM_INTRAOP_THREADS": "2",
            "TF_NUM_INTEROP_THREADS": "2",
        },
        check=True,
        timeout=180,
    )
