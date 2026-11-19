"""Persistent synchronized learner replicas with one global row budget."""

import json
import os
import pathlib
import subprocess
import time

import tensorflow as tf

from training import monitor


def publish_stage(args, group, root, stage):
    """Release workers after generated files reach both host-local roots."""
    path = root / f"{stage}-ready.json"
    if group.chief:
        pending = path.with_suffix(".pending.json")
        monitor.write_json(pending, {"stopped": (root / "STOP").exists()})
        if args.sync_command:
            command = json.loads(pathlib.Path(args.sync_command).read_text())
            subprocess.run(
                [
                    value.format(root=str(root), stage=stage)
                    for value in command
                ],
                check=True,
            )
        pending.replace(path)
    else:
        while not path.exists():
            time.sleep(1)
    group.barrier()
    return json.loads(path.read_text())["stopped"]


class LearnerGroup:
    """One process per GPU; chief owns collection and completed artifacts."""

    def __init__(self):
        self.chief = True
        if "TF_CONFIG" in os.environ:
            self.chief = (
                json.loads(os.environ["TF_CONFIG"])["task"]["index"] == 0
            )
            self.strategy = tf.distribute.MultiWorkerMirroredStrategy(
                communication_options=(
                    tf.distribute.experimental.CommunicationOptions(
                        implementation=(
                            tf.distribute.experimental.CommunicationImplementation.NCCL
                        )
                    )
                )
            )
        else:
            self.strategy = tf.distribute.get_strategy()

        @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
        def synchronize(value):
            values = self.strategy.run(
                lambda: tf.distribute.get_replica_context().all_reduce(
                    tf.distribute.ReduceOp.SUM, tf.identity(value)
                )
            )
            return self.strategy.experimental_local_results(values)[0]

        self._synchronize = synchronize

    @property
    def replicas(self):
        return self.strategy.num_replicas_in_sync

    def barrier(self):
        self._synchronize(tf.constant(0.0)).numpy()

    def stopped(self, root):
        return bool(
            self._synchronize(
                tf.constant(float(self.chief and (root / "STOP").exists()))
            ).numpy()
        )

    def bind(self, learner, critic_learner, batch_size):
        self.local_batch = (batch_size + self.replicas - 1) // self.replicas
        self.packing = critic_learner.feature_packing
        self._pack = tf.function(
            self.packing.pack,
            input_signature=[
                tf.nest.map_structure(
                    lambda spec: tf.TensorSpec((None, *spec.shape), spec.dtype),
                    self.packing.structure,
                )
            ],
        )
        self._updates = {
            "critic": self._make_update(
                critic_learner.critic_update.python_function, actor=False
            ),
            "actor": self._make_update(
                learner.update.python_function, actor=True
            ),
        }

    def _make_update(self, operation, actor):
        def step(batch, count):
            if actor:
                batch = {
                    **batch,
                    "features": self.packing.unpack(batch["features"]),
                }
            return operation(batch, count)

        @tf.function
        def update(batch, count):
            metrics = self.strategy.run(step, args=(batch, count))
            names = tuple(metrics)
            packed = self.strategy.run(
                lambda values: tf.stack([values[name] for name in names]),
                args=(metrics,),
            )
            values = self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, packed, axis=None
            )
            return dict(zip(names, tf.unstack(values)))

        return update

    def update(self, update, batch):
        with tf.device("/CPU:0"):
            rows = {
                "features": self._pack(batch["features"]),
                **{
                    name: tf.convert_to_tensor(batch[name])
                    for name in ("head", "legal", "chosen", "payment_points")
                },
            }
            count = tf.shape(rows["chosen"])[0]

            def shard(ctx):
                begin = (
                    tf.cast(ctx.replica_id_in_sync_group, tf.int32)
                    * self.local_batch
                )
                indices = tf.minimum(
                    begin + tf.range(self.local_batch), count - 1
                )
                return (
                    tf.nest.map_structure(
                        lambda value: tf.gather(value, indices), rows
                    ),
                    tf.clip_by_value(count - begin, 0, self.local_batch),
                )

            rows, counts = (
                self.strategy.experimental_distribute_values_from_function(
                    shard
                )
            )
        return self._updates[update](rows, counts)
