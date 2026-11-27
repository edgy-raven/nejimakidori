"""Persistent synchronized learner replicas with one global row budget."""

import json
import os
import pathlib
import subprocess
import time

import tensorflow as tf

from log_dataset import call_states
from training import match_records, monitor, restoration


def publish_stage(args, group, root, stage):
    """Release workers after generated files reach their output roots."""
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
        self.rank = 0
        if "TF_CONFIG" in os.environ:
            self.rank = json.loads(os.environ["TF_CONFIG"])["task"]["index"]
            self.chief = self.rank == 0
            self.strategy = tf.distribute.MultiWorkerMirroredStrategy(
                communication_options=(
                    tf.distribute.experimental.CommunicationOptions(
                        implementation="NCCL"
                    )
                )
            )
        else:
            self.strategy = tf.distribute.get_strategy()

        @tf.function
        def synchronize(value):
            values = self.strategy.run(
                lambda: tf.distribute.get_replica_context().all_reduce(
                    tf.distribute.ReduceOp.SUM, tf.cast(value, tf.float64)
                )
            )
            # GPU collectives require floating-point inputs. Float64 keeps
            # the integer stop flags and row counts exact during reduction.
            return tf.cast(
                self.strategy.experimental_local_results(values)[0], tf.int32
            )

        self.sum_counts = synchronize

        @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
        def maximum(value):
            values = self.strategy.run(
                lambda: tf.reduce_max(
                    tf.distribute.get_replica_context().all_gather(
                        tf.cast(value[None], tf.float32), axis=0
                    )
                )
            )
            return tf.cast(
                self.strategy.experimental_local_results(values)[0], tf.int32
            )

        self.maximum = maximum

    @property
    def replicas(self):
        return self.strategy.num_replicas_in_sync

    def barrier(self):
        self.sum_counts(tf.zeros(2, tf.int32)).numpy()

    def configure(self, packing, batch_size):
        self.set_batch_size(batch_size)
        self.packing = packing
        self._pack = tf.function(
            lambda features: call_states.pack(features, self.packing),
            input_signature=[call_states.input_specs()],
        )

    def bind(self, learner, critic_learner, batch_size):
        self.configure(critic_learner.feature_packing, batch_size)
        self._updates = {
            "critic": self._make_update(
                critic_learner.critic_update.python_function, actor=False
            ),
            "actor": self._make_update(
                learner.update.python_function, actor=True
            ),
        }

    def bind_expert(self, learner, rollout_actor, batch_size):
        self.configure(learner.feature_packing, batch_size)
        self._updates = {
            "imitation": self._make_update(
                learner.update.python_function, actor=True
            ),
            "imitation-rollout": self._make_update(
                learner.critic_update.python_function, actor=False
            ),
            "imitation-mmd": self._make_update(
                rollout_actor.update.python_function, actor=True
            ),
        }

    def set_batch_size(self, batch_size):
        self.local_batch = (batch_size + self.replicas - 1) // self.replicas

    def _make_update(self, operation, actor):
        def step(batch, valid):
            if actor:
                batch = {
                    **batch,
                    "features": call_states.unpack(
                        batch["features"], self.packing
                    ),
                }
            return operation(batch, valid)

        @tf.function(reduce_retracing=True)
        def update(batch, valid):
            metrics = self.strategy.run(step, args=(batch, valid))
            names = tuple(metrics)
            packed = self.strategy.run(
                lambda values: tf.stack([values[name] for name in names]),
                args=(metrics,),
            )
            # Trainers already reduce masked statistics and gradients globally.
            # Re-reducing identical metrics adds another collective per update.
            values = self.strategy.experimental_local_results(packed)[0]
            return dict(zip(names, tf.unstack(values)))

        return update

    def _indices(self, count):
        begin = self.rank * self.local_batch
        return (
            tf.minimum(begin + tf.range(self.local_batch), count - 1),
            tf.clip_by_value(count - begin, 0, self.local_batch),
        )

    def decode_records(self, raw):
        """Decode only this rank's rows from the unchanged global raw batch."""
        count = tf.shape(raw)[0]
        indices, local_count = self._indices(count)
        return {
            **match_records.decode_records(tf.gather(raw, indices)),
            "local_count": local_count,
            "global_count": count,
        }

    def localize(self, batch):
        """Partition packed root rows and remap their sparse call owners."""
        with tf.device("/CPU:0"):
            count = tf.shape(batch["chosen"])[0]
            indices, local_count = self._indices(count)
            return {
                "features": call_states.tensor_packed_select(
                    batch["features"], indices
                ),
                **{
                    name: tf.nest.map_structure(
                        lambda value: tf.gather(value, indices), values
                    )
                    for name, values in batch.items()
                    if name != "features"
                },
                "local_count": local_count,
                "global_count": count,
            }

    def update(self, update, batch, root):
        """Poll STOP and count eligible rows in one collective per batch."""
        with tf.device("/CPU:0"):
            valid = restoration.trainable_decisions(
                batch["features"], batch["head"], batch["legal"]
            ) & (tf.range(tf.size(batch["head"])) < batch["local_count"])
            control = tf.stack(
                [
                    int(self.chief and (root / "STOP").exists()),
                    tf.reduce_sum(tf.cast(valid, tf.int32)),
                ]
            )
        if self.replicas > 1:
            control = self.sum_counts(control)
        stopped, trained = control.numpy()
        if stopped:
            return None
        if not trained:
            # Skip both Adam moments and decoupled weight decay. Every rank
            # takes this branch together, including ranks with only padding.
            return {"trained_positions": 0}
        with tf.device("/CPU:0"):
            rows = {
                "features": self._pack(batch["features"]),
                **{
                    name: tf.nest.map_structure(tf.convert_to_tensor, values)
                    for name, values in batch.items()
                    if name not in ("features", "local_count", "global_count")
                },
            }
            rows, valid = (
                self.strategy.experimental_distribute_values_from_function(
                    lambda ctx: (rows, valid)
                )
            )
        metrics = self._updates[update](rows, valid)
        # One device-to-host transfer instead of one scalar read per metric.
        values = tf.stack(list(metrics.values())).numpy()
        return {**dict(zip(metrics, values)), "trained_positions": int(trained)}
