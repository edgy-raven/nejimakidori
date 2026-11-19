"""One frozen-policy collection cohort ahead of the learner."""

import concurrent.futures
import time

import numpy
import tensorflow as tf
from log_dataset import _native_collection, training_data
from model import export, outcomes
from training import self_play


def root_scorer(network):
    """Compare discard max-Q with the current policy choice."""
    module = export.ServingModel(network)
    specs = {
        name: tf.TensorSpec([None, *spec["shape"]], spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    signature = module.root_evaluation.get_concrete_function(**specs)
    packing = training_data.PackedBatch(specs)

    @tf.function(reduce_retracing=True)
    def predict(inputs):
        outputs = signature(**packing.unpack(inputs))
        state_rows = outputs["critic_action_indices"][:, 0]
        q_objective = outputs["critic_objective"]
        state_count = tf.shape(outputs["discard_policy_logits"])[0]
        policy_scores = tf.gather_nd(
            outputs["discard_policy_logits"],
            outputs["critic_action_indices"],
        )
        action_positions = tf.range(tf.size(q_objective))
        selected = []
        # Minimize -Q for the best action, then -policy logit for the comparison.
        # Ties choose the first candidate in the existing action order.
        for scores in (-q_objective, -policy_scores):
            minimum = tf.math.unsorted_segment_min(
                scores, state_rows, state_count
            )
            tied_for_minimum = scores == tf.gather(minimum, state_rows)
            selected.append(
                tf.math.unsorted_segment_min(
                    tf.where(
                        tied_for_minimum, action_positions, tf.size(q_objective)
                    ),
                    state_rows,
                    state_count,
                )
            )
        best, preferred = selected
        different_actions = best != preferred
        q_gap_objective = tf.gather(q_objective, best) - tf.gather(
            q_objective, preferred
        )
        return {
            "q_values": tf.tensor_scatter_nd_update(
                tf.fill([state_count, 37], tf.constant(-float("inf"))),
                outputs["critic_action_indices"],
                q_objective,
            ),
            "priority": tf.cast(
                different_actions
                & (q_gap_objective >= 0.25 / outcomes.OBJECTIVE_ST3_SCALE),
                tf.float32,
            ),
            "actions": tf.where(
                different_actions[:, None],
                tf.gather(
                    outputs["critic_action_indices"][:, 1],
                    tf.stack([best, preferred], axis=1),
                ),
                -1,
            ),
        }

    def evaluate(features):
        # Stable encoder shapes avoid recompiling for each sampled root count.
        packed = packing.pack_numpy(features)
        count = len(features["decision_phase"])
        # Small self-play requests need small buckets, not 64 padded roots.
        chunk_size = min(64, 1 << (count - 1).bit_length())
        outputs = []
        for begin in range(0, count, chunk_size):
            indices = numpy.minimum(
                numpy.arange(begin, begin + chunk_size), count - 1
            )
            with tf.device("/CPU:0"):
                inputs = {
                    name: tf.convert_to_tensor(value[indices])
                    for name, value in packed.items()
                }
            outputs.append(
                {
                    name: value.numpy()[: min(chunk_size, count - begin)]
                    for name, value in predict(inputs).items()
                }
            )
        return {
            name: numpy.concatenate([output[name] for output in outputs])
            for name in outputs[0]
        }

    return evaluate


class BackgroundCollector:
    def __init__(self, *, weights, device, games, workers, root_copies, seed):
        self.device = device
        self.games = games
        self.positions_per_game = None
        self.pending = None
        with tf.device(device):
            self.network = export.load_network(
                weights, self_play.initial_features(seed)
            )
            self.collector = self_play.Collector(
                self_play.predictor(self.network.actor),
                workers,
                focus="targeted",
                root_copies=root_copies,
                root_evaluate=root_scorer(self.network),
            )

    def __enter__(self):
        self.collector.__enter__()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return self

    def start(self, weights, positions, seed):
        assert self.pending is None
        with tf.device(self.device):
            self.network.set_weights(weights)
        self.pending = self.executor.submit(self.collect, positions, seed)

    def collect(self, positions, seed):
        started = time.monotonic()
        before = dict(self.collector.metrics)
        batches = []
        fresh = games = 0
        with tf.device(self.device):
            while fresh < positions:
                count = (
                    self.games
                    if self.positions_per_game is None
                    else min(
                        self.games,
                        int(
                            numpy.ceil(
                                (positions - fresh) / self.positions_per_game
                            )
                        ),
                    )
                )
                batch = self.collector.collect(count, seed + games)
                games += count
                fresh += len(batch["chosen"])
                self.positions_per_game = len(batch["chosen"]) / count
                batches.append(batch)
        return {
            "batches": batches,
            "base_games": games,
            "positions": fresh,
            "seconds": time.monotonic() - started,
            "metrics": {
                name: value - before[name]
                for name, value in self.collector.metrics.items()
            },
        }

    def finish(self):
        result = self.pending.result()
        self.pending = None
        return result

    def __exit__(self, exc_type, exc, traceback):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.collector.__exit__(exc_type, exc, traceback)
