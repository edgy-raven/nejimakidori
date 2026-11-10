"""One frozen-policy collection cohort ahead of the AWR learner."""

import concurrent.futures
import time

import numpy
import tensorflow as tf

from log_dataset import _native_collection, training_data
from model import export
from training import self_play


def disagreement_evaluator(network):
    """Compare discard max-Q and closest-to-V in projected current-hand ST3."""
    module = export.ServingModel(network)
    specs = {
        name: tf.TensorSpec([None, *spec["shape"]], spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    signature = module.root_evaluation.get_concrete_function(**specs)
    packing = training_data.PackedBatch(specs)

    @tf.function(reduce_retracing=True)
    def predict(inputs):
        return signature(**packing.unpack(inputs))

    def evaluate(features):
        with tf.device("/CPU:0"):
            inputs = {
                name: tf.convert_to_tensor(value)
                for name, value in packing.pack_numpy(features).items()
            }
        outputs = predict(inputs)
        values = outputs["state_value"].numpy() * 100
        indices = outputs["critic_action_indices"].numpy()
        estimates = outputs["critic_projected_st3"].numpy()
        priority = numpy.zeros(len(values), numpy.float32)
        pairs = numpy.full((len(values), 2), -1, numpy.int32)
        # Encoder tf.where order is preserved by the export's active mask.
        boundaries = numpy.searchsorted(
            indices[:, 0], numpy.arange(len(values) + 1)
        )
        for row, (begin, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            codes = indices[begin:end, 1]
            q = estimates[begin:end]
            best = q.argmax()
            closest = numpy.abs(q - values[row]).argmin()
            if best == closest:
                continue
            pairs[row] = codes[[best, closest]]
            priority[row] = (
                q[best] - q[closest] >= 0.25
                or abs(float(q[closest] - values[row])) >= 8
            )
        return {"priority": priority, "actions": pairs}

    return evaluate


class Prefetch:
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
                root_evaluate=disagreement_evaluator(self.network),
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
