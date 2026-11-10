"""One frozen-policy collection cohort ahead of the AWR learner."""

import concurrent.futures
import time

import numpy
import tensorflow as tf

from model import export
from training import online_training, self_play


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
                root_evaluate=online_training.disagreement_evaluator(
                    self.network
                ),
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
