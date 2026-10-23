"""Fixed-policy south-game collection for on-policy training."""

import argparse
import json
import multiprocessing
import multiprocessing.connection
import os
import subprocess
import sys
import time

import numpy

from log_dataset import _native_collection
from model import actions, outcomes

FEATURE_DTYPE = numpy.dtype(
    [
        (name, spec["dtype"], tuple(spec["shape"]))
        for name, spec in _native_collection.INPUTS.items()
    ]
)


def predictor(actor):
    """Pack on the CPU and transfer a few lossless buffers per inference."""
    import tensorflow as tf

    from log_dataset import training_data

    strategy = tf.distribute.get_strategy()
    packing = training_data.PackedBatch(
        {
            name: tf.TensorSpec([None, *spec["shape"]], spec["dtype"])
            for name, spec in _native_collection.INPUTS.items()
        }
    )

    def infer(inputs):
        outputs = actor(packing.unpack(inputs), training=False)
        return tf.concat(
            [outputs[name] for name in actions.TRAIN_POLICY_HEADS],
            axis=1,
        )

    @tf.function(
        input_signature=[
            {
                dtype: tf.TensorSpec(
                    [None, None],
                    (
                        "float32"
                        if dtype in ("bool", "int8", "int16", "int32")
                        else dtype
                    ),
                )
                for dtype in packing.groups
            },
            tf.TensorSpec([], tf.int32),
        ]
    )
    def predict(inputs, count):
        if strategy.num_replicas_in_sync == 1:
            outputs = strategy.run(infer, args=(inputs,))
        else:
            size = tf.shape(next(iter(inputs.values())))[0] // (
                strategy.num_replicas_in_sync
            )
            with tf.device("/CPU:0"):
                distributed = (
                    strategy.experimental_distribute_values_from_function(
                        lambda ctx: {
                            name: value[
                                ctx.replica_id_in_sync_group
                                * size : (ctx.replica_id_in_sync_group + 1)
                                * size
                            ]
                            for name, value in inputs.items()
                        }
                    )
                )
            outputs = strategy.gather(
                strategy.run(infer, args=(distributed,)), axis=0
            )
        return dict(
            zip(
                actions.TRAIN_POLICY_HEADS,
                tf.split(
                    outputs[:count],
                    [
                        actions.ACTOR_OUTPUTS[name][0]
                        for name in actions.TRAIN_POLICY_HEADS
                    ],
                    axis=1,
                ),
            )
        )

    def packed_predict(inputs):
        count = len(next(iter(inputs.values())))
        size = (count + strategy.num_replicas_in_sync - 1) // (
            strategy.num_replicas_in_sync
        )
        size = (1 << (size - 1).bit_length()) * strategy.num_replicas_in_sync
        indices = numpy.minimum(numpy.arange(size), count - 1)
        with tf.device("/CPU:0"):
            packed = {
                name: tf.convert_to_tensor(value[indices])
                for name, value in packing.pack_numpy(inputs).items()
            }
            count = tf.constant(count, tf.int32)
        return predict(packed, count)

    return packed_predict


def collection_counts():
    return dict.fromkeys(
        (
            "games",
            "hands",
            "wins",
            "tsumo_wins",
            "win_points",
            "dealin_hands",
            "dealin_points",
            "riichi_hands",
            "riichi_wins",
            "riichi_dealins",
            "riichi_turn_sum",
            "open_hands",
            "draws",
            "exhaustive_draws",
            "exhaustive_tenpai_seats",
        ),
        0,
    )


def collect(predict, games, seed, *, retain_features=True):
    """Predict from packed structured rows; optionally retain training inputs."""
    collector = _native_collection.NativeCollector(games, seed)
    generator = numpy.random.default_rng(seed)
    batches = []
    while (request := collector.next()) is not None:
        count = len(request["head"])
        batch = {
            "features": numpy.frombuffer(request["features"], FEATURE_DTYPE),
            "head": numpy.asarray(request["head"]),
            "legal": numpy.frombuffer(request["legal"], numpy.bool_).reshape(
                count, 150
            ),
            "indices": numpy.asarray(
                [request[name] for name in ("game", "hand", "seat", "event")]
            ).T,
        }
        outputs = predict(batch["features"])
        scores = numpy.full((count, 150), -1e9, numpy.float32)
        for head, name in enumerate(actions.TRAIN_POLICY_HEADS):
            rows = batch["head"] == head
            scores[rows, : actions.ACTOR_OUTPUTS[name][0]] = numpy.asarray(
                outputs[name]
            )[rows]
        scores[~batch["legal"]] = -1e9
        scores -= scores.max(axis=1, keepdims=True)
        batch["behavior"] = scores - numpy.log(
            numpy.exp(scores).sum(axis=1, keepdims=True)
        )
        batch["behavior"][~batch["legal"]] = -1e9
        cumulative = numpy.exp(batch["behavior"]).cumsum(
            axis=1, dtype=numpy.float64
        )
        cumulative /= cumulative[:, -1:]
        batch["chosen"] = (
            generator.random((len(cumulative), 1)) >= cumulative
        ).sum(axis=1)
        collector.submit(batch["chosen"].tolist())
        # Copy only this small target input before releasing a worker's rows.
        batch["riichi_state"] = batch["features"]["riichi_state"].copy()
        if not retain_features:
            del batch["features"]
        batches.append(batch)
    logs, statistics = collector.finish()
    logs = [json.loads(events) for events in logs]
    for events in logs:
        outcomes.normalize_draw_payments(events)
    settlements = [outcomes.hand_continuation_states(events) for events in logs]
    payments = []
    for events in logs:
        starts = [
            i
            for i, event in enumerate(events)
            if event["type"] == "start_kyoku"
        ]
        payments.append(
            [
                outcomes.terminal_payments(events[start:end])
                for start, end in zip(starts, starts[1:] + [len(events)])
            ]
        )
    indices = numpy.concatenate([batch.pop("indices") for batch in batches])
    hand_indices = (
        numpy.concatenate(
            ([0], numpy.cumsum([len(game) for game in payments]))
        )[indices[:, 0]]
        + indices[:, 1]
    )
    batch = {
        **{
            name: numpy.concatenate([batch[name] for batch in batches])
            for name in ("head", "legal", "chosen", "behavior")
        },
        "settlement": numpy.concatenate(settlements)[
            hand_indices, indices[:, 2]
        ],
    }
    payment_kind, payer, recipient = numpy.nonzero(outcomes.PAYMENT_CELLS)
    batch["payment_points"] = numpy.concatenate(payments)[
        hand_indices[:, None],
        payment_kind,
        (indices[:, 2, None] + payer) % 4,
        (indices[:, 2, None] + recipient) % 4,
    ] + numpy.int32(1000) * (
        (payment_kind == 3)
        & (
            numpy.concatenate([part["riichi_state"] for part in batches])[
                :, recipient
            ]
            == 2
        )
    )
    # Call forecasts condition on acceptance. A competing call or ron leaves
    # the policy return valid, but supplies no outcome for that critic branch.
    batch["critic_valid"] = (batch["head"] != 2) | (batch["chosen"] == 0)
    for game, events in enumerate(logs):
        rows = numpy.flatnonzero(
            (indices[:, 0] == game) & ~batch["critic_valid"]
        )
        resolutions = numpy.asarray(
            [
                (
                    cursor,
                    (
                        event["actor"]
                        if event["type"] in ("chi", "pon", "daiminkan")
                        else -1
                    ),
                )
                for cursor, event in enumerate(events)
                if event["type"]
                in ("chi", "pon", "daiminkan", "hora", "tsumo", "ryukyoku")
            ]
        )
        batch["critic_valid"][rows] = (
            resolutions[
                numpy.searchsorted(resolutions[:, 0], indices[rows, 3]), 1
            ]
            == indices[rows, 2]
        )
    if retain_features:
        features = numpy.concatenate(
            [part["features"] for part in batches], dtype=FEATURE_DTYPE
        )
        batch["features"] = {
            name: features[name] for name in FEATURE_DTYPE.names
        }
    return batch, statistics


class Collector:
    def __init__(self, predict, workers):
        if workers < 1:
            raise ValueError("Collection workers must be positive")
        self.infer = predict
        self.metrics = dict(
            inference_seconds=0.0,
            inference_calls=0,
            inference_rows=0,
            worker_wait_seconds=0.0,
            **collection_counts(),
        )
        self.workers = workers
        self.connections = []
        self.processes = []

    def __enter__(self):
        return self

    def predict(self, features):
        started = time.monotonic()
        outputs = {
            name: numpy.asarray(value)
            for name, value in self.infer(
                {name: features[name] for name in FEATURE_DTYPE.names}
            ).items()
        }
        self.metrics["inference_seconds"] += time.monotonic() - started
        self.metrics["inference_calls"] += 1
        self.metrics["inference_rows"] += len(features["decision_phase"])
        return outputs

    def __exit__(self, exc_type, exc, traceback):
        try:
            for connection, process in zip(self.connections, self.processes):
                if process.poll() is None:
                    if exc_type is None:
                        connection.send(None)
                    else:
                        process.terminate()
        finally:
            for process in self.processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            for connection in self.connections:
                connection.close()

    def collect(self, games, seed):
        if games < 1:
            raise ValueError("Collection games must be positive")
        if self.workers == 1:
            numpy.random.seed(seed)
            batch, statistics = collect(self.predict, games, seed)
            for name, value in statistics.items():
                self.metrics[name] += value
            return batch
        count = min(self.workers, games)
        while len(self.processes) < count:
            parent, child = multiprocessing.Pipe()
            self.connections.append(parent)
            with child:
                self.processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "model.self_play",
                            "--worker-fd",
                            str(child.fileno()),
                        ],
                        pass_fds=(child.fileno(),),
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": "",
                            "TF_NUM_INTEROP_THREADS": "1",
                            "TF_NUM_INTRAOP_THREADS": "1",
                            "OMP_NUM_THREADS": "1",
                        },
                    )
                )
        offset = 0
        for index, connection in enumerate(self.connections[:count]):
            size = games // count + (index < games % count)
            connection.send((size, seed + offset))
            offset += size
        active = list(range(count))
        results = {}
        features = [[] for _ in range(count)]
        while active:
            requests = []
            waiting = list(active)
            # Leave half the workers preparing their next decisions while
            # inference serves the other half. A full barrier serializes CPU
            # feature work and GPU inference, and waits for every straggler.
            while waiting and len(requests) < max(1, (len(active) + 1) // 2):
                started = time.monotonic()
                ready = multiprocessing.connection.wait(
                    [self.connections[index] for index in waiting]
                )
                self.metrics["worker_wait_seconds"] += (
                    time.monotonic() - started
                )
                # Drain this ready set; the threshold only limits waiting.
                for connection in ready:
                    index = self.connections.index(connection)
                    waiting.remove(index)
                    kind, value = connection.recv()
                    if kind == "result":
                        results[index], statistics = value
                        for name, total in statistics.items():
                            self.metrics[name] += total
                        active.remove(index)
                    else:
                        value = numpy.frombuffer(value, FEATURE_DTYPE)
                        features[index].append(value)
                        requests.append((index, value))
            if requests:
                outputs = self.predict(
                    requests[0][1]
                    if len(requests) == 1
                    else numpy.concatenate(
                        [request[1] for request in requests],
                        dtype=FEATURE_DTYPE,
                    )
                )
                offset = 0
                for index, request_features in requests:
                    size = len(request_features["decision_phase"])
                    self.connections[index].send(
                        {
                            name: value[offset : offset + size]
                            for name, value in outputs.items()
                        }
                    )
                    offset += size
        features = numpy.concatenate(
            [batch for worker_batches in features for batch in worker_batches],
            dtype=FEATURE_DTYPE,
        )
        return {
            "features": {name: features[name] for name in FEATURE_DTYPE.names},
            **{
                name: (
                    results[0][name]
                    if count == 1
                    else numpy.concatenate(
                        [results[index][name] for index in range(count)]
                    )
                )
                for name in results[0]
            },
        }


def worker(fd):
    with multiprocessing.connection.Connection(fd) as connection:

        def predict(features):
            connection.send(("predict", features.tobytes()))
            return connection.recv()

        while (task := connection.recv()) is not None:
            games, seed = task
            numpy.random.seed(seed)
            connection.send(
                (
                    "result",
                    collect(predict, games, seed, retain_features=False),
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-fd", type=int, required=True)
    worker(parser.parse_args().worker_fd)
