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


def initial_features(seed):
    """Valid model-build inputs from fresh simulator states, without records."""
    collector = _native_collection.NativeCollector(8, seed)
    request = collector.next()
    rows = numpy.frombuffer(request["features"], FEATURE_DTYPE)
    return {name: rows[name].copy() for name in FEATURE_DTYPE.names}


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


def target_categories(features, heads, priority):
    """Observable targets shared by recorded decisions and generated replay."""
    distance = numpy.minimum.reduce(
        [
            features[name]
            for name in (
                "normal_self_shanten",
                "chiitoitsu_self_shanten",
                "kokushi_self_shanten",
            )
        ]
    )
    # The feature stores actual shanten + 1: tenpai=1, one-away=2.
    return (
        ((distance <= 2) & (features["live_wall_count"] <= 16)).astype(
            numpy.uint8
        )
        | ((distance <= 1).astype(numpy.uint8) << 1)
        | (((heads == 2) | (heads == 3)).astype(numpy.uint8) << 2)
        | ((priority > 0).astype(numpy.uint8) << 3)
    )


def collect(
    predict,
    games,
    seed,
    *,
    retain_features=True,
    collector=None,
    root_callback=None,
):
    return collect_streams(
        predict,
        [
            collection_steps(
                games,
                seed,
                retain_features=retain_features,
                collector=collector,
                root_callback=root_callback,
            )
        ],
    )[0]


def collect_streams(predict, streams):
    """Batch independent trajectories without merging their RNGs or targets."""
    results = [None] * len(streams)
    active = {index: next(stream) for index, stream in enumerate(streams)}
    while active:
        outputs = predict(
            next(iter(active.values()))
            if len(active) == 1
            else numpy.concatenate(list(active.values()), dtype=FEATURE_DTYPE)
        )
        offset = 0
        for index, features in list(active.items()):
            count = len(features)
            try:
                active[index] = streams[index].send(
                    {
                        name: value[offset : offset + count]
                        for name, value in outputs.items()
                    }
                )
            except StopIteration as finished:
                results[index] = finished.value
                del active[index]
            offset += count
    return results


def collection_steps(
    games,
    seed,
    *,
    retain_features=True,
    collector=None,
    root_callback=None,
    forced_root=None,
):
    """Collect fresh games or supplied native continuations, without records."""
    first = True
    if collector is None:
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
        outputs = yield batch["features"]
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
        batch["targets"] = target_categories(
            batch["features"],
            batch["head"],
            (
                outputs["_root_priority"]
                if "_root_priority" in outputs
                else numpy.zeros(count, numpy.float32)
            ),
        )
        if root_callback is not None:
            root_callback(collector, batch)
        cumulative = numpy.exp(batch["behavior"]).cumsum(
            axis=1, dtype=numpy.float64
        )
        cumulative /= cumulative[:, -1:]
        batch["chosen"] = (
            generator.random((len(cumulative), 1)) >= cumulative
        ).sum(axis=1)
        batch["paired_root"] = numpy.zeros(count, numpy.bool_)
        batch["root_path"] = numpy.zeros(count, numpy.uint8)
        if first and forced_root is not None:
            decision, action, path = forced_root
            rows = numpy.arange(games) * (count // games) + decision
            assert batch["legal"][rows, action].all()
            batch["chosen"][rows] = action
            batch["behavior"][rows] = -1e9
            batch["behavior"][rows, action] = 0
            batch["paired_root"][rows] = True
            batch["root_path"][rows] = path
        first = False
        collector.submit(batch["chosen"].tolist())
        # Copy only this small target input before releasing a worker's rows.
        batch["riichi_state"] = batch["features"]["riichi_state"].copy()
        if not retain_features:
            batch["_feature_rows"] = outputs["_feature_rows"]
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
            for name in (
                "head",
                "legal",
                "chosen",
                "behavior",
                "targets",
                "paired_root",
                "root_path",
            )
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
    if retain_features:
        features = numpy.concatenate(
            [part["features"] for part in batches], dtype=FEATURE_DTYPE
        )
        batch["features"] = {
            name: features[name] for name in FEATURE_DTYPE.names
        }
    else:
        batch["_feature_rows"] = numpy.concatenate(
            [part["_feature_rows"] for part in batches]
        )
    return batch, statistics


def focused_action_pair(features, head, legal, policy):
    """Policy-preferred choices in distinct call or shanten groups."""
    codes = numpy.flatnonzero(legal)
    if head in (2, 3):
        calls = codes[codes > 0]
        if not len(calls):
            return None
        assert legal[0], "Optional calls require the pass choice"
        return numpy.array([calls[policy[calls].argmax()], 0], numpy.int32)
    # Riichi codes add 37 to the corresponding discard code.
    distance = features["discard_action_all_shanten"][codes % 37]
    keep = codes[distance == distance.min()]
    leave = codes[distance > distance.min()]
    if not len(leave):
        return None
    return numpy.array(
        [keep[policy[keep].argmax()], leave[policy[leave].argmax()]],
        numpy.int32,
    )


def collect_focused(
    predict,
    games,
    seed,
    *,
    focus,
    root_copies,
    root_evaluate,
    retain_features=True,
):
    """Retain natural play and add continuations from sampled eligible roots."""
    if focus == "natural":
        return collect(predict, games, seed, retain_features=retain_features)
    if (
        focus not in ("keiten", "tenpai", "calls", "disagreement", "targeted")
        or root_copies < 1
    ):
        raise ValueError("Invalid self-play focus settings")
    roots = {}
    seen = numpy.zeros(games, numpy.int64)
    generator = numpy.random.default_rng(seed + 1000000000)

    focus_mask = {
        "keiten": 1,
        "tenpai": 2,
        "calls": 4,
        "disagreement": 8,
        "targeted": 15,
    }[focus]

    def capture(collector, batch):
        for row in numpy.flatnonzero(batch["targets"] & focus_mask):
            if batch["head"][row] == 4:
                continue
            # Calls have their own path even when the hand is also tenpai.
            eligible = (
                int(batch["targets"][row])
                & focus_mask
                & (4 if batch["head"][row] in (2, 3) else 11)
            )
            paths = [bit for bit in (1, 2, 4, 8) if eligible & bit]
            if not paths:
                continue
            path = int(generator.choice(paths))
            pair = (
                None
                if path == 8
                else focused_action_pair(
                    batch["features"][row],
                    batch["head"][row],
                    batch["legal"][row],
                    batch["behavior"][row],
                )
            )
            if path != 8 and pair is None:
                continue
            game = int(batch["indices"][row, 0])
            seen[game] += 1
            if generator.integers(seen[game]) == 0:
                roots[game] = (
                    collector.snapshot(int(row)),
                    batch["features"][row].copy(),
                    pair,
                    path,
                )

    batch, statistics = collect(
        predict,
        games,
        seed,
        retain_features=retain_features,
        root_callback=capture,
    )
    batches = [batch]
    # Keep approximately the former two continuation-games per base game.
    # Spend that budget on fewer roots and repeated, explicit action pairs.
    root_games = sorted(roots)
    streams = []
    if root_games:
        pairs = numpy.array(
            [
                [-1, -1] if roots[game][3] == 8 else roots[game][2]
                for game in root_games
            ],
            dtype=numpy.int32,
        )
        disagreement = [
            row for row, game in enumerate(root_games) if roots[game][3] == 8
        ]
        if disagreement:
            pairs[disagreement] = root_evaluate(
                numpy.array(
                    [roots[root_games[row]][1] for row in disagreement],
                    dtype=FEATURE_DTYPE,
                )
            )["actions"]
        eligible = numpy.flatnonzero(pairs[:, 0] >= 0)
        selected = generator.choice(
            eligible,
            size=min(
                len(eligible), max(1, round(len(root_games) / root_copies))
            ),
            replace=False,
        )
        for row in sorted(selected):
            game = root_games[row]
            pair = pairs[row]
            root = roots[game][0]
            for action in pair:
                streams.append(
                    collection_steps(
                        root_copies,
                        seed + 1000000000 + game * root_copies,
                        retain_features=retain_features,
                        collector=root.fork(
                            root_copies, seed + 1000000000 + game * root_copies
                        ),
                        forced_root=(
                            root.decision,
                            int(action),
                            roots[game][3],
                        ),
                    )
                )
    for batch, counts in collect_streams(predict, streams):
        batches.append(batch)
        for name, value in counts.items():
            statistics[name] += value
    statistics["selected_roots"] = len(streams) // 2
    statistics["root_continuations"] = len(streams) * root_copies
    return {
        name: (
            {
                key: numpy.concatenate([part[name][key] for part in batches])
                for key in batches[0][name]
            }
            if name == "features"
            else numpy.concatenate([part[name] for part in batches])
        )
        for name in batches[0]
    }, statistics


class Collector:
    def __init__(
        self,
        predict,
        workers,
        *,
        focus="natural",
        root_copies=1,
        root_evaluate=None,
    ):
        if workers < 1:
            raise ValueError("Collection workers must be positive")
        self.focus = focus
        self.root_copies = root_copies
        self.root_evaluate = root_evaluate
        self.infer = predict
        self.metrics = dict(
            inference_seconds=0.0,
            inference_calls=0,
            inference_rows=0,
            worker_wait_seconds=0.0,
            selected_roots=0,
            root_continuations=0,
            root_evaluation_seconds=0.0,
            root_evaluation_rows=0,
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
        if self.focus in ("disagreement", "targeted"):
            priority = numpy.zeros(len(features), numpy.float32)
            selected = numpy.flatnonzero(
                (features["decision_phase"] == 0)
                & (self.root_generator.random(len(features)) < 1 / 64)
            )
            if len(selected):
                priority[selected] = self.evaluate_roots(
                    {
                        name: features[name][selected]
                        for name in FEATURE_DTYPE.names
                    }
                )["priority"]
            outputs["_root_priority"] = priority
        return outputs

    def evaluate_roots(self, features):
        started = time.monotonic()
        outputs = self.root_evaluate(features)
        self.metrics["root_evaluation_seconds"] += time.monotonic() - started
        self.metrics["root_evaluation_rows"] += len(features["decision_phase"])
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
        self.root_generator = numpy.random.default_rng(seed + 2000000000)
        if self.workers == 1:
            numpy.random.seed(seed)
            batch, statistics = collect_focused(
                self.predict,
                games,
                seed,
                focus=self.focus,
                root_copies=self.root_copies,
                root_evaluate=lambda rows: self.evaluate_roots(
                    {name: rows[name] for name in FEATURE_DTYPE.names}
                ),
            )
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
                            "training.self_play",
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
            connection.send((size, seed + offset, self.focus, self.root_copies))
            offset += size
        active = list(range(count))
        results = {}
        features = [[] for _ in range(count)]
        feature_counts = [0] * count
        while active:
            requests = []
            root_requests = []
            waiting = list(active)
            # Combine the workers' concurrent trajectories in one GPU batch.
            while waiting:
                started = time.monotonic()
                ready = multiprocessing.connection.wait(
                    [self.connections[index] for index in waiting]
                )
                self.metrics["worker_wait_seconds"] += (
                    time.monotonic() - started
                )
                for connection in ready:
                    index = self.connections.index(connection)
                    waiting.remove(index)
                    kind, value = connection.recv()
                    if kind == "result":
                        results[index], statistics = value
                        for name, total in statistics.items():
                            self.metrics[name] += total
                        active.remove(index)
                    elif kind == "roots":
                        root_requests.append(
                            (index, numpy.frombuffer(value, FEATURE_DTYPE))
                        )
                    else:
                        value = numpy.frombuffer(value, FEATURE_DTYPE)
                        features[index].append(value)
                        feature_counts[index] += len(value)
                        requests.append((index, value))
            if root_requests:
                root_rows = numpy.concatenate(
                    [rows for _, rows in root_requests], dtype=FEATURE_DTYPE
                )
                scores = self.evaluate_roots(
                    {name: root_rows[name] for name in FEATURE_DTYPE.names}
                )
                offset = 0
                for index, rows in root_requests:
                    self.connections[index].send(
                        {
                            name: value[offset : offset + len(rows)]
                            for name, value in scores.items()
                        }
                    )
                    offset += len(rows)
            if requests:
                outputs = self.predict(
                    requests[0][1]
                    if len(requests) == 1
                    else numpy.concatenate(
                        [request[1] for request in requests],
                        dtype=FEATURE_DTYPE,
                    )
                )
                outputs["_feature_rows"] = numpy.concatenate(
                    [
                        numpy.arange(
                            feature_counts[index] - len(rows),
                            feature_counts[index],
                        )
                        for index, rows in requests
                    ]
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
        ordered = numpy.empty(sum(feature_counts), dtype=FEATURE_DTYPE)
        offset = 0
        for index in range(count):
            worker_features = numpy.concatenate(
                features[index], dtype=FEATURE_DTYPE
            )
            features[index].clear()
            size = len(worker_features)
            ordered[offset : offset + size] = worker_features[
                results[index].pop("_feature_rows")
            ]
            del worker_features
            offset += size
        return {
            "features": {name: ordered[name] for name in FEATURE_DTYPE.names},
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

        def root_evaluate(features):
            connection.send(("roots", features.tobytes()))
            return connection.recv()

        while (task := connection.recv()) is not None:
            games, seed, focus, root_copies = task
            numpy.random.seed(seed)
            connection.send(
                (
                    "result",
                    collect_focused(
                        predict,
                        games,
                        seed,
                        retain_features=False,
                        focus=focus,
                        root_copies=root_copies,
                        root_evaluate=root_evaluate,
                    ),
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-fd", type=int, required=True)
    worker(parser.parse_args().worker_fd)
