"""Fixed-policy south-game collection for on-policy training."""

import json

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
        return outputs[:count]

    boundaries = numpy.cumsum(
        [actions.ACTOR_OUTPUTS[name][0] for name in actions.TRAIN_POLICY_HEADS]
    )[:-1]

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
        # Transfer one contiguous buffer; simulation consumes NumPy logits.
        return dict(
            zip(
                actions.TRAIN_POLICY_HEADS,
                numpy.split(predict(packed, count).numpy(), boundaries, axis=1),
            )
        )

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


def improving_call_choices(features):
    """Legal call/discard choices that strictly reduce actual shanten."""
    # Family-category inputs clip high shanten; efficiency retains its value.
    distance = features["self_efficiency"][..., 0] * 8
    chi = features["chi_discard_legal"].astype(bool) & (
        features["chi_discard_shanten"] < distance[..., None, None] + 1
    )
    pon = features["pon_discard_legal"].astype(bool) & (
        features["pon_discard_shanten"] < distance[..., None, None] + 1
    )
    # Kan efficiency stores actual shanten / 8, before the replacement draw.
    kan = features["kan_legal_mask"][..., 1:].astype(bool) & (
        features["kan_efficiency"][..., 0] * 8 < distance[..., None]
    )
    decline = numpy.zeros((*distance.shape, 1), bool)
    response = numpy.concatenate(
        (
            decline,
            chi.reshape(*distance.shape, 111),
            pon.reshape(*distance.shape, 37),
            (kan.any(axis=-1) & (features["response_legal_mask"][..., 3] > 0))[
                ..., None
            ],
        ),
        axis=-1,
    )
    return response, numpy.concatenate((decline, kan), axis=-1)


def policy_log_probabilities(outputs, heads, legal):
    """Normalize active policy heads over their exact legal action masks."""
    scores = numpy.full((len(heads), 150), -1e9, numpy.float32)
    for head, name in enumerate(actions.TRAIN_POLICY_HEADS):
        rows = heads == head
        scores[rows, : actions.ACTOR_OUTPUTS[name][0]] = numpy.asarray(
            outputs[name]
        )[rows]
    scores[~legal] = -1e9
    scores -= scores.max(axis=1, keepdims=True)
    log_probability = scores - numpy.log(
        numpy.exp(scores).sum(axis=1, keepdims=True)
    )
    log_probability[~legal] = -1e9
    return log_probability


def request_batch(request):
    """Decode one native request for simulation or recorded-match discovery."""
    return {
        "features": numpy.frombuffer(request["features"], FEATURE_DTYPE),
        "head": numpy.asarray(request["head"]),
        "legal": numpy.frombuffer(request["legal"], numpy.bool_).reshape(
            len(request["head"]), 150
        ),
        "indices": numpy.asarray(
            [request[name] for name in ("game", "hand", "seat", "event")]
        ).T,
    }


def collect(
    predict,
    games,
    seed,
    *,
    collector=None,
    root_callback=None,
    greedy=False,
):
    return collect_streams(
        predict,
        [
            collection_steps(
                games,
                seed,
                collector=collector,
                root_callback=root_callback,
                greedy=greedy,
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
    collector=None,
    root_callback=None,
    forced_root=None,
    greedy=False,
):
    """Collect fresh games or supplied native continuations, without records."""
    if greedy:
        import tensorflow as tf

        from model import actor as actor_layers

    first = True
    if collector is None:
        collector = _native_collection.NativeCollector(games, seed)
    generator = numpy.random.default_rng(seed)
    batches = []
    while (request := collector.next()) is not None:
        count = len(request["head"])
        batch = request_batch(request)
        outputs = yield batch["features"]
        batch["behavior"] = policy_log_probabilities(
            outputs, batch["head"], batch["legal"]
        )
        if root_callback is not None:
            root_callback(collector, batch)
        if greedy:
            with tf.device("/CPU:0"):
                batch["chosen"] = actor_layers.greedy_choices(
                    batch["behavior"], batch["head"]
                ).numpy()
        else:
            cumulative = numpy.exp(batch["behavior"]).cumsum(
                axis=1, dtype=numpy.float64
            )
            cumulative /= cumulative[:, -1:]
            batch["chosen"] = (
                generator.random((len(cumulative), 1)) >= cumulative
            ).sum(axis=1)
        batch["paired_root"] = numpy.zeros(count, numpy.bool_)
        if first and forced_root is not None:
            decision, action = forced_root
            rows = numpy.arange(games) * (count // games) + decision
            assert batch["legal"][rows, action].all()
            batch["chosen"][rows] = action
            batch["behavior"][rows] = -1e9
            batch["behavior"][rows, action] = 0
            batch["paired_root"][rows] = True
        first = False
        collector.submit(batch["chosen"].tolist())
        batches.append(batch)
    logs, statistics = collector.finish()
    logs = [json.loads(events) for events in logs]
    for events in logs:
        outcomes.normalize_draw_payments(events)
    payments = []
    for events in logs:
        assert events[-1]["type"] in ("end_game", "start_kyoku")
        starts = [
            i
            for i, event in enumerate(events)
            if event["type"] == "start_kyoku"
        ]
        completed = len(starts) - (events[-1]["type"] != "end_game")
        payments.append(
            [
                outcomes.terminal_payments(events[start:end])
                for start, end in zip(
                    starts[:completed], starts[1:] + [len(events)]
                )
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
                "paired_root",
            )
        },
    }
    features = numpy.concatenate(
        [part["features"] for part in batches], dtype=FEATURE_DTYPE
    )
    payment_kind, payer, recipient = numpy.nonzero(outcomes.PAYMENT_CELLS)
    batch["payment_points"] = numpy.concatenate(payments)[
        hand_indices[:, None],
        payment_kind,
        (indices[:, 2, None] + payer) % 4,
        (indices[:, 2, None] + recipient) % 4,
    ] + numpy.int32(1000) * (
        (payment_kind == 3) & (features["riichi_state"][:, recipient] == 2)
    )
    batch["features"] = {name: features[name] for name in FEATURE_DTYPE.names}
    return batch, statistics
