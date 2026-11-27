"""Fixed-policy south-game collection for on-policy training."""

import itertools
import json

import numpy

from log_dataset import _native_collection, call_states
from model import actions, outcomes


def initial_features(seed):
    """Valid model-build inputs from fresh simulator states, without records."""
    collector = _native_collection.NativeCollector(8, seed)
    request = collector.next()
    return call_states.from_request(request)


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
        outputs = actor(call_states.unpack(inputs, packing), training=False)
        return tf.concat(
            [outputs[name] for name in actions.TRAIN_POLICY_HEADS],
            axis=1,
        )

    @tf.function(
        input_signature=[
            {
                "base": {
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
                "post": {
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
                "owner": tf.TensorSpec([None], "int32"),
                "option": tf.TensorSpec([None], "int32"),
            },
            tf.TensorSpec([], tf.int32),
        ]
    )
    def predict(inputs, count):
        if strategy.num_replicas_in_sync == 1:
            outputs = strategy.run(infer, args=(inputs,))
        else:
            size = tf.shape(next(iter(inputs["base"].values())))[0] // (
                strategy.num_replicas_in_sync
            )
            with tf.device("/CPU:0"):
                distributed = (
                    strategy.experimental_distribute_values_from_function(
                        lambda ctx: call_states.tensor_packed_select(
                            inputs,
                            tf.range(
                                ctx.replica_id_in_sync_group * size,
                                (ctx.replica_id_in_sync_group + 1) * size,
                            ),
                        )
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
        count = len(inputs["decision_phase"])
        size = (count + strategy.num_replicas_in_sync - 1) // (
            strategy.num_replicas_in_sync
        )
        size = (1 << (size - 1).bit_length()) * strategy.num_replicas_in_sync
        indices = numpy.minimum(numpy.arange(size), count - 1)
        with tf.device("/CPU:0"):
            packed = tf.nest.map_structure(
                tf.convert_to_tensor,
                call_states.pack_numpy(
                    call_states.select(inputs, indices), packing
                ),
            )
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
            "simulated_decisions",
        ),
        0,
    )


def improving_call_choices(features):
    """Legal call/discard choices that strictly reduce actual shanten."""
    # Family-category inputs clip high shanten; efficiency retains its value.
    distance = features["self_efficiency"][..., 0] * 8
    calls = features["call_discard_legal"].astype(bool) & (
        features["call_discard_shanten"] < distance[..., None, None] + 1
    )
    # Kan efficiency stores actual shanten / 8, before the replacement draw.
    kan = features["kan_legal_mask"][..., 1:].astype(bool) & (
        features["kan_efficiency"][..., 0] * 8 < distance[..., None]
    )
    decline = numpy.zeros((*distance.shape, 1), bool)
    response = numpy.concatenate(
        (
            decline,
            calls.reshape(*distance.shape, 148),
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
        "features": call_states.from_request(request),
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
    with_labels=False,
    reference_policy=None,
):
    return collect_streams(
        predict,
        [
            collection_steps(
                games,
                seed,
                collector=collector,
                with_labels=with_labels,
                reference_policy=reference_policy,
            )
        ],
    )[0]


def collect_streams(predict, streams, max_active=None):
    """Batch independent trajectories without merging their RNGs or targets."""
    if max_active is None:
        streams = list(streams)
        max_active = len(streams)
    streams = iter(streams)
    results = []
    active = {}
    while True:
        for stream in itertools.islice(streams, max_active - len(active)):
            active[len(results)] = (stream, next(stream))
            results.append(None)
        if not active:
            break
        outputs = predict(
            next(iter(active.values()))[1]
            if len(active) == 1
            else call_states.concatenate([item[1] for item in active.values()])
        )
        offset = 0
        for index, (stream, features) in list(active.items()):
            count = len(features["decision_phase"])
            try:
                active[index] = (
                    stream,
                    stream.send(
                        {
                            name: value[offset : offset + count]
                            for name, value in outputs.items()
                        }
                    ),
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
    forced_root=None,
    with_labels=False,
    reference_policy=None,
    root_only=False,
    root_affinity=None,
):
    """Collect fresh games or supplied native continuations, without records.

    root_affinity supplies visible route labels shared by root-only rows.
    """
    first = True
    if collector is None:
        collector = _native_collection.NativeCollector(games, seed)
    generator = numpy.random.default_rng(seed)
    batches = []
    simulated_decisions = 0
    while (request := collector.next()) is not None:
        count = len(request["head"])
        simulated_decisions += count
        batch = request_batch(request)
        outputs = yield batch["features"]
        batch["behavior"] = policy_log_probabilities(
            outputs, batch["head"], batch["legal"]
        )
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
        if root_only:
            if batch["paired_root"].any():
                selected = numpy.flatnonzero(batch["paired_root"])
                batches.append(
                    {
                        "features": call_states.select(
                            batch["features"], selected
                        ),
                        **{
                            name: value[selected]
                            for name, value in batch.items()
                            if name != "features"
                        },
                    }
                )
        else:
            batches.append(batch)
    logs, statistics = collector.finish()
    statistics["simulated_decisions"] = simulated_decisions
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
        name: numpy.concatenate([batch[name] for batch in batches])
        for name in ("head", "legal", "chosen", "behavior", "paired_root")
    }
    features = call_states.concatenate([part["features"] for part in batches])
    if with_labels:
        from training import match_records

        batch["labels"] = match_records.label_rollouts(
            logs, indices, features, root_affinity=root_affinity
        )
    if reference_policy is not None:
        batch["reference_log_probability"] = numpy.empty(
            (len(indices), 150), numpy.float32
        )
        for game, events in enumerate(logs):
            selected = numpy.flatnonzero(indices[:, 0] == game)
            batch["reference_log_probability"][selected] = (
                reference_policy.distribution(
                    events,
                    numpy.column_stack(
                        (
                            indices[selected, 3] - 1,
                            indices[selected, 2],
                            features["decision_phase"][selected],
                        )
                    ),
                    batch["head"][selected],
                    batch["legal"][selected],
                )
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
    batch["features"] = features
    return batch, statistics
