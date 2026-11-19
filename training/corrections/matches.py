"""Mine seeded match trajectories while scoring roots with a frozen policy."""

import gzip
import json

import numpy

from log_dataset import _native_collection
from model import actions
from training import self_play


def discover(predict, games, capture):
    """Replay entries with path/seed/duplicate; retain only selected snapshots.

    Native replay verifies every generated event against the original log.
    Recorded choices advance discovery only. Root comparisons and subsequent
    counterfactual continuations use the supplied frozen policy instead.
    """
    collectors = []
    for game in games:
        with gzip.open(game["path"], "rt") as stream:
            events = [json.loads(line) for line in stream]
        collectors.append(
            _native_collection.NativeCollector.from_match(
                json.dumps(events), game["seed"], game["duplicate"]
            )
        )
    statistics = self_play.collection_counts()
    active = dict(enumerate(collectors))
    while active:
        requests = {}
        for game, collector in list(active.items()):
            request = collector.next()
            if request is None:
                for name, value in collector.finish()[1].items():
                    statistics[name] += value
                del active[game]
            else:
                requests[game] = request
        if not requests:
            break
        features = numpy.concatenate(
            [
                numpy.frombuffer(request["features"], self_play.FEATURE_DTYPE)
                for request in requests.values()
            ]
        )
        outputs = predict(features)
        start = 0
        for game, request in requests.items():
            count = len(request["head"])
            heads = numpy.asarray(request["head"])
            legal = numpy.frombuffer(request["legal"], numpy.bool_).reshape(
                count, 150
            )
            scores = numpy.full((count, 150), -1e9, numpy.float32)
            for head, name in enumerate(actions.TRAIN_POLICY_HEADS):
                rows = heads == head
                scores[rows, : actions.ACTOR_OUTPUTS[name][0]] = outputs[name][
                    start : start + count
                ][rows]
            scores[~legal] = -1e9
            scores -= scores.max(axis=1, keepdims=True)
            behavior = scores - numpy.log(
                numpy.exp(scores).sum(axis=1, keepdims=True)
            )
            behavior[~legal] = -1e9
            capture(
                collectors[game],
                {
                    "features": features[start : start + count],
                    "head": heads,
                    "legal": legal,
                    "behavior": behavior,
                    "indices": numpy.column_stack(
                        (
                            numpy.full(count, game, numpy.int64),
                            *[
                                request[name]
                                for name in ("hand", "seat", "event")
                            ],
                        )
                    ),
                },
            )
            collectors[game].submit_recorded()
            start += count
    return {**statistics, "logged_games": len(games)}
