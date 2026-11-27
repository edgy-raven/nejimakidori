"""Deterministic policies for replay alignment and native collection checks."""

import numpy

from log_dataset import call_states
from model import actions


class Policy:
    provenance = {"files_sha256": {"policy": "fixture"}}

    def __init__(self):
        self.requests = []

    def distribution_many(self, items):
        return numpy.concatenate([self.distribution(*item) for item in items])

    def distribution(self, events, requests, heads, legal):
        requests = numpy.asarray(requests, numpy.int64)
        assert requests.shape == (len(heads), 3)
        assert (requests[:, 0] >= 0).all()
        assert (requests[:, 0] < len(events)).all()
        self.requests.extend(requests.tolist())
        logits = numpy.where(
            legal,
            (requests[:, :1] + requests[:, 1:2] + 1)
            * numpy.arange(150)[None, :]
            / 1000,
            -1e9,
        )
        return (
            logits - numpy.logaddexp.reduce(logits, axis=1, keepdims=True)
        ).astype(numpy.float32)


def uniform_logits(features):
    return {
        name: numpy.zeros(
            (
                len(features["decision_phase"]),
                actions.ACTOR_OUTPUTS[name][0],
            ),
            numpy.float32,
        )
        for name in actions.TRAIN_POLICY_HEADS
    }


def greedy(request):
    rows = numpy.frombuffer(request["features"], call_states.ROW_DTYPE)
    legal = numpy.frombuffer(request["legal"], numpy.bool_).reshape(-1, 150)
    scores = numpy.zeros_like(legal, numpy.float32)
    scores[:, :37] = (
        -100 * rows["discard_action_all_shanten"].astype(numpy.float32)
        + rows["discard_action_all_ukeire_count"]
    )
    scores[numpy.asarray(request["head"]) != 0] = numpy.arange(150)
    return numpy.where(legal, scores, -1e9).argmax(1).tolist()


def finish(collector):
    while (request := collector.next()) is not None:
        collector.submit(greedy(request))
    return collector.finish()[0]
