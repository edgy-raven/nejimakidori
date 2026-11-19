"""Hand-boundary continuations retain the same settlement as full matches."""

import json

import numpy

from integration.big import test_selected_roots
from log_dataset import _native_collection
from model import actions, outcomes
from training import self_play


def test_hand_only_forks_preserve_outcomes_and_real_terminal_flags():
    collector = _native_collection.NativeCollector(1, 941)
    roots = {}
    while (request := collector.next()) is not None:
        if request["hand"][0] not in roots:
            roots[request["hand"][0]] = collector.snapshot(0)
        collector.submit(test_selected_roots.greedy(request))

    for hand in (0, max(roots)):
        full = json.loads(
            test_selected_roots.finish(roots[hand].fork(1, 941, False))[0]
        )
        short = roots[hand].fork(1, 941, False, until_hand_end=True)
        partial = json.loads(test_selected_roots.finish(short)[0])
        assert full[: len(partial)] == partial
        starts = [
            index
            for index, event in enumerate(full)
            if event["type"] == "start_kyoku"
        ]
        end = starts[hand + 1] if hand + 1 < len(starts) else len(full)
        assert partial[-1]["type"] == (
            "start_kyoku" if hand + 1 < len(starts) else "end_game"
        )
        numpy.testing.assert_array_equal(
            outcomes.terminal_payments(partial[starts[hand] :]),
            outcomes.terminal_payments(full[starts[hand] : end]),
        )
        assert short.finish()[1]["hands"] == 1
    assert len(partial) == len(full)


def test_multi_copy_hand_collection_has_aligned_root_labels():
    collector = _native_collection.NativeCollector(1, 941)
    while (request := collector.next()) is not None:
        if request["hand"][0] > 0:
            root = collector.snapshot(0)
            break
        collector.submit(test_selected_roots.greedy(request))

    def predict(features):
        return {
            name: numpy.zeros(
                (len(features), actions.ACTOR_OUTPUTS[name][0]), numpy.float32
            )
            for name in actions.TRAIN_POLICY_HEADS
        }

    branch = root.fork(2, 123, until_hand_end=True)
    batch, statistics = self_play.collect(predict, 2, 123, collector=branch)
    expected = []
    for raw in branch.finish()[0]:
        events = json.loads(raw)
        outcomes.normalize_draw_payments(events)
        starts = [
            index
            for index, event in enumerate(events)
            if event["type"] == "start_kyoku"
        ]
        completed = len(starts) - (events[-1]["type"] != "end_game")
        payments = outcomes.terminal_payments(events[starts[completed - 1] :])
        order = (request["seat"][0] + numpy.arange(4)) % 4
        expected.append(payments[:, order][:, :, order][outcomes.PAYMENT_CELLS])
    numpy.testing.assert_array_equal(
        batch["payment_points"][[0, len(request["head"])]], expected
    )
    assert "settlement" not in batch
    assert statistics["hands"] == 2
    assert numpy.isfinite(batch["payment_points"]).all()
