"""Persistent CPU workers preserve complete games and deterministic cohorts."""

import json
import pathlib
import types

import numpy
import pytest
import tensorflow as tf

from log_dataset import _native_collection
from model import actions, joint_settlement, outcomes, self_play


def uniform_predict(features):
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


def test_native_nagashi_targets_charge_deposits_once(monkeypatch):
    recorded = json.loads(
        pathlib.Path(
            "integration/fixtures/model/nagashi_deposit.json"
        ).read_text()
    )
    starts = [
        index
        for index, event in enumerate(recorded["events"])
        if event["type"] == "start_kyoku"
    ]
    events = recorded["events"][: starts[1]] + [{"type": "end_game"}]
    request = iter(
        [
            {
                "features": numpy.zeros(4, self_play.FEATURE_DTYPE).tobytes(),
                "head": [0] * 4,
                "legal": numpy.tile(numpy.arange(150) == 0, (4, 1)).tobytes(),
                "game": [0] * 4,
                "hand": [0] * 4,
                "seat": list(range(4)),
                "event": [starts[0] + 1] * 4,
            },
            None,
        ]
    )
    monkeypatch.setattr(
        _native_collection,
        "NativeCollector",
        lambda games, seed: types.SimpleNamespace(
            next=lambda: next(request),
            submit=lambda choices: None,
            finish=lambda: ([json.dumps(events)], {}),
        ),
    )
    batch, _ = self_play.collect(uniform_predict, 1, 0)
    labels = joint_settlement.payment_labels(
        batch["payment_points"], {"pot": tf.zeros([4])}
    )
    numpy.testing.assert_array_equal(
        tf.gather(
            joint_settlement.payment_values(labels),
            labels["payment"],
            axis=1,
            batch_dims=1,
        ),
        batch["payment_points"],
    )
    for seat in range(4):
        numpy.testing.assert_array_equal(
            batch["settlement"][seat, :4],
            numpy.roll(recorded["scores"], -seat),
        )
        assert batch["settlement"][seat, 11] == 1000
    payments = numpy.zeros((4, 4, 4), numpy.int32)
    payments[2, :, 2] = [2000, 2000, 0, 4000]
    payments[3, 3, 3] = -1000
    numpy.testing.assert_array_equal(
        batch["payment_points"][0], payments[outcomes.PAYMENT_CELLS]
    )
    # Stacked yakuman remain in the declared monetary support.
    for count in range(1, 5):
        outcomes.payment_classes(
            numpy.array([8000, 16000, 32000, 48000]) * count
        )
    with pytest.raises(tf.errors.InvalidArgumentError, match="Unsupported"):
        joint_settlement.payment_labels(
            tf.constant([[5000]]), {"pot": tf.zeros([1])}
        )


def test_parallel_complete_games_are_repeatable_and_workers_are_reaped():
    with self_play.Collector(uniform_predict, 4) as collector:
        first = collector.collect(2, 12)
        first_counts = {
            name: collector.metrics[name]
            for name in self_play.collection_counts()
        }
        assert first_counts["games"] == 2
        assert first_counts["hands"] >= 2
        assert collector.metrics["inference_rows"] == len(first["chosen"])
        assert collector.metrics["inference_seconds"] > 0
        assert collector.metrics["inference_calls"] > 0
        assert len(collector.processes) == 2
        processes = list(collector.processes)
        for process in processes:
            environment = pathlib.Path(
                f"/proc/{process.pid}/environ"
            ).read_bytes()
            assert b"CUDA_VISIBLE_DEVICES=\x00" in environment
            assert b"TF_NUM_INTRAOP_THREADS=1\x00" in environment
            assert (
                "libtensorflow"
                not in pathlib.Path(f"/proc/{process.pid}/maps").read_text()
            )
        tf.nest.map_structure(
            numpy.testing.assert_array_equal,
            first,
            collector.collect(2, 12),
        )
        assert collector.processes == processes
        with self_play.Collector(uniform_predict, 1) as serial:
            expected = [serial.collect(1, seed) for seed in (12, 13)]
            assert serial.processes == []
            for name, value in first_counts.items():
                assert serial.metrics[name] == value
                assert collector.metrics[name] == 2 * value
        tf.nest.map_structure(
            numpy.testing.assert_array_equal,
            first,
            tf.nest.map_structure(
                lambda *arrays: numpy.concatenate(arrays), *expected
            ),
        )
        probability = numpy.exp(first["behavior"])
        numpy.testing.assert_allclose(probability.sum(-1), 1, atol=1e-6)
        assert numpy.all(probability[~first["legal"]] == 0)
        assert numpy.all(
            first["legal"][numpy.arange(len(first["chosen"])), first["chosen"]]
        )
        assert len(numpy.unique(first["settlement"], axis=0)) > 1
        # A dead persistent worker must fail promptly on the next cohort.
        processes[0].kill()
        processes[0].wait()
        with pytest.raises((EOFError, BrokenPipeError, ConnectionResetError)):
            collector.collect(1, 14)
    assert all(process.poll() is not None for process in processes)
    assert processes[1].returncode == 0


def test_inference_failure_reaps_workers_waiting_for_outputs():
    def fail_predict(features):
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        with self_play.Collector(fail_predict, 2) as collector:
            collector.collect(1, 12)
    assert len(collector.processes) == 1
    assert collector.processes[0].poll() is not None


def test_collection_rejects_empty_pool_and_empty_cohort():
    with pytest.raises(ValueError, match="workers must be positive"):
        self_play.Collector(uniform_predict, 0)
    with self_play.Collector(uniform_predict, 2) as collector:
        with pytest.raises(ValueError, match="games must be positive"):
            collector.collect(0, 12)
        assert collector.processes == []
