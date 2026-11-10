"""Candidate AWR passes preserve source budgets and observed targets."""

import json
import pathlib
import types

import numpy
import pytest
import tensorflow as tf

from log_dataset import afk, replay, training_data
from training import self_play
from training.candidate import run


def test_exact_source_quotas_and_disk_replay(tmp_path):
    generator = numpy.random.default_rng(941)
    pools = [
        run.Replay(tmp_path / str(index), 13, generator, reservoir=index < 2)
        for index in range(4)
    ]
    for index, pool in enumerate(pools):
        for start in (0, 7, 14, 21):
            pool.add(
                {
                    "source": numpy.full(7, index, numpy.int32),
                    "row": numpy.arange(start, start + 7),
                }
            )
        pool.flush()
        assert pool.seen == 28
        assert len(pool.arrays[0]) == 13
    assert set(range(21, 28)) <= set(pools[2].arrays[0])
    assert set(pools[2].arrays[0]) <= set(range(14, 28))
    for fraction in (1, 0.8, 0.6, 0.4, 0.2, 0):
        seen = numpy.zeros(4, numpy.int64)
        for start in range(0, 103, 11):
            end = min(start + 11, 103)
            quotas = run.allocation(end, fraction, 0.4, start=start)
            numpy.testing.assert_array_equal(
                quotas,
                run.allocation(end, fraction, 0.4)
                - run.allocation(start, fraction, 0.4),
            )
            batch = run.mixed_batch(pools, quotas, generator)
            seen += numpy.bincount(batch["source"], minlength=4)
        numpy.testing.assert_array_equal(
            seen, run.allocation(103, fraction, 0.4)
        )
        if fraction == 0:
            assert seen[:2].sum() == 0
        numpy.testing.assert_array_equal(
            run.allocation(103, fraction, 0.4, start=103),
            numpy.zeros(4, numpy.int64),
        )


@pytest.mark.parametrize("reservoir", (False, True))
def test_replay_selected_rows_match_stream_replacement(tmp_path, reservoir):
    pool = run.Replay(
        tmp_path / "replay",
        1031,
        numpy.random.default_rng(941),
        reservoir=reservoir,
    )
    generator = numpy.random.default_rng(941)
    expected = numpy.full(1031, -1, numpy.int64)
    seen = 0
    for step, count in enumerate((700, 900, 1200, 3300)):
        values = numpy.arange(count, dtype=numpy.int64) + step * 10000
        batch = {
            "row": values,
            "features": {
                "pair": numpy.stack((values, -values), axis=1),
                "flag": values % 2 == 0,
            },
        }
        rows = (
            None
            if step == 0
            else (
                numpy.empty(0, numpy.int64)
                if step == 1
                else numpy.flatnonzero(values % 3 != 0)[::-1]
            )
        )
        retained = values if rows is None else values[rows]
        if reservoir:
            slots = generator.integers(
                numpy.arange(seen, seen + len(retained)) + 1
            )
            for offset, value in enumerate(retained):
                slot = seen + offset if seen + offset < 1031 else slots[offset]
                if slot < 1031:
                    expected[slot] = value
        else:
            for offset, value in enumerate(
                retained[generator.permutation(len(retained))]
            ):
                expected[(seen + offset) % 1031] = value
        seen += len(retained)
        pool.add(batch, rows)
        assert pool.seen == seen
        assert (
            pool.generator.bit_generator.state == generator.bit_generator.state
        )
        actual = tf.nest.pack_sequence_as(
            pool.structure, [value[: min(seen, 1031)] for value in pool.arrays]
        )
        numpy.testing.assert_equal(
            actual,
            {
                "row": expected[: min(seen, 1031)],
                "features": {
                    "pair": numpy.stack(
                        (
                            expected[: min(seen, 1031)],
                            -expected[: min(seen, 1031)],
                        ),
                        axis=1,
                    ),
                    "flag": expected[: min(seen, 1031)] % 2 == 0,
                },
            },
        )


def test_mixed_replay_preserves_shuffle_alignment_and_rng():
    class Pool:
        def __init__(self, offset):
            rows = numpy.arange(10, dtype=numpy.int32) + offset
            self.batch = {
                "chosen": rows,
                "features": {
                    "pair": numpy.stack((rows, -rows), axis=1),
                    "flag": rows % 2 == 0,
                },
            }

        def sample(self, count):
            return tf.nest.map_structure(
                lambda value: value[:count], self.batch
            )

    pools = [Pool(0), None, Pool(100), Pool(200)]
    for counts in ([3, 0, 5, 2], [0, 0, 7, 0]):
        generator = numpy.random.default_rng(941)
        reference = numpy.random.default_rng(941)
        parts = [
            pool.sample(count) for pool, count in zip(pools, counts) if count
        ]
        selected = reference.permutation(sum(counts))
        expected = tf.nest.map_structure(
            lambda *values: numpy.concatenate(values)[selected], *parts
        )
        actual = run.mixed_batch(pools, counts, generator)
        numpy.testing.assert_equal(actual, expected)
        assert generator.bit_generator.state == reference.bit_generator.state
        actual["features"]["pair"][:] = 0
        numpy.testing.assert_array_equal(
            parts[0]["features"]["pair"][:, 0], parts[0]["chosen"]
        )


def test_record_adapter_keeps_legality_return_and_target_categories(
    monkeypatch,
):
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    targets = training_data.game_targets(events + [{"type": "end_game"}])
    observations = list(
        replay.Replay(
            "candidate-probe", afk.retained_hands(events)[0]
        ).observations()
    )[:3]
    raw = tf.constant(
        [
            training_data.serialize(row, targets[row.kyoku_id])
            for row in observations
        ]
    )
    batch = run.record_batch(raw)
    expected = training_data.parse_batch(raw)
    numpy.testing.assert_allclose(batch["returns"], expected[2]["returns"])
    assert tf.reduce_all(
        tf.gather(batch["legal"], batch["chosen"], batch_dims=1)
    )
    features = {
        name: numpy.array([2, 1, 3, 3])
        for name in (
            "normal_self_shanten",
            "chiitoitsu_self_shanten",
            "kokushi_self_shanten",
        )
    }
    features["live_wall_count"] = numpy.array([16, 30, 30, 30])
    numpy.testing.assert_array_equal(
        self_play.target_categories(
            features, numpy.array([0, 1, 2, 3]), numpy.array([0, 0, 0, 4])
        ),
        [1, 2, 4, 12],
    )

    monkeypatch.setattr(
        training_data,
        "dataset",
        lambda **kwargs: tf.data.Dataset.from_tensor_slices(raw).batch(2),
    )
    args = types.SimpleNamespace(data="fixture", batch_size=2, seed=941)
    batches = list(run.record_stream(args).as_numpy_iterator())
    assert [len(batch["chosen"]) for batch in batches] == [2, 1]
    numpy.testing.assert_array_equal(
        numpy.concatenate([batch["returns"] for batch in batches]),
        expected[2]["returns"],
    )


def test_record_stream_consumes_rows_without_a_small_fixed_cache(monkeypatch):
    batches = [
        {"chosen": numpy.arange(start, start + 4)} for start in (0, 4, 8)
    ]
    monkeypatch.setattr(
        run,
        "record_stream",
        lambda *a, **kw: types.SimpleNamespace(
            as_numpy_iterator=lambda: iter(batches)
        ),
    )
    monkeypatch.setattr(
        run,
        "classify_records",
        lambda batch, *a: (batch["chosen"] % 2).astype(numpy.uint8),
    )
    monkeypatch.setattr(run, "prepare", lambda batch, trainer: batch)
    targeted = []
    stream = run.RecordStream(
        None,
        types.SimpleNamespace(feature_packing=None),
        types.SimpleNamespace(
            add=lambda batch, rows: targeted.extend(batch["chosen"][rows])
        ),
        None,
        numpy.random.default_rng(941),
    )
    seen = numpy.concatenate(
        [stream.sample(count)["chosen"] for count in (3, 6, 3)]
    )
    numpy.testing.assert_array_equal(seen, numpy.arange(12))
    assert targeted == list(range(1, 12, 2))


def test_recorded_and_mixed_restarts_accept_capped_ramp(tmp_path):
    command = tmp_path / "evaluation.json"
    command.write_text(json.dumps(["evaluate", "{weights}", "{output}"]))
    args = run.argument_parser().parse_args(
        [
            "--initial",
            "awr4.weights.h5",
            "--data",
            str(tmp_path),
            "--output",
            str(tmp_path / "run"),
            "--evaluation-command",
            str(command),
            "--collection-device",
            "GPU-test",
            "--mixed-only",
            "--record-fractions",
            "0.8,0.6,0.4,0.2,0.2",
        ]
    )
    fractions, _ = run.settings(args)
    assert fractions == [0.8, 0.6, 0.4, 0.2, 0.2]
    args.awr_pull = 0.3
    with pytest.raises(ValueError, match="fixed --awr-reference"):
        run.settings(args)
    args.awr_reference = "reference.weights.h5"
    args.awr_q_std = 1.0
    run.settings(args)
    for penalty in (-1.0, float("nan"), float("inf")):
        args.awr_q_std = penalty
        with pytest.raises(ValueError, match="standard-deviation"):
            run.settings(args)
    args.awr_q_std = 1.0
    for coefficient in (-0.1, float("nan"), float("inf")):
        args.awr_pull = coefficient
        with pytest.raises(ValueError, match="pull"):
            run.settings(args)
    args.awr_pull = 0.3
    args.mixed_only = False
    run.settings(args)


def test_cohort_checkpoints_are_immutable_and_latest_tracks_progress(tmp_path):
    network = types.SimpleNamespace(weights=b"first cohort")
    network.save_weights = lambda path: path.write_bytes(network.weights)
    run.save_cohort(network, tmp_path, 100)
    first = tmp_path / "checkpoints/000000000100.weights.h5"
    assert (tmp_path / "latest.weights.h5").resolve() == first
    network.weights = b"second cohort"
    run.save_cohort(network, tmp_path, 200)
    assert first.read_bytes() == b"first cohort"
    assert (tmp_path / "latest.weights.h5").read_bytes() == b"second cohort"
    with pytest.raises(FileExistsError):
        run.save_cohort(network, tmp_path, 100)
    assert first.read_bytes() == b"first cohort"
    assert (tmp_path / "latest.weights.h5").read_bytes() == b"second cohort"


@pytest.mark.parametrize("mixed_only", [False, True])
def test_awr_rate_is_constant_within_each_pass(mixed_only):
    args = types.SimpleNamespace(
        mixed_only=mixed_only,
        awr_learning_rate=5e-6,
        awr_final_learning_rate=3e-6,
        record_fractions="0.8,0.6,0.4,0.2,0.2",
    )
    trainer = types.SimpleNamespace(
        awr_optimizer=types.SimpleNamespace(learning_rate=tf.Variable(0.0))
    )
    rates = (
        [5e-6, 13e-6 / 3, 11e-6 / 3, 3e-6, 3e-6]
        if mixed_only
        else [5e-6, 4.5e-6, 4e-6, 3.5e-6, 3e-6, 3e-6]
    )
    for phase, expected in enumerate(rates):
        for position in (0, 50, 99):
            run.set_awr_learning_rate(
                trainer, args, phase * 100 + position, 100
            )
            assert float(trainer.awr_optimizer.learning_rate) == pytest.approx(
                expected
            )
