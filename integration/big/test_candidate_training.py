"""Candidate passes preserve source budgets and settlement targets."""

import errno
import json
import pathlib
import types

import numpy
import pytest
import tensorflow as tf
from log_dataset import afk
from log_dataset import replay as game_replay
from log_dataset import training_data
from model import actions
from training.candidate import replay, run


def test_checkpoint_matches_use_disjoint_wall_seeds(tmp_path, monkeypatch):
    commands = []
    fitted = []
    monkeypatch.setattr(
        run.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    network = types.SimpleNamespace(save_weights=lambda path: None)
    command = [
        "python",
        "--weights",
        "{weights}",
        "--output",
        "{output}",
        "--seed",
        "941",
        "--games",
        "10000",
    ]
    stages = ["record-imitation"] + [f"mixed-policy-{i}" for i in range(1, 6)]
    for stage in stages + [stages[-1]]:
        run.finish_checkpoint(
            stage=stage,
            root=tmp_path,
            network=network,
            pools=[],
            report=lambda *args: None,
            evaluation_command=command,
            fit_critic=lambda stage: fitted.append((stage, len(commands))),
        )
    seeds = [int(argv[argv.index("--seed") + 1]) for argv in commands]
    assert seeds == [941 + i * 2500 for i in range(6)] + [13441]
    assert (
        len({seed + game // 4 for seed in seeds[:6] for game in range(10000)})
        == 15000
    )
    assert command[command.index("--seed") + 1] == "941"
    assert commands[-1] == commands[-2]
    assert fitted == [
        (stage, index + 1) for index, stage in enumerate(stages[:-1])
    ]


def test_exact_source_quotas_and_disk_replay(tmp_path):
    generator = numpy.random.default_rng(941)
    pools = [
        replay.ReplayBuffer(
            tmp_path / str(index), 13, generator, reservoir=index < 2
        )
        for index in range(4)
    ]
    for index, pool in enumerate(pools):
        for start in (0, 7, 14, 21):
            pool.add(
                {
                    "source": numpy.full(7, index, numpy.int32),
                    "row": numpy.arange(start, start + 7),
                    "features": {"source": numpy.full(7, index)},
                }
            )
        pool.flush()
        assert pool.seen == 28
        assert len(pool.arrays[0]) == 13
    assert set(range(21, 28)) <= set(pools[2].sample(100)["row"])
    assert set(pools[2].sample(100)["row"]) <= set(range(14, 28))
    for fraction in (1, 0.8, 0.6, 0.4, 0.2, 0):
        seen = numpy.zeros(4, numpy.int64)
        for start in range(0, 103, 11):
            end = min(start + 11, 103)
            quotas = replay.allocation(end, fraction, 0.4, start=start)
            batch = replay.mixed_batch(pools, quotas, generator)
            numpy.testing.assert_array_equal(
                batch["source"], batch["features"]["source"]
            )
            seen += numpy.bincount(batch["source"], minlength=4)
        numpy.testing.assert_array_equal(
            seen, replay.allocation(103, fraction, 0.4)
        )
        if fraction == 0:
            assert seen[:2].sum() == 0
        numpy.testing.assert_array_equal(
            replay.allocation(103, fraction, 0.4, start=103),
            numpy.zeros(4, numpy.int64),
        )


def test_replay_disk_exhaustion_fails_before_rows_are_written(
    tmp_path, monkeypatch
):
    pool = replay.ReplayBuffer(
        tmp_path / "replay", 13, numpy.random.default_rng(941), reservoir=False
    )

    def full_disk(fd, offset, length):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(replay.os, "posix_fallocate", full_disk)
    with pytest.raises(OSError) as error:
        pool.add({"row": numpy.arange(1, 8)})
    assert error.value.errno == errno.ENOSPC
    assert pool.seen == 0
    numpy.testing.assert_array_equal(pool.arrays[0], 0)


@pytest.mark.parametrize("reservoir", (False, True))
def test_replay_retains_selected_rows_and_nested_alignment(tmp_path, reservoir):
    pools = [
        replay.ReplayBuffer(
            tmp_path / str(index),
            13,
            numpy.random.default_rng(941),
            reservoir=reservoir,
        )
        for index in range(2)
    ]
    for start, count in ((0, 7), (100, 40)):
        values = numpy.arange(start, start + count)
        batch = {
            "row": values,
            "features": {"pair": numpy.stack([values, -values], axis=1)},
        }
        rows = numpy.flatnonzero(values % 3 != 0)[::-1]
        for pool in pools:
            pool.add(batch, rows)
            pool.add(batch, numpy.array([], dtype=int))
        first, second = [pool.sample(100) for pool in pools]
        numpy.testing.assert_equal(first, second)
        numpy.testing.assert_array_equal(
            first["features"]["pair"],
            numpy.stack([first["row"], -first["row"]], axis=1),
        )
        assert (first["row"] % 3 != 0).all()
        if not reservoir:
            assert first["row"].min() >= start
    assert pools[0].seen == 31


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
        game_replay.Replay(
            "candidate-probe", afk.retained_hands(events)[0]
        ).observations()
    )[:3]
    raw = tf.constant(
        [
            training_data.serialize(row, targets[row.kyoku_id])
            for row in observations
        ]
    )
    _, labels, _ = training_data.parse_batch(raw)
    distill = tf.reduce_max(
        tf.stack(
            [
                tf.cast(labels[name], tf.int32)
                for name in actions.TRAIN_POLICY_HEADS
            ],
            axis=1,
        ),
        axis=1,
    )
    batch = replay.decode_records(raw, distill)
    packed = replay.pack_batch(
        tf.nest.map_structure(lambda value: value.numpy(), batch),
        training_data.PackedBatch(batch["features"]),
    )
    for name in batch["beliefs"]:
        numpy.testing.assert_array_equal(packed["beliefs"][name], labels[name])
        assert packed["beliefs"][name].dtype == numpy.int8
    for name in ("ron_legal", "ron_payments"):
        numpy.testing.assert_array_equal(packed[name], labels[name])
        assert packed[name].shape == (3, 3, 37)
    numpy.testing.assert_array_equal(batch["recorded"], [True] * 3)
    numpy.testing.assert_array_equal(packed["distill"], distill)
    assert tf.reduce_all(
        tf.gather(batch["legal"], batch["chosen"], batch_dims=1)
    )
    monkeypatch.setattr(
        replay.hybrid,
        "raw_batches",
        lambda *args: tf.data.Dataset.from_tensor_slices((raw, distill)).batch(
            2
        ),
    )
    args = types.SimpleNamespace(
        distill_labels="fixture", batch_size=2, seed=941
    )
    batches = list(replay.recorded_batches(args).as_numpy_iterator())
    assert [len(batch["chosen"]) for batch in batches] == [2, 1]
    numpy.testing.assert_array_equal(
        numpy.concatenate([batch["recorded"] for batch in batches]),
        [True] * 3,
    )


def test_record_stream_consumes_rows_without_a_small_fixed_cache(monkeypatch):
    batches = [
        {"chosen": numpy.arange(start, start + 4)} for start in (0, 4, 8)
    ]
    monkeypatch.setattr(
        replay,
        "recorded_batches",
        lambda *a, **kw: types.SimpleNamespace(
            as_numpy_iterator=lambda: iter(batches)
        ),
    )
    monkeypatch.setattr(
        replay,
        "classify_records",
        lambda batch, *a: (batch["chosen"] % 2).astype(numpy.uint8),
    )
    monkeypatch.setattr(replay, "pack_batch", lambda batch, trainer: batch)
    targeted = []
    stream = replay.RecordedStream(
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
