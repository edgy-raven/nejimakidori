"""Candidate passes preserve teacher, source budgets and observed targets."""

import json
import pathlib
import types

import numpy
import pytest
import tensorflow as tf

from log_dataset import afk, replay, training_data
from model import actions
from training import candidate_training, self_play
from training import trainer as training


def test_exact_source_quotas_and_disk_replay(tmp_path):
    generator = numpy.random.default_rng(941)
    pools = [
        candidate_training.Replay(
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
            quotas = candidate_training.allocation(
                end, fraction, 0.4, start=start
            )
            numpy.testing.assert_array_equal(
                quotas,
                candidate_training.allocation(end, fraction, 0.4)
                - candidate_training.allocation(start, fraction, 0.4),
            )
            batch = candidate_training.mixed_batch(pools, quotas, generator)
            seen += numpy.bincount(batch["source"], minlength=4)
        numpy.testing.assert_array_equal(
            seen, candidate_training.allocation(103, fraction, 0.4)
        )
        if fraction == 0:
            assert seen[:2].sum() == 0
        numpy.testing.assert_array_equal(
            candidate_training.allocation(103, fraction, 0.4, start=103),
            numpy.zeros(4, numpy.int64),
        )


@pytest.mark.parametrize("reservoir", (False, True))
def test_replay_selected_rows_match_stream_replacement(tmp_path, reservoir):
    pool = candidate_training.Replay(
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
        actual = candidate_training.mixed_batch(pools, counts, generator)
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
    batch = candidate_training.record_batch(raw)
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

    fingerprints = tf.raw_ops.Fingerprint(data=raw, method="farmhash64").numpy()
    companion = []
    for index, fingerprint in enumerate(fingerprints):
        example = tf.train.Example()
        example.features.feature["fingerprint"].bytes_list.value.append(
            fingerprint.tobytes()
        )
        example.features.feature["legacy"].bytes_list.value.append(
            numpy.array([index], numpy.float32).tobytes()
        )
        companion.append(example.SerializeToString())
    monkeypatch.setattr(
        training_data,
        "dataset",
        lambda **kwargs: tf.data.Dataset.from_tensor_slices(
            {"record": raw, "teacher": companion}
        ).batch(2),
    )
    args = types.SimpleNamespace(
        data="fixture", batch_size=2, seed=941, teacher_features="companion"
    )
    fields = {"legacy": {"shape": [1], "dtype": "float32"}}
    batches = list(
        candidate_training.record_stream(
            args, teacher_fields=fields
        ).as_numpy_iterator()
    )
    assert [len(batch["chosen"]) for batch in batches] == [2, 1]
    numpy.testing.assert_array_equal(
        numpy.concatenate([batch["features"]["legacy"] for batch in batches]),
        [[0], [1], [2]],
    )
    companion.reverse()
    with pytest.raises(tf.errors.InvalidArgumentError, match="alignment"):
        list(
            candidate_training.record_stream(
                args, teacher_fields=fields
            ).as_numpy_iterator()
        )


def test_distillation_changes_only_student_and_ignores_padding():
    class Actor(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.logits = self.add_weight(shape=(2,), initializer="zeros")

        def call(self, features, training=False):
            assert set(features) == {"x"}
            return {
                name: tf.broadcast_to(
                    self.logits, [tf.shape(features["x"])[0], 2]
                )
                for name in actions.TRAIN_POLICY_HEADS
            }

    strategy = tf.distribute.MirroredStrategy(devices=["/CPU:0"])
    with strategy.scope():
        student = Actor()
        student({"x": tf.zeros((1, 1))})
        teacher_logits = tf.Variable([2.0, -2.0])

        @tf.function
        def teacher(**features):
            return {
                name: tf.broadcast_to(
                    teacher_logits, [tf.shape(features["x"])[0], 2]
                )
                + features["legacy"] * tf.constant([[1.0, -1.0]])
                for name in actions.TRAIN_POLICY_HEADS
            }

        trainer = types.SimpleNamespace(
            trainer=types.SimpleNamespace(
                network=types.SimpleNamespace(actor=student)
            ),
            strategy=strategy,
            feature_packing=training_data.PackedBatch({"x": tf.zeros((1, 1))}),
        )
        trainer.distribute = (
            lambda batch: training.DistributedTrainer.distribute(trainer, batch)
        )
        distiller = candidate_training.Distiller(
            trainer,
            teacher.get_concrete_function(
                x=tf.TensorSpec([None, 1], tf.float32),
                legacy=tf.TensorSpec([None, 1], tf.float32),
            ),
            0.01,
        )
    batch = {
        "features": distiller.packing.pack_numpy(
            {
                "x": numpy.zeros((4, 1), numpy.float32),
                "legacy": numpy.ones((4, 1), numpy.float32),
            }
        ),
        "head": numpy.zeros(4, numpy.int32),
        "legal": numpy.tile(numpy.arange(150) < 2, (4, 1)),
        "chosen": numpy.zeros(4, numpy.int32),
    }
    first = float(distiller.train(batch, 1))
    assert float(student.logits[0]) > 0
    assert float(distiller.train(batch, 1)) < first
    numpy.testing.assert_array_equal(teacher_logits, [2, -2])
    assert int(distiller.optimizer.iterations.numpy()) == 2


def test_record_stream_consumes_rows_without_a_small_fixed_cache(monkeypatch):
    batches = [
        {"chosen": numpy.arange(start, start + 4)} for start in (0, 4, 8)
    ]
    monkeypatch.setattr(
        candidate_training,
        "record_stream",
        lambda *a, **kw: types.SimpleNamespace(
            as_numpy_iterator=lambda: iter(batches)
        ),
    )
    monkeypatch.setattr(
        candidate_training,
        "classify_records",
        lambda batch, *a: (batch["chosen"] % 2).astype(numpy.uint8),
    )
    monkeypatch.setattr(
        candidate_training, "prepare", lambda batch, trainer: batch
    )
    targeted = []
    stream = candidate_training.RecordStream(
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


def test_mixed_only_restart_accepts_capped_ramp_without_teacher(tmp_path):
    command = tmp_path / "evaluation.json"
    command.write_text(json.dumps(["evaluate", "{weights}", "{output}"]))
    args = candidate_training.argument_parser().parse_args(
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
    fractions, _ = candidate_training.settings(args)
    assert fractions == [0.8, 0.6, 0.4, 0.2, 0.2]
    assert args.teacher is None
    args.awr_pull = 0.3
    candidate_training.settings(args)
    for coefficient in (-0.1, float("nan"), float("inf")):
        args.awr_pull = coefficient
        with pytest.raises(ValueError, match="pull"):
            candidate_training.settings(args)
    args.awr_pull = 0.3
    args.mixed_only = False
    with pytest.raises(ValueError, match="Distillation requires"):
        candidate_training.settings(args)
    args.skip_distill = True
    candidate_training.settings(args)
    assert candidate_training.restore_distillation(args, 100) == {
        "exposures": 0,
        "cohort": 0,
        "games": 0,
        "elapsed_seconds": 0,
    }
    args.skip_distill = False
    args.resume_after_distill = True
    candidate_training.settings(args)


def test_distillation_boundary_resume_preserves_progress(tmp_path, monkeypatch):
    args = types.SimpleNamespace(
        output=str(tmp_path),
        initial=str(tmp_path / "saved.weights.h5"),
        resume_after_distill=True,
    )
    pathlib.Path(args.initial).write_bytes(b"saved checkpoint")
    (tmp_path / "distill.weights.h5").write_bytes(b"saved checkpoint")
    for name in ("record-targeted", "generated-natural", "generated-targeted"):
        (tmp_path / "replay" / name).mkdir(parents=True)
    progress = dict(
        stage="record-awr",
        exposures=100,
        cohort=0,
        games=0,
        elapsed_seconds=300,
        updated_at=1100,
    )
    (tmp_path / "status.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="distillation boundary"):
        candidate_training.restore_distillation(args, 100)
    progress["stage"] = "distill_evaluation"
    (tmp_path / "status.json").write_text(json.dumps(progress))
    monkeypatch.setattr(candidate_training.time, "time", lambda: 1200)
    restored = candidate_training.restore_distillation(args, 100)
    assert restored["exposures"] == 100
    assert restored["elapsed_seconds"] == 400
    assert not list((tmp_path / "replay").iterdir())
    calls, reports = [], []
    monkeypatch.setattr(
        candidate_training.subprocess,
        "run",
        lambda argv, check: calls.append(argv),
    )
    candidate_training.finish_checkpoint(
        stage="distill",
        root=tmp_path,
        network=object(),
        pools=(),
        report=lambda stage, metrics: reports.append(stage),
        evaluation_command=["evaluate", "{weights}", "{output}"],
        save=False,
    )
    assert calls == [
        [
            "evaluate",
            str(tmp_path / "distill.weights.h5"),
            str(tmp_path / "distill-evaluation"),
        ]
    ]
    assert reports == ["distill_evaluation", "distill_complete"]
    args.resume_after_distill = False
    assert candidate_training.restore_distillation(args, 100) == dict(
        exposures=0, cohort=0, games=0, elapsed_seconds=0
    )


def test_cohort_checkpoints_are_immutable_and_latest_tracks_progress(tmp_path):
    network = types.SimpleNamespace(weights=b"first cohort")
    network.save_weights = lambda path: path.write_bytes(network.weights)
    candidate_training.save_cohort(network, tmp_path, 100)
    first = tmp_path / "checkpoints/000000000100.weights.h5"
    assert (tmp_path / "latest.weights.h5").resolve() == first
    network.weights = b"second cohort"
    candidate_training.save_cohort(network, tmp_path, 200)
    assert first.read_bytes() == b"first cohort"
    assert (tmp_path / "latest.weights.h5").read_bytes() == b"second cohort"
    with pytest.raises(FileExistsError):
        candidate_training.save_cohort(network, tmp_path, 100)
    assert first.read_bytes() == b"first cohort"
    assert (tmp_path / "latest.weights.h5").read_bytes() == b"second cohort"


@pytest.mark.parametrize(
    "mixed_only,skip_distill", [(False, False), (True, False), (False, True)]
)
def test_awr_rate_ramps_across_passes_without_counting_distillation(
    mixed_only, skip_distill
):
    args = types.SimpleNamespace(
        mixed_only=mixed_only,
        skip_distill=skip_distill,
        awr_learning_rate=1e-5,
        awr_final_learning_rate=3e-6,
    )
    trainer = types.SimpleNamespace(
        awr_optimizer=types.SimpleNamespace(learning_rate=tf.Variable(0.0))
    )
    start = 0 if mixed_only or skip_distill else 100
    budget = 500 if mixed_only else 600
    for fraction, expected in ((0, 1e-5), (0.5, 6.5e-6), (1, 3e-6)):
        candidate_training.set_awr_learning_rate(
            trainer, args, start + fraction * budget, 100
        )
        assert float(trainer.awr_optimizer.learning_rate) == pytest.approx(
            expected
        )
