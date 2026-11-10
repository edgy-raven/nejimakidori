"""Dashboard metrics remain comparable across candidate phases."""

import hashlib
import json

import numpy
import pytest
import tensorboard.backend.event_processing.event_accumulator as accumulator
import tensorboard.util.tensor_util

from training import candidate_monitor


def test_candidate_progress_units_and_stable_tags(tmp_path, monkeypatch):
    clock = iter([1.0, 2.0, 4.0, 10.0, 12.0, 14.0])
    monkeypatch.setattr(
        candidate_monitor.time, "monotonic", lambda: next(clock)
    )
    monitor = candidate_monitor.Monitor(
        tmp_path,
        100,
        {"awr_temperature": 3, "mixed_only": False, "skip_distill": False},
    )
    monitor.report(stage="distill", metrics={}, exposures=0, cohort=0, games=0)
    monitor.report(
        stage="distill",
        metrics={"teacher_kl": 0.2},
        exposures=100,
        cohort=0,
        games=0,
    )
    monitor.report(
        stage="mixed-awr-1_collecting",
        metrics={},
        exposures=200,
        cohort=0,
        games=0,
    )
    monitor.report(
        stage="mixed-awr-1_updating",
        metrics={},
        exposures=200,
        cohort=0,
        games=2,
    )
    monitor.report(
        stage="mixed-awr-1",
        metrics={
            "value_loss": 0.25,
            "value_bias": -0.1,
            "payment_nll": 2.0,
            "effective_sample_fraction": 0.6,
            "weight_clipped_fraction": 0.2,
            "replay/record_fraction": 0.8,
        },
        exposures=250,
        cohort=1,
        games=2,
    )
    monitor.close()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase_index"] == 3
    assert status["phase_fraction"] == 0.5
    assert status["budget_exposures"] == 700
    assert status["window_positions_per_second"] == 25
    events = accumulator.EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    for name, expected in {
        "awr/value_rmse_st3": 50,
        "awr/value_bias_st3": -10,
        "distill/teacher_kl": 0.2,
        "replay/record_fraction": 0.8,
        "progress/total_fraction": 250 / 700,
    }.items():
        numpy.testing.assert_allclose(
            tensorboard.util.tensor_util.make_ndarray(
                events.Tensors(name)[-1].tensor_proto
            ),
            expected,
            rtol=1e-6,
        )
    assert "custom_scalars__config__" in events.Tags()["tensors"]


def test_phase_evaluation_passes_loadable_export_to_matches(
    tmp_path, monkeypatch
):
    import pathlib
    import types

    from training import candidate_evaluation

    calls = []
    weights = tmp_path / "weights.h5"
    weights.write_bytes(b"saved checkpoint")

    def run(argv, check):
        assert check
        calls.append(argv)
        if argv[2] == "model.export":
            exported = (
                pathlib.Path(argv[argv.index("--output") + 1]) / "saved_model"
            )
            exported.mkdir(parents=True)
            (exported / "saved_model.pb").write_bytes(b"fixture")
            (exported.parent / "export-verification.json").write_text(
                json.dumps(
                    {
                        "weight_sha256": hashlib.sha256(
                            weights.read_bytes()
                        ).hexdigest()
                    }
                )
            )
        else:
            exported = pathlib.Path(argv[argv.index("--new") + 1])
            assert (exported / "saved_model.pb").is_file()
            assert argv[argv.index("--seed") + 1] == "941"
            assert argv[argv.index("--batch-games") + 1] == "32"
            assert argv[argv.index("--devices") + 1] == "GPU-a,GPU-b"

    monkeypatch.setattr(candidate_evaluation.subprocess, "run", run)
    args = types.SimpleNamespace(
        output=str(tmp_path),
        weights=str(weights),
        opponent="incumbent/saved_model",
        games=512,
        seed=941,
        workers=4,
        devices="GPU-a,GPU-b",
        batch_games=32,
    )
    candidate_evaluation.run(args)
    assert len(calls) == 2
    candidate_evaluation.run(args)
    assert len(calls) == 3
    assert calls[-1][2] == "training.evaluate"
    weights.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="different weights"):
        candidate_evaluation.run(args)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "mixed_only,phase,budget",
    [(True, "mixed-awr-1", 500), (False, "record-awr", 600)],
)
def test_restart_monitor_starts_at_zero(tmp_path, mixed_only, phase, budget):
    monitor = candidate_monitor.Monitor(
        tmp_path,
        100,
        {"mixed_only": mixed_only, "skip_distill": not mixed_only},
    )
    monitor.report(stage=phase, metrics={}, exposures=0, cohort=0, games=0)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase_index"] == 1
    assert status["phase_exposures"] == status["total_exposures"] == 0
    assert status["budget_exposures"] == budget
    monitor.report(
        stage="complete", metrics={}, exposures=budget, cohort=5, games=10
    )
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["total_fraction"] == status["phase_fraction"] == 1
    monitor.close()


def test_resumed_monitor_preserves_exposures_and_elapsed_time(tmp_path):
    monitor = candidate_monitor.Monitor(
        tmp_path,
        100,
        {"mixed_only": False, "skip_distill": False},
        resume={"exposures": 100, "elapsed_seconds": 300},
    )
    monitor.report(
        stage="distill_evaluation",
        metrics={},
        exposures=100,
        cohort=0,
        games=0,
    )
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase_index"] == 1
    assert status["total_exposures"] == 100
    assert status["elapsed_seconds"] >= 300
    monitor.report(
        stage="record-awr",
        metrics={},
        exposures=101,
        cohort=0,
        games=0,
    )
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase_index"] == 2
    assert status["phase_exposures"] == 1
    assert status["budget_exposures"] == 700
    monitor.close()
