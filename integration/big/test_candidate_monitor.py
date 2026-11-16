"""Dashboard metrics remain comparable across candidate phases."""

import hashlib
import json

import numpy
import pytest
import tensorboard.backend.event_processing.event_accumulator as accumulator
import tensorboard.util.tensor_util
from training.candidate import monitor


def test_candidate_progress_units_and_stable_tags(tmp_path, monkeypatch):
    clock = iter([1.0, 2.0, 4.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    monkeypatch.setattr(monitor.time, "monotonic", lambda: next(clock))
    progress = monitor.Monitor(
        tmp_path,
        100,
        {"learning_rate": 2e-5},
    )
    progress.report(
        stage="record-imitation", metrics={}, exposures=0, cohort=0, games=0
    )
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase_index"] == 1
    assert status["phase_exposures"] == status["total_exposures"] == 0
    progress.report(
        stage="record-imitation",
        metrics={"policy": 0.2},
        exposures=100,
        cohort=0,
        games=0,
    )
    progress.report(
        stage="mixed-policy-1_collecting",
        metrics={},
        exposures=100,
        cohort=0,
        games=0,
    )
    progress.report(
        stage="mixed-policy-1_updating",
        metrics={},
        exposures=100,
        cohort=0,
        games=2,
    )
    progress.report(
        stage="mixed-policy-1",
        metrics={
            "belief_loss": 0.25,
            "joint_wait_brier": 0.1,
            "payment_nll": 2.0,
            "effective_sample_fraction": 0.6,
            "weight_clipped_fraction": 0.2,
            "replay/record_fraction": 0.8,
        },
        exposures=150,
        cohort=1,
        games=2,
    )
    for count in (100, 200):
        progress.report(
            stage="mixed-policy-1_q-fit",
            metrics={"q_total_exposures": count, "q/payment_nll": 1.5},
            exposures=150,
            cohort=1,
            games=2,
        )
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase_index"] == 2
    assert status["phase_fraction"] == 0.5
    assert status["budget_exposures"] == 600
    assert status["window_positions_per_second"] == 25
    events = accumulator.EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    for name, expected in {
        "policy/belief_loss": 0.25,
        "policy/joint_wait_brier": 0.1,
        "policy/policy": 0.2,
        "replay/record_fraction": 0.8,
        "progress/total_fraction": 150 / 600,
    }.items():
        numpy.testing.assert_allclose(
            tensorboard.util.tensor_util.make_ndarray(
                events.Tensors(name)[-1].tensor_proto
            ),
            expected,
            rtol=1e-6,
        )
    assert "custom_scalars__config__" in events.Tags()["tensors"]
    assert [event.step for event in events.Tensors("q/payment_nll")] == [
        100,
        200,
    ]

    progress.report(
        stage="complete", metrics={}, exposures=600, cohort=5, games=10
    )
    assert (
        json.loads((tmp_path / "status.json").read_text())["total_fraction"]
        == 1
    )
    progress.close()


def test_phase_evaluation_passes_loadable_export_to_matches(
    tmp_path, monkeypatch
):
    import pathlib
    import types

    from training.candidate import evaluation

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

    monkeypatch.setattr(evaluation.subprocess, "run", run)
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
    evaluation.run(args)
    assert len(calls) == 2
    evaluation.run(args)
    assert len(calls) == 3
    assert calls[-1][2] == "training.evaluate"
    weights.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="different weights"):
        evaluation.run(args)
    assert len(calls) == 3
