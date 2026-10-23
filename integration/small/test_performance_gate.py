"""Performance checks exclude warmup and reject sustained workflow slowdowns."""

import subprocess

import pytest

import smoke
from integration import performance


def test_warmed_median_gate_rejects_regressions_and_incompatible_baselines(
    monkeypatch,
):
    clock = [0.0]
    durations = iter([50, 50, 1, 1, 10, 1, 1])

    def workload():
        clock[0] += next(durations)

    monkeypatch.setattr(performance.time, "perf_counter", lambda: clock[0])
    result = performance.measure(workload)
    assert result == {"seconds": 1, "samples": [1, 1, 10, 1, 1]}
    baseline = {"environment": {"cpu": "test"}, "workloads": {"flow": result}}
    measured = {
        **baseline,
        "workloads": {"flow": {"seconds": 1.25}},
    }
    performance.compare(baseline, measured)
    measured["workloads"]["flow"]["seconds"] = 1.26
    with pytest.raises(RuntimeError, match="Performance regression.*flow"):
        performance.compare(baseline, measured)
    with pytest.raises(ValueError, match="environment"):
        performance.compare(baseline, {**measured, "environment": {}})
    with pytest.raises(ValueError, match="workloads"):
        performance.compare(baseline, {**measured, "workloads": {}})


def test_big_propagates_performance_failure_while_small_runs_correctness(
    monkeypatch, tmp_path
):
    calls = []

    def execute(*, args, cwd, env, check):
        calls.append(args)
        assert check and env["CUDA_VISIBLE_DEVICES"] == "-1"
        if "integration.performance" in args:
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(smoke.subprocess, "run", execute)
    monkeypatch.setattr(
        smoke.sys,
        "argv",
        [
            "smoke.py",
            "big",
            "--performance-baseline",
            str(tmp_path / "cpu.json"),
        ],
    )
    with pytest.raises(subprocess.CalledProcessError):
        smoke.run()
    assert len(calls) == 1
    assert calls[0][-2:] == ["--baseline", str(tmp_path / "cpu.json")]
    calls.clear()
    monkeypatch.setattr(smoke.sys, "argv", ["smoke.py", "small"])
    smoke.run()
    assert calls
    assert all("integration.performance" not in call for call in calls)
