"""Performance checks exclude warmup and reject sustained workflow slowdowns."""

import pytest

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
