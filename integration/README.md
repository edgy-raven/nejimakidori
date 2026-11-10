# Verification

Run checks from the repository root after [setup](../SETUP.md):

```sh
python -m pip install pytest
python smoke.py small
python smoke.py big
```

## Layout

| Path | Purpose |
| --- | --- |
| `small/` | Fast rules, schema, sampling and settlement checks |
| `big/` | Native collection, model gradients, candidate updates and export workflows |
| `fixtures/model/` | Recorded game regressions |
| `*_reference.py` | Independent or test-only calculations used by the workflows |
| `archive_fixture.py` | Minimal input archives for dataset tests |
| `performance.py`, `performance_cpu.json` | Warmed CPU timings and regression baseline |

## Scope

Tests cover the model and selected candidate on `main`. Experimental objectives
and their tests stay on their experiment branches. Each workflow test should
show the supported behavior and preserve regressions that affect correctness.
The big suite includes the small suite and CPU performance gate.
