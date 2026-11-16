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

## Test organization

Keep tests around complete workflows: build records, sample replay, update the
model, save/export weights, and play a match. Use the real producer and consumer
when checking a data contract. Distributed parity runs in one isolated process
with two logical CPUs.

Retain regressions for corrupted labels, illegal actions, settlement arithmetic,
gradient boundaries, exhausted storage and worker cleanup. Avoid tests that copy
an implementation, restate configuration defaults, or count mocked helper calls.
Performance is checked by running the measured workflows in the full suite.
