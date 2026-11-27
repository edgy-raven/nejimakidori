# Verification

Run checks from the repository root after [setup](../SETUP.md):

```sh
python -m pip install pytest
python smoke.py small
python smoke.py big
```

The small suite checks rules, sampling and settlement arithmetic. The big
suite also runs native collection, feature packing, model gradients,
training, checkpoint and export workflows. Each test file runs in a separate
CPU process so TensorFlow releases its graphs between workflows.

| Path | Purpose |
| --- | --- |
| `small/` | Rules and numerical contracts |
| `big/` | Dataset, model and training workflows |
| `fixtures/model/` | Game events for regression tests |
| `*_reference.py` | Independent calculations and recorded-policy fixtures |
| `archive_fixture.py` | Input archives for dataset tests |
| `performance.py` | Optional CPU performance comparison |

## Performance

Record a baseline on a known-good checkout, then compare on the same machine
and software environment. Store the measurements outside the repository:

```sh
python -m integration.performance \
  --baseline /tmp/nejimakidori-performance.json --record-baseline
python -m integration.performance \
  --baseline /tmp/nejimakidori-performance.json
```

The comparison measures warmed feature extraction, native collection and
batched inference. It fails when a median exceeds its baseline by more than
25%. To include this check before the full correctness suite, pass
`--performance-baseline /tmp/nejimakidori-performance.json` to
`python smoke.py big`.

## Adding tests

Exercise a complete public behavior with representative inputs and visible
outputs. Use the real producer and consumer for data contracts. Preserve
regressions for illegal actions, corrupted labels, settlement arithmetic,
gradient boundaries and worker cleanup. Reference calculations should be
independent of the implementation they verify.
