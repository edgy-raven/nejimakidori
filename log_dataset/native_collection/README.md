# Native features and self-play

Rust handles RiichiEnv replay, legal transitions, scoring and the Python
interface. C++ calculates board and candidate-hand features. Both recorded
positions and self-play use this engine.

## Build

Complete the [repository setup](../../README.md#setup), including the patched
RiichiEnv checkout, then run from the repository root:

```sh
python -m log_dataset.build_native_collection
```

This builds `log_dataset/_native_collection.so`. The build needs Rust, a C++17
compiler and `ar`. Rebuild after changing native code or the input layout.

## Input tensors

[src/schema.rs](src/schema.rs) defines field names, shapes, dtypes and packed
order. The build derives C++ offsets from those declarations. Inspect the
installed library's schema with:

```sh
python -c 'from log_dataset import _native_collection as n; print(n.INPUTS)'
```

Shapes exclude the batch dimension. Seats are actor-relative unless the producer
explicitly converts them. Packed rows contain consecutive fields in declaration
order, without alignment padding. `training.self_play.FEATURE_DTYPE` exposes NumPy
views of those rows.

Self-shanten fields encode actual shanten + 1: 0 is complete, 1 is tenpai,
and 2 is one away. Target samplers must account for that offset.

| Source | Responsibility |
| --- | --- |
| [src/features.rs](src/features.rs) | Public-state snapshots and legal post-call hands. |
| [cpp/features.cpp](cpp/features.cpp) | Table, river and meld features. |
| [cpp/candidate.cpp](cpp/candidate.cpp) | Candidate hands, action masks and efficiency features. |
| [cpp/yaku.cpp](cpp/yaku.cpp) | Public yaku feasibility and fixed-meld values. |
| [cpp/scoring.cpp](cpp/scoring.cpp) | Completion search, with scoring delegated to RiichiEnv. |
| [src/lib.rs](src/lib.rs) | Python bindings. |

Opponent concealed hands are excluded from feature snapshots. Sampling metadata
is optional: `FeatureReplay.pack(..., include_focus=True)` adds category,
late-placement and score-gap flags for dataset construction.

## Collector interface

`NativeCollector` runs complete games through three operations:

1. `next()` returns pending decisions and packed feature rows, or `None` when
   finished.
2. `submit(codes)` supplies one legal action per pending decision.
3. `finish()` returns completed game logs and counters.

[training/self_play.py](../../training/self_play.py) batches worker
requests for inference and retains their packed rows for training. Completed
worker results contain actions, behavior probabilities and targets.

`snapshot(decision)` captures a pending decision between `next()` and
`submit()`, including every simultaneous request in that game. The returned
root exposes `decision` (its local request offset) and
`fork(copies, seed, shuffle_remaining=True)`. Forks retain public history,
concealed hands, pending actions and prefix logs. They shuffle unrevealed wall
tiles without moving revealed dora indicators, and seed future hands from the
branch seed. With shuffling disabled, the initial engine state is identical;
subsequent collector randomness still uses the supplied seed. Repeated forks
with the same seed are reproducible. The parent collector is unchanged.
These continuations condition on the captured hidden hands rather than drawing
an information-set posterior. Python collection emits only suffix decisions.
Hand counters count completed hands, including the initial partial hand of a
fork. Riichi turn totals include declarations before the root in that hand,
matching the retained riichi flags and completed-hand denominators. Fork
statistics describe root-conditioned continuations, not ordinary match rates.

Action heads and dimensions are defined in
[model/actions.py](../../model/actions.py). A running [API](../../service/README.md)
reports its loaded export's input and output schemas at `GET /`.

## Checks

```sh
python -m pytest integration/big/test_native_collection.py \
  integration/big/test_feature_generation.py \
  integration/big/test_self_play.py \
  integration/big/test_selected_roots.py
```

Python reference calculations under `integration/` check native parity.
