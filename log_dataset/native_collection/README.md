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

### Action-conditioned win values

`action_win_value_range` has axes `[295, closed/open, ron/tsumo, min/max]`;
`action_win_value_valid` marks legal, available completions. Closed dama actions
are scored without riichi. Declaration actions and actions after accepted
riichi include it. No-yaku dama has zero ron range and a false ron-valid flag;
its menzen-tsumo completion can still be legal. One-shanten searches preserve
the action's declaration state rather than assuming a later riichi.

These fields replace `action_value_range` and `action_value_valid`, which assumed
riichi for every closed hand. Rebuild recorded datasets from source archives and
collect fresh replay before training with this contract. Old TFRecords are
rejected because the new required fields are absent; renaming old fields does
not repair their values. Old exports retain their original input contract and
must not be paired with this encoder. Warm-start weights require retraining
and evaluation with the corrected features before publication.

`magnet_win_value_range` stores the fixed reference policy's original scoring
ranges as int16 hundreds of points, with `magnet_win_value_valid` as int8.
Both use the same axes as the candidate fields. Closed-hand reference ranges
include riichi, preserving the released magnet's inputs. The learned candidate
encoder ignores these two fields; only fixed-magnet inference substitutes them
(expanding ranges by 100). The previous-policy mirror uses corrected values.

Collector requests also carry `opponent_shanten`, `opponent_ukeire` and
`opponent_hand_counts` as separate training labels in actor-relative order.
These match the recorded labels, including red-five categories and known meld
copies. They never enter public feature rows or inference inputs. Candidate
replay retains them as int8 arrays for direct belief supervision.

Requests also carry `ron_legal` (int8) and `ron_payments` (int32), each
`[3, 37]`, for opponents and discard tile codes. These hidden labels remain in the replay schema but do not contribute
to the training objective. Unknown values are
-1. Own discard/declaration/kan phases use exact simulator scoring, including
ura when available; response/win requests leave these labels unknown. Ron
amounts include honba per claimant. Triple ron is an abort, not three
payments. Liability amounts without ordinary support remain unknown.

The expert priors add five decision-time fields:
`policy_prior_no_threat` (scalar), `riichi_dama_points` (37 discards), and
`riichi_extra_han` (37 discards), `call_open_points` and
`call_riichi_points` (each 4 by 37 chi/pon continuations).
The threat gate reuses the native focused
sampling detector. Scores use the minimum across available winning-tile
variants, include visible/red dora, and exclude ura, ippatsu and honba.
Dama points are normalized to non-dealer ron; a no-yaku completion is zero.
Riichi and response decisions calculate their respective scoring facts.
Call values score the post-call hand, with hypothetical closed-riichi value
computed by concealing chi/pon melds and treating kans as concealed. These
fields bypass learned encoder projections and affect only expert CE.

Existing TFRecords require these fields; the reader does not silently fill
missing values. Amend a corpus into a separate directory with:

```sh
python -m log_dataset.amend_riichi_prior \
  --source /path/to/original/data --output /path/to/amended/data --workers 8
```

The amendment uses stored candidate hands and the existing threat metadata,
calls the same native scorer, and preserves every original field. A completed
full amendment updates the dataset schema and shard byte counts. Use the
amended corpus only after `amendment.json` reports `complete: true`.
`--replace` may regenerate an inactive derived cache from the original corpus;
it replaces shards atomically and never permits identical source/output paths.

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
[model/actions.py](../../model/actions.py). SavedModel exports expose their
input and output schemas through the `serving_default` signature.

## Checks

```sh
python -m pytest integration/big/test_native_collection.py \
  integration/big/test_feature_generation.py \
  integration/big/test_self_play.py \
  integration/big/test_selected_roots.py
```

Python reference calculations under `integration/` check native parity.
