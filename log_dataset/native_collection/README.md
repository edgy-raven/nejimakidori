# Native features and self-play

Rust handles RiichiEnv replay, legal transitions, scoring and the Python
interface. C++ calculates board and candidate-hand features. Recorded
positions and self-play use the same engine.

## Build

Complete the [repository setup](../../SETUP.md), including the patched
RiichiEnv checkout, then run from the repository root:

```sh
python -m log_dataset.build_native_collection
```

This builds `log_dataset/_native_collection.so` using Rust, a C++17 compiler
and `ar`. Rebuild after changing native code or the input layout.

## Human hand continuations

`NativeCollector.from_hand(events_json, seed)` accepts one complete hand from
`start_kyoku` through `end_kyoku`. It reconstructs a physical wall compatible
with initial hands, observed draws, kans and revealed dora. Unknown slots are
filled without replacement. Recorded decision events are checked during
replay; archive annotations and terminal scoring detail are not required to
match simulator serialization. `from_match` retains exact seeded-match replay
validation.

At a pending request, `snapshot(decision, phase=...)` can preserve a prepared
record's decision phase, including an ordinary discard after declining riichi.
Human restoration also exposes post-call discards, preserves recorded meld
ordering, and consumes kan-dora reveals at the prepared-record boundary.
Forking with `shuffle_remaining=True` reshuffles all unseen tiles, including
unrevealed dora/ura, while preserving root hands and revealed indicators.
Dataset preparation verifies restored feature parity before saving roots.
`NativeRoot.dumps()` and `NativeRoot.loads(bytes)` preserve the complete pending
state, including simultaneous requests, legal choices, public history, furiten,
wall and recorded hands. Loading does not replay events. The dataset's
`root-states/` store compresses these snapshots by hand; training loads and forks
them directly. Serialized rows require the matching packed input schema.

## Input tensors

[src/schema.rs](src/schema.rs) defines names, shapes, dtypes and packed order.
The build derives C++ offsets from those declarations. Inspect the installed
library's schema with:

```sh
python -c 'from log_dataset import _native_collection as n; print(n.INPUTS)'
```

Shapes exclude the batch dimension. A row contains 101 tensors, 6,428 scalar
values and 8,958 bytes. Fields are consecutive without alignment padding;
`log_dataset.call_states.ROW_DTYPE` provides NumPy views of packed rows.
Opponent concealed hands are excluded from feature snapshots.

Seats are actor-relative except `scores[4]` and `ranks[4]`, which use current
seat-wind order: East, South, West, North. Scores are raw int32 points; ranks
are int8 values 1–4. Equal scores use starting-dealer seat priority.
`prevailing_wind` and signed `hands_remaining` describe South-game progress:
East 1 is 7, South 4 is 0 and West 1 is −1. The latter is
`7 - (4 * round_wind + zero_based_hand_number)`, including extensions.
Self-shanten family categories encode shanten + 1, capped at 4; disabled
special families use 5. Use `self_efficiency[0] * 8` for uncapped current
shanten. Candidate discard/call shanten fields encode actual shanten + 1:
0 is complete and 1 is tenpai.

### Tiles and resulting hands

Public tile codes are 0–33 for base identities and 34–36 for red fives.
`meld_tiles[4,4,4]` and `last_tedashi_tile_id[4]` use 37 for padding. A meld
slot is occupied when its first tile is below 37. River fields carry their
own validity masks and separate base identity from red-five flags.

Each state has a 16-slot local hand bank: the current hand, ordinary-discard
hands and kan hands. `discard_candidate_hand` and `kan_candidate_hand` are
gather indices into that bank. Fixed melds use `candidate_meld_tiles[16,4]`
(base identities, padding 34), `candidate_meld_types[16,4]` (kinds 0–4), and
`candidate_meld_red[16,3]` (red-five counts). `candidate_fu[16]` stores only
the fixed-meld fu contribution.

`candidate_hand_yaku_possibility[16,19]` describes coarse feasibility from
fixed melds and public tile availability. These flags do not estimate
readiness or identify the eventual winning yaku. Bank deduplication includes
the flags as well as tile and meld contents. Opponent feasibility uses public
context. [Model architecture](../../model/ARCHITECTURE.md) describes semantic
decoding and learned feature ownership.

### Joint calls and discards

Chi and pon share four options: chi patterns 0–2 and pon at 3.
`call_discard_legal`, `call_discard_shanten` and `call_discard_ukeire_count`
have shape `[4,37]`; `call_efficiency` has shape `[4,3]`. Each option's legal
discard mask follows its executed call and preserves kuikae restrictions.

`FeatureReplay.pack(actor, phase)` returns field buffers plus `post_call`:
for each legal chi/pon, one option byte followed by the full feature row
with the meld formed and the follow-up discard pending. These states have
their own local hand banks. Legal calls accompanying a ron decision remain
available if the actor declines the win. Open kan has no immediate discard
state. Collector requests expose these sparse rows as `post_call_features`,
`post_call_owner` and `post_call_option`; [call_states.py](../call_states.py)
assembles model inputs. The exported signature has 101 base tensors, 101
accepted-call tensors, and owner and option vectors.

### Action-conditioned win values

Canonical action identity uses 295 slots. Scoring transport stores 110 rows:

| Scoring rows | Canonical actions |
| --- | --- |
| 0–36 | Ordinary cuts, also used by decline-riichi cuts 37–73. |
| 37–73 | Declared-riichi cuts 74–110. |
| 74 | Response pass 111. |
| 75 | Kan decline 260. |
| 76–109 | Kan actions 261–294. |

Joint chi/pon actions use their accepted-call state's discard scoring.
`action_win_value_range[110,2,2]` has ron/tsumo and min/max axes.
`action_win_value_valid[110,2]` marks legal available completions;
`action_value_eligible[110]` marks supported searches, and
`action_open_yaku_restricted[110,2]` identifies open no-yaku restrictions.
Closed dama is scored without riichi. Declaration actions and actions after
accepted riichi include it. No-yaku dama can have a legal menzen-tsumo range
and no legal ron range. One-shanten searches preserve the action's declaration
state rather than assuming a later riichi.

Collector requests also expose `reference_dora_tile_id` and
`reference_dora_valid`, each `[B,5]` int8 bytes, to preserve ordered indicator
slots for a frozen reference policy. These sidebands are separate from learned
inputs, which contain dora multiplicities.

Datasets, native libraries and exports must agree on the complete schema.
Rebuild records from source after a schema change; changing summary metadata
does not convert serialized values. Existing exports retain their own input
contracts.

## Collector interface

`NativeCollector` runs complete games through three operations:

1. `next()` returns pending decisions and packed rows, or `None` when finished.
2. `submit(codes)` supplies one legal action per pending decision.
3. `finish()` returns completed game logs and counters.

[training/self_play.py](../../training/self_play.py) batches worker requests
for inference and retains their rows, selected actions and behavior
probabilities. Payments and belief targets are reconstructed from logs.

`snapshot(decision)` captures a pending decision between `next()` and
`submit()`, including simultaneous requests in that game. The root exposes
`decision` (its local request offset) and
`fork(copies, seed, shuffle_remaining=True, *, until_hand_end=False)`.
Forks preserve public history, concealed hands, pending actions and prefix
logs. They shuffle unrevealed wall tiles without moving revealed indicators,
and seed future hands from the branch seed. Duplicate-wall forks apply the
same physical-tile permutation to each remaining player queue, preserving
queue order, counterclockwise borrowing and shared rinshan tiles. With
shuffling disabled, the initial engine state is identical; subsequent
collector randomness still uses the supplied seed. Equal seeds reproduce
forks, and forking leaves the parent unchanged.

Continuations condition on captured hidden hands; they do not sample an
information-set posterior. Python collection emits suffix decisions only.
With `until_hand_end=True`, a branch stops after the current hand settles,
retaining the next hand's start as a boundary or the real `end_game` event.
The pending next hand supplies no training rows. Counters include the initial
partial hand and its earlier riichi declarations, so fork statistics describe
root-conditioned continuations rather than ordinary match rates.

`NativeCollector.from_match(events_json, seed, duplicate)` recreates a seeded
trajectory. Use `next()` and `submit_recorded()` to inspect and advance it.
Replay verifies generated events, including discard origin, consumed tiles
and settlements. Incompatible seeds or simulator histories fail. Triple-ron
abortive draws reconstruct all three legal claims. Unobserved losing claims
are replayed as passes to reconstruct the public trajectory; they are not
imitation labels. Snapshots leave replay mode, so counterfactual claims follow
the supplied policy.

## Source and checks

| Source | Responsibility |
| --- | --- |
| [src/features.rs](src/features.rs) | Public snapshots and legal accepted-call states. |
| [cpp/features.cpp](cpp/features.cpp) | Table, river and meld features. |
| [cpp/candidate.cpp](cpp/candidate.cpp) | Hands, masks and efficiency features. |
| [cpp/yaku.cpp](cpp/yaku.cpp) | Public feasibility and fixed-meld values. |
| [cpp/scoring.cpp](cpp/scoring.cpp) | Completion search scored by RiichiEnv. |
| [src/lib.rs](src/lib.rs) | Python bindings. |

```sh
python -m pytest integration/big/test_native_collection.py \
  integration/big/test_feature_generation.py \
  integration/big/test_self_play.py \
  integration/big/test_selected_roots.py
```

Independent Python calculations under `integration/` check native parity.
