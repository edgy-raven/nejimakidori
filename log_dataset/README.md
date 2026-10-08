# Dataset job

[`dataset.py`](../dataset.py) prepares archives and builds training records.
The output contains `pipeline.json` and `data/`; pass `data/` to
[`train.py`](../train.py). See the [root commands](../README.md#dataset-and-training).
Store datasets outside the source checkout.

## Inputs and recovery

- `--archives DIR` reads rank-enriched archives containing `player_ranks.json`.
- `--databases DIR` downloads pinned 2024/2025 Houou logs and enriches ranks
  from historical Tenhou databases. Combine it with `--archives` to enrich
  local raw archives instead.
- Repeatable `--extra-archives DIR` adds enriched sources.
- `--own-bots ID ...` snapshots games against the approved
  [RiichiLab teachers](riichilab.py) into `teachers/`.
- `--workers` controls record builders; `--limit-games` caps the game count.

Downloads verify pinned hashes and resume partial files. Raw and enriched
archives stay separate. A resumed job reuses prepared archives and the teacher
snapshot; a new snapshot requires a new output directory. Duplicate game IDs
are rejected.

Rerun the same command to resume. Worker count may change; changed inputs or
record contracts require a fresh output. A writer lock prevents overlapping
builds. Shards publish metadata atomically, verify completed sizes on resume,
and rebuild interrupted shards without metadata.

## Selection and records

[`review.py`](review.py) converts cached browser rounds to MJAI and reconstructs
decision-time state. Both the model API and offline rollout collection use this
reader; reconstruction does not depend on HTTP services.

The reserve file contains a `games` list of objects with `game_id`. Those games
are excluded before deterministic whole-game train, validation and held-out
partitioning.

Historical ranks select all 9+ dan seats, 75% of 8d and 50% of 7d using fixed
hash order within each year/rank stratum. Missing and lower ranks supply context
but no imitation targets. RiichiLab teachers use explicit rank assignments;
action penalties exclude the entire game and disconnected teachers are excluded.
All opponents' events remain available for state and outcome reconstruction.
RiichiLab draw deltas are normalized before target generation: native nagashi
and zero-transfer exhaustive draws can include already-paid riichi deposits.
The correction requires the discrepancy to equal the accepted deposits;
other nonconserving draw payments fail. Raw archives remain unchanged.
Excluded seats are skipped before decision features and labels are generated;
decision identities retain their original sequence. Forced-only decisions
produce no training rows.

[`critic_data.py`](critic_data.py) writes joint policy/payment TFRecords, shard
counts and `dataset_summary.json`. [`records.py`](records.py) defines the record
contract; consumers require matching record, scoring and sampling contracts.
The training reader groups the stored 130 score scenarios into 27 equivalent
score targets plus occurrence counts on CPU. This preserves the score loss
while reducing GPU label transfer. Stored bonus/payment-scenario fields remain
part of the record contract but are not decoded by the training reader.
Each training shard has a corresponding `validation-*.tfrecord.gz` containing
a deterministic 1/16 decision-hash sample from game partition 7. The hash
includes game, hand, decision and actor, with a fixed validation salt; it does
not depend on labels. Sample counts and file sizes are recorded in each shard
manifest and the dataset summary. Validation reads all these sample shards,
so later years and supplemental teachers are eligible without a corpus-prefix
cutoff.
Decision-time features select focused records; future outcomes supply labels,
never sampling eligibility or model inputs. Payment labels use RiichiEnv's
native Rust scorer with Tenhou yakuman and hidden-bonus rules. See the
[model contract](../model/PAYMENT_CRITIC.md) for labels and objectives.

Completion evidence is reused in a bounded per-worker cache keyed by concealed
counts, melds and winds. Candidate affinity requests only its three supervised
families; necessary tile-shape conditions skip scoring routes that cannot carry
any of them. Copy weights, scorer decisions and unknown-search labels are
unchanged. Replay observations are reused only until the next applied event.
Discard-route searches share their parent hand's structural neighborhood while
retaining separate route weights and search limits. Native structural lookups
use a bounded cache with full-key equality checks. The local native library is
built in release mode and rebuilt when its source or build recipe changes.
Each worker also reuses a bounded set of immutable archive indexes across shards.

## Suspected AFK hands

[`afk.py`](afk.py) excludes a kyoku when either rule qualifies:

- Two consecutive complete hands from one seat have at least eight discards
  each, all tsumogiri, with no tedashi, riichi, call, kan or win. Every hand in
  the streak is removed. A shorter hand or active choice resets the streak.
- One hand contains at least 12 unforced tsumogiris, including at least two
  shanten-improving draws, while the held hand is not tenpai and no opponent
  has declared riichi during the run. Earlier tedashi or calls do not exempt
  the rest of the hand.

An active choice ends that seat's run; other players' calls do not. Own riichi
ends the run and suppresses forced post-riichi discards. Improving draws are
actual discarded tiles in the fixed held hand's structural ukeire, including
red/ordinary five equivalence. Unchanged-shanten upgrades do not count.
Only long-run candidates need shanten checks. Multiple runs or inactive seats
still exclude each kyoku once. Hidden tiles support offline quality filtering
and never become policy inputs.

All four players' decisions from an excluded kyoku are removed before feature
generation. Other hands retain their original starting state, payments and final
match labels, including first-dealer tie-breaking. A hanchan disappears only
when no hands remain. Filtering does not invent counterfactual match results.

The archives have no disconnect/timeout fields, so these rules identify
suspected inactivity. Folding or another yaku route can be false positives;
brief absences and runs during opponent riichi can be missed. The detector
returns hand indices and seat/discard evidence through `retained_hands`;
summaries declare `afk_filter.scope = "kyoku"`. Rebuild records to apply changes.

## Implementation

| Modules | Responsibility |
| --- | --- |
| `download`, `archives`, `player_ranks`, `riichilab` | Source acquisition, validation and rank selection. |
| `afk`, `shanten`, `scoring` | Game analysis and quality filtering. |
| `critic_data`, `records`, `replay` | Record building, readers and event reconstruction. |
| `training_targets`, `payment_bonus`, `sakigiri_regret` | Offline labels. |
| `focus_sampling`, `rollout_store`, `rollout_collection` | Record selection, one-hand sample readers, disagreement marks and collection queue. |
