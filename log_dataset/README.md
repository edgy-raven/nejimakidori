# Build a dataset

[`dataset.py`](../dataset.py) prepares game archives and writes the TFRecords
consumed by [`train.py`](../train.py). Dataset building is CPU work. Keep raw
archives, records and model weights outside the source checkout.

## Build from enriched archives

Use the [development environment](../README.md#work-with-the-source) or the
[training image](../model/README.md#docker). An enriched archive directory
contains yearly ZIP files and `player_ranks.json`, which records ranks and
archive checksums. ZIP members contain MJAI games (`.mjson` or `.jsonl`).
[archives.py](archives.py) validates the format and rejects duplicate game IDs.

Create a reserve file listing games to exclude. For no exclusions:

```sh
export DATASET_PATH=/absolute/path/to/dataset
mkdir -p "$DATASET_PATH"
printf '%s\n' '{"games": []}' > "$DATASET_PATH/reserve.json"
python dataset.py --archives /absolute/path/to/enriched-archives \
  --reserve "$DATASET_PATH/reserve.json" --output "$DATASET_PATH" \
  --workers 24
```

To reserve specific games, use `{"games": [{"game_id": "..."}]}`. These games
are excluded before the remaining corpus is split into training, validation
and holdout. Pass `"$DATASET_PATH/data"` to the trainer.

## Other input sources

| Arguments | Input preparation |
| --- | --- |
| `--archives DIR` | Use existing enriched archives. |
| `--databases DIR` | Download the pinned 2024/2025 Houou archives and enrich them from historical Tenhou rank databases. |
| `--archives DIR --databases DIR` | Enrich existing raw archives instead of downloading them. |
| `--extra-archives DIR` | Add another enriched source; repeat for multiple directories. |
| `--own-bots ID ...` | Snapshot games against the approved teachers defined in [riichilab.py](riichilab.py). |
| `--limit-games N` | Limit the build for a small pipeline check. |

Downloading requires the historical rank databases. The model release does
not distribute archives or databases. [player_ranks.py](player_ranks.py)
defines database parsing and enrichment; [download.py](download.py) pins
source archives and their hashes.

## Outputs and resuming

The output contains `pipeline.json`, prepared inputs when needed, and `data/`.
The data directory contains training shards, validation shards, shard manifests
and `dataset_summary.json` with counts and coverage.

Rerun the same command to resume; the worker count may change. Other changed
inputs or record schemas require a fresh output directory. Completed shards
are checked against their metadata; interrupted shards are rebuilt. A writer
lock prevents overlapping builds. Downloads resume partial files and verify
hashes. A resumed job reuses its teacher snapshot.

## Which decisions are used

Historical ranks select all 9+ dan seats, 75% of 8d and 50% of 7d, using fixed
hash order within each year/rank stratum. Lower or missing ranks provide game
context but no imitation targets. RiichiLab teachers have explicit rank
assignments; action penalties exclude the whole game, and disconnected
teachers are excluded. Forced-only decisions produce no training rows.

Partitioning keeps complete games together. Training reads three natural
records for each focused record. [focus_sampling.py](focus_sampling.py)
defines the focus categories and weights from decision-time features;
recorded outcomes never determine focus eligibility.

Validation shards contain a fixed 1/16 decision-hash sample from game partition
7. All years and supplemental sources can contribute. Validation reads this
natural sample, including the final partial batch.

## Labels

| Label group | Source |
| --- | --- |
| Policy | Recorded action at the decision. |
| Opponent beliefs | Reconstructed concealed tiles, shanten and improving tiles. |
| Settlement | Recorded hand result, scored values, transfers and bank movement. |
| Immediate ron | Native legality/scoring of candidate discards and kan robbery. |
| Completion yaku/value | Best supported ron and tsumo completions at the next observed tenpai. |
| Attack routes | Structural support for tanyao, flush and outside hands. |
| Sakigiri | Verified exchanges of early discard order with matching later state. |

The policy sees only its own hand and public information. Hidden hands and
future events supply offline labels. [The model reference](../model/PAYMENT_CRITIC.md)
defines unknown labels, masks and scoring assumptions.

RiichiLab draw transfers are normalized before labels are built because some
native draw deltas include an already-paid riichi deposit. The correction
must match the accepted deposits; other nonconserving payments fail. Raw
archives are preserved. A zero-payment draw alone does not distinguish
all-tenpai, nobody-tenpai and abortive outcomes.

## Inactivity filtering

[afk.py](afk.py) excludes a complete hand if either condition holds:

- A seat has two consecutive complete hands with at least eight discards
  each, all tsumogiri, and no tedashi, riichi, call, kan or win. Every hand in
  the streak is removed; a shorter hand or active choice resets it.
- A seat makes at least 12 consecutive unforced tsumogiris, including at least
  two shanten-improving draws, while its held hand is not tenpai and no opponent
  has declared riichi during that run.

Own active choices end a run; other players' calls do not. Own riichi ends the
run and suppresses forced post-riichi discards. Improvements use structural
ukeire, including red/ordinary-five equivalence. All four seats' records are
removed for an excluded hand; retained hands keep their original scores and
match results. Summaries report `afk_filter.scope = "kyoku"`.

This is a behavioral heuristic: the archives contain no disconnect or timeout
fields. Folding can cause false positives; short absences and inactivity
against riichi can be missed.

## Implementation reference

| Module | Responsibility |
| --- | --- |
| [review.py](review.py) | Convert browser rounds and reconstruct decision-time state for serving and offline review. |
| [critic_data.py](critic_data.py), [records.py](records.py) | Build and read the current TFRecord schema. |
| [training_targets.py](training_targets.py), [payment_bonus.py](payment_bonus.py) | Completion and score labels. |
| [sakigiri_regret.py](sakigiri_regret.py) | Discard-order witnesses. |
| [rollout_collection.py](rollout_collection.py), [rollout_store.py](rollout_store.py) | Queue and store paired one-hand trials. |

Label generation caches repeated completion searches and structural lookups
within each worker. The reader groups 130 stored score scenarios into 27
same-score targets plus occurrence counts before GPU transfer, preserving
their original loss weighting. These optimizations do not change the stored
record schema or label meaning.
