# Build a training dataset

The dataset builder converts ranked game archives into compressed TFRecords
for policy, belief and settlement training. Run commands from the repository
root after completing [setup](../README.md#build-without-docker).

Records include optional ron/tsumo decisions: `win_action` is `0` for pass,
`1` for win, and `-1` outside a win decision. Corpora produced while wins were
automatic omit these examples and must be regenerated to train win/pass.

## Build

For existing rank-enriched archives containing `player_ranks.json`:

```sh
python dataset.py \
  --archives /path/to/enriched-archives \
  --reserve /path/to/reserve.json \
  --output /path/to/dataset \
  --workers 24
```

The reserve file excludes evaluation games. Its format is:

```json
{"games": [{"game_id": "evaluation-game-id"}]}
```

Use `{"games": []}` when there are no reserved games. Keep generated data
outside the source checkout. The builder requires at least 15 GiB free before
starting each shard, in addition to space for the corpus itself.

| Option | Use |
| --- | --- |
| `--archives DIR` | Read existing archives. |
| `--databases DIR` | Enrich ranks from historical Tenhou databases. Without `--archives`, download the pinned Houou archives first. |
| `--extra-archives DIR` | Add another enriched source; repeat for multiple sources. |
| `--own-bots ID ...` | Import games against the configured [RiichiLab expert players](riichilab.py). |
| `--limit-games N` | Build a small sample before a full corpus. |
| `--workers N` | Set the number of CPU record builders. |

## Output and recovery

Pass the output's **`data/` subdirectory** to training and export commands.

| Path | Contents |
| --- | --- |
| `pipeline.json` | Source selection and build settings. |
| `data/train-*.tfrecord.gz` | Main record shards. |
| `data/validation-*.tfrecord.gz` | Sampled validation records. |
| `data/*.tfrecord.json` | Completed shard metadata. |
| `data/dataset_summary.json` | Completed corpus counts and schema. |
| `data/status.json` | Build progress. |

Rerun the same command to resume an interrupted build. Worker count may change;
inputs and record definitions must stay the same. Completed shards are checked
against their metadata, and incomplete shards are rebuilt. A writer lock
prevents two builds from using the same directory.

## Which positions enter training

The reserve is excluded first, then games are deterministically assigned to
training, validation and held-out partitions. Historical ranks select all 9+
dan seats, 75% of 8d and 50% of 7d seats. Other players still supply game context.
Forced-only decisions produce no training rows.

The mixed training stream contains 50% natural positions, 25% focused positions
and 25% placement-gap positions. Focus categories cover every response-call
and kan opportunity, tanyao, flush, outside hands, threats and late-match
decisions. Call eligibility is derived from decision phase at sampling time,
including passes, matching candidate replay and targeted self play. Stored
fast-call metadata remains descriptive; it does not restrict call selection.
Existing records use this same rule without regenerating their features. Selection uses visible state
at the decision; future outcomes supply labels. Definitions and weights are in
[focus_sampling.py](focus_sampling.py).

Validation uses a deterministic 1/16 decision sample from the natural validation
partition. It is stored in separate shards so validation can run without
scanning the training corpus.

[afk.py](afk.py) removes hands with suspected inactivity: consecutive long
all-tsumogiri hands, or a long unforced tsumogiri run that ignores improving
draws outside tenpai and opponent riichi. These are behavioral heuristics;
the archives have no timeout or disconnect signal. A flagged hand is removed
for all seats, while the remaining hands keep their original context.

## Features and labels

The [native engine](native_collection/README.md) extracts public features.
[replay.py](replay.py) reconstructs recorded decisions;
[training_targets.py](training_targets.py), [scoring.py](scoring.py) and
[sakigiri_regret.py](sakigiri_regret.py) produce their targets. Hidden hands and
later outcomes are available to label generation, never to the policy inputs.

[records.py](records.py) defines label names, shapes and serialization.
Dataset consumers check its schema and scoring definitions against
`dataset_summary.json`. Rebuild in a new directory after a schema change;
editing an old summary does not convert its records.

[training_data.py](training_data.py) also stores `critic/settlement`: twelve
int32 values containing actor-relative post-hand scores and zero-based ranks,
next relative dealer, next hand number, terminal flag and outstanding deposits.
Complete source games supply these labels before hand filtering. Parsing applies
the shared fixed ST3 continuation forecast to obtain the observed return for
value regression and AWR. These outcomes are targets, never encoder inputs.
Older records without this required field must be rebuilt.

For prediction meanings, see [the model architecture](../model/ARCHITECTURE.md).
