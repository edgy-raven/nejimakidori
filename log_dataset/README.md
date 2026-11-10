# Build a training dataset

The dataset builder converts ranked game archives into compressed TFRecords
for policy, belief and settlement training. Run commands from the repository
root after completing [setup](../README.md#setup).

Records include optional ron/tsumo decisions: `win_action` is `0` for pass,
`1` for win, and `-1` outside a win decision. Corpora produced while wins were
automatic omit these examples and must be regenerated to train win/pass.

## Build

For existing rank-enriched archives containing `player_ranks.json`:

```sh
python -m log_dataset \
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

## Reference

| Topic | Read |
| --- | --- |
| Sampling, labels and compatibility | [Record contract](RECORDS.md) |
| Feature tensors and simulator interface | [Native engine](native_collection/README.md) |
| Model prediction meanings | [Architecture](../model/ARCHITECTURE.md) |
