# Train, export and evaluate

[`train.py`](../train.py) runs three stages: joint training, verified SavedModel
export, then a duplicate match against a frozen reference policy. Each stage
runs in a separate process on the GPUs you select.

You need a [prepared dataset](../log_dataset/README.md), compatible initial
weights, a reference SavedModel with its source, and the
[development dependencies](../README.md#work-with-the-source).

## Initialization

Choose one starting point:

| Starting point | Arguments | Behavior |
| --- | --- | --- |
| Joint checkpoint, such as v1 | `--initial FILE --beliefs FILE` | Loads all joint weights; starts a fresh optimizer and schedule. |
| Separately trained beliefs | `--beliefs FILE` | Loads the shared encoder and belief heads; initializes the remaining heads afresh. |

The CLI currently requires `--beliefs` in both cases. When `--initial` is
present, it takes precedence and the belief file is not read; the example
below passes the joint checkpoint to both arguments. The v1 release includes
joint weights but no standalone belief initializer. A SavedModel export is
for inference and cannot replace either training checkpoint.

## Run a short end-to-end check

Set these to absolute paths outside the source checkout:

```sh
export DATASET_PATH=/path/to/dataset
export RELEASE_PATH=/path/to/weights/nejimakidori-v1
export RUN_PATH=/path/to/new-run
export TRAIN_DEVICES=GPU-UUID

python train.py \
  --data "$DATASET_PATH/data" --output "$RUN_PATH" \
  --initial "$RELEASE_PATH/final.weights.h5" \
  --beliefs "$RELEASE_PATH/final.weights.h5" \
  --opponent "$RELEASE_PATH/saved_model" --devices "$TRAIN_DEVICES" \
  --mode smoke --games 4
```

This runs four optimizer updates, verifies the export, and plays four games.
It checks the pipeline, not playing strength. For a full run, use a fresh
output directory and omit `--mode smoke --games 4`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode` | `full` | `smoke`: 4 updates; `experiment`: 12,000; `full`: 770,181,120 position exposures. |
| `--batch-size` | `9216` | Global batch size; must divide the full exposure budget and replica count. |
| `--phase-batches` | Unset | Override the update budget for smoke or experiment mode. |
| `--devices` | Required | Comma-separated GPU UUIDs for training and evaluation. |
| `--awr-mix` | `1` | Final AWR share; `0` gives unweighted imitation. |
| `--awr-temperature` | `3` | Scale of the critic advantage before exponentiation. |
| `--games` | `10000` | Evaluation games; a positive multiple of four. |
| `--seed` | `0` | First evaluation seed. |
| `--match-workers` | `1` | Evaluation worker processes. |
| `--batch-games` | `8` | Concurrent games per evaluation worker. |

See [architecture and objectives](PAYMENT_CRITIC.md) for what the model learns.

## Outputs and recovery

```text
run/
├── pipeline.json              Fixed job arguments
├── source/                    Source snapshot used by all stages
├── candidate/                 Checkpoints, final weights and validation
│   └── tensorboard/           Training event files
├── policy/                    Verified SavedModel and matching source
├── evaluation/                Match records, replays and summary.json
├── pipeline_status.json       Current stage
└── training.log, export.log, evaluation.log
```

```sh
tensorboard --logdir "$RUN_PATH/candidate/tensorboard"
```

Rerun the same command to resume. The trainer restores optimizer state from
recovery checkpoints; export reuses a verified result; evaluation skips
completed games. Changed arguments require a new output directory. An
interrupted export is rebuilt before it replaces the final export directory.

`policy/saved_model` exposes full diagnostic and policy-only signatures.
Verification compares their policy outputs. Evaluation loads each player's
own frozen feature producer; the reference SavedModel needs its matching
`source/model` tree beside or above it, as in the released bundle.

Each evaluation seed runs a four-seat rotation, with one candidate against
three reference actors per game. `evaluation/summary.json` reports paired
estimates and confidence intervals across complete seed groups. Training
completion does not publish or deploy the model.

## Optional one-hand rollouts

A completed rollout store can add settlement labels and paired win-opportunity
preferences:

```sh
# Add these arguments to the training command:
# --rollouts /path/to/rollouts.sqlite
# --rollout-overhead 0.1 --rollout-stale-weight 0.5
```

Collection is a separate CPU job. Mark cached reviews made by the specified
model, then collect trials:

```sh
python -m model.collect_rollouts mark \
  --store /path/to/rollouts.sqlite \
  --model "$RELEASE_PATH/saved_model" --revision MODEL_REVISION \
  /path/to/review.json.gz
python -m model.collect_rollouts collect \
  --store /path/to/rollouts.sqlite --seconds 3600 --trials 32
```

`MODEL_REVISION` must identify the model that produced the reviews. The input
is a complete annotated review with recoverable hidden hands, not a raw game
log. Use one collector per store. Rerunning resumes incomplete work. Training
opens completed samples read-only; confirmation and holdout trials are excluded.

Each position receives at most `--trials` trials. Different trials use
different future walls; the alternatives within a trial share its wall.
`--trial-batch-size` defaults to 16 walls. A batch already admitted can overrun
`--seconds`; use a smaller batch for finer time control. The training replay
budget is separate from this collection budget.

See [rollout supervision](PAYMENT_CRITIC.md#rollout-supervision) for nominations,
losses and the limits of these counterfactual labels.

## Docker

The training image builds from this checkout and needs no release build context:

```sh
docker build -f docker/Dockerfile.model --target training \
  -t nejimakidori-training .
```

Mount a common artifact directory and keep all CLI paths inside that mount:

```sh
docker run --rm --gpus '"device=GPU-UUID"' \
  --mount type=bind,src=/absolute/path/to/artifacts,dst=/artifacts \
  nejimakidori-training \
  python train.py --data /artifacts/dataset/data \
  --output /artifacts/run \
  --initial /artifacts/weights/nejimakidori-v1/final.weights.h5 \
  --beliefs /artifacts/weights/nejimakidori-v1/final.weights.h5 \
  --opponent /artifacts/weights/nejimakidori-v1/saved_model \
  --devices GPU-UUID --mode smoke --games 4
```

Replace both `GPU-UUID` values with the same device identifier. The mount must
contain the dataset, complete release bundle and writable output directory.
