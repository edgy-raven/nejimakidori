# Train, export and evaluate

Training requires a [source installation](../README.md#build-without-docker),
a [prepared dataset](../log_dataset/README.md) and NVIDIA GPUs. Start from the
[released weights](../weights/README.md), or weights from a compatible run.
Use the dataset's `data/` directory in the commands below.

## Train with recorded games and self-play

`training.candidate_training` uses advantage-weighted regression (AWR). It
first trains on recorded games, then runs five passes that mix recorded games
with newly generated self-play. It saves and evaluates a model after every
pass.

Use one GPU for collection and at least one other GPU for training. Find their
UUIDs with `nvidia-smi -L`. All visible GPUs except `--collection-device` are
used for training; the batch size must divide evenly between them.

First create `evaluation-command.json`, replacing the model path and GPU UUID:

```json
["python", "-m", "training.candidate_evaluation",
 "--weights", "{weights}", "--output", "{output}",
 "--opponent", "/path/to/release/saved_model",
 "--games", "512", "--seed", "941", "--devices", "GPU-TRAIN"]
```

`{weights}` and `{output}` are filled in by the runner. Evaluation runs between
training passes, so it can use a training GPU. Use the same opponent and seeds
when comparing passes, then use fresh seeds for a separate confirmation match.

Launch training from the repository root:

```sh
CUDA_VISIBLE_DEVICES=GPU-TRAIN,GPU-COLLECT \
python -m training.candidate_training \
  --initial /path/to/release/final.weights.h5 \
  --data /path/to/dataset/data \
  --output /path/to/new-run \
  --collection-device GPU-COLLECT \
  --evaluation-command evaluation-command.json \
  --batch-size 128 --games 32 --collection-workers 4 \
  --fresh-positions 8192 --cohort-exposures 32768 \
  --replay-positions 65536
```

Choose a new output directory for each run. Use a fast disk with room for
checkpoints and memory-mapped replay pools. Batch size and worker count depend
on available memory and hardware; measure complete collection/update cycles
before scaling them up.

## Candidate replay schedule

Each pass has an update budget equal to the number of training positions in
the recorded dataset. Reusing a replay position counts toward that budget.

| Pass | Recorded games | Self-play |
| --- | --- | --- |
| 1 | 100% | 0% |
| 2 | 80% | 20% |
| 3 | 60% | 40% |
| 4 | 40% | 60% |
| 5 | 20% | 80% |
| 6 | 0% | 100% |

Set `--record-fractions` to five nonincreasing fractions to change passes 2–6.
For example, `0.8,0.6,0.4,0.2,0.2` retains 20% recorded data in the last pass.
Use `--mixed-only` to skip pass 1.

Within each source, `--target-fraction` controls the share drawn from positions
selected for calls, tenpai, late-hand decisions or policy/value disagreement.
Its default is 0.4. See [replay selection](REFERENCE.md#candidate-replay-schedule)
for the exact predicates and paired-rollout behavior.

| Option | Purpose |
| --- | --- |
| `--awr-learning-rate`, `--awr-final-learning-rate` | Initial and final learning rates. Each pass uses a constant rate, interpolated by self-play share. |
| `--awr-temperature` | Controls how strongly AWR favors higher-return actions. |
| `--awr-pull` | Penalizes divergence from the policy saved at the previous cohort boundary. |
| `--fresh-positions` | New self-play positions collected per cohort. |
| `--cohort-exposures` | Replay positions consumed by updates per cohort. |
| `--replay-positions` | Capacity of each replay pool. |
| `--root-copies` | Repeated continuations per selected action; default 1. |

Run `python -m training.candidate_training --help` for all options.

## Monitor and restart

```sh
tensorboard --logdir /path/to/new-run/tensorboard
```

The runner writes `status.json` for progress and TensorBoard events for losses,
value error, AWR weights, collection throughput and learning rate. A loss
improvement alone does not establish stronger play; compare match results too.

| Output | Contents |
| --- | --- |
| `plan.json` | Run settings. |
| `checkpoints/<exposures>.weights.h5` | Immutable model weights at cohort boundaries. |
| `latest.weights.h5` | Link to the latest saved weights. |
| `<phase>.weights.h5` | Model weights at a completed pass. |
| `<phase>-evaluation/` | Export and match results for that pass. |

Candidate checkpoints contain weights only. To restart, pass a saved checkpoint
as `--initial` and choose a new output directory. This starts fresh optimizers,
replay, counters and schedules. Candidate runs do not resume optimizer state.

## Export and evaluate

Export trained weights:

```sh
python -m model.export \
  --weights /path/to/run/mixed-awr-5.weights.h5 \
  --data /path/to/dataset/data \
  --output /path/to/export
```

Export verifies finite outputs, normalized distributions and round-trip parity.
The output contains `saved_model/` and its matching `source/`. Keep them
together when moving an export.

Compare two exports:

```sh
python -m training.evaluate \
  --new /path/to/export/saved_model \
  --old /path/to/opponent/saved_model \
  --output /path/to/match \
  --games 5000 --workers 1 --devices GPU-UUID
```

Each player loads its own source snapshot. Games rotate each seed through all
four seats, so `--games` must be divisible by four. Repeating an unchanged
command resumes completed games. `summary.json` reports placements, scores,
ST3 points and paired uncertainty bounds.

To export without a recorded dataset, replace `--data` with
`--self-play-seed 941`. See [serving](../service/README.md) to run the export.

## Phasic training

`training.trainer` supports supervised foundation updates, recorded AWR,
self-play PPO and full optimizer-state recovery. It has different scheduling
and checkpoint semantics from the candidate runner. See
[phasic training](REFERENCE.md#phasic-training) for commands and budgets.

## Self-play-only training

`training.online_training` trains from generated games without a recorded
corpus. See [self-play-only training](REFERENCE.md#self-play-only-training).

## Value experiments

The [experiment guide](EXPERIMENTS.md) covers repeated-continuation collection,
value fitting, architecture comparisons and held-out confirmation.
