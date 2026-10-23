# Train, export and evaluate

Start with [setup](../README.md#setup) and a [built dataset](../log_dataset/README.md).
Use its `data/` directory in the commands below. The [architecture guide](ARCHITECTURE.md)
explains the policy, beliefs and settlement model.

Training requires CUDA GPUs visible to TensorFlow. Select devices by UUID with
`CUDA_VISIBLE_DEVICES`; batch and phase allocations must divide evenly across
the replicas. Initial weights must match the current architecture and feature
schema.

## Phasic training

`model.training` combines expert records with fresh self-play. This small example
starts from joint weights and exercises every phase:

```sh
CUDA_VISIBLE_DEVICES=GPU-UUID python -m model.training \
  --data /path/to/dataset/data \
  --initial /path/to/initial.weights.h5 \
  --output /path/to/run/candidate \
  --batch-size 128 \
  --foundation-exposures 0 \
  --awr-only-exposures 16384 \
  --awr-exposures 49152 \
  --final-awr-exposures 16384 \
  --ppo-learning-rate 1e-5
```

Budgets count positions used by optimizer updates, including repeated records.
For a pass budget `N`, the example follows this schedule:

| Phase | Exposures | Control |
| --- | --- | --- |
| Optional supervised foundation | Separate budget | `--foundation-exposures` |
| Expert AWR prefix | `N` | `--awr-only-exposures N` |
| Alternating expert AWR and self-play PPO | `2N` each | `--awr-exposures 3N`, `--ppo-exposure-ratio 1` |
| Anchored expert AWR tail | `N` | `--final-awr-exposures N` |

Set every run's AWR budget explicitly.
Choose `N` from the rebuilt corpus size and round to the batch allocation.
The AWR total includes its prefix. PPO's budget is
`(awr_exposures - awr_only_exposures) × ppo_exposure_ratio`.
The final AWR budget is additional, giving `6N` overall when foundation is zero.

AWR reads expert records. Each cycle finishes AWR, collects fresh games with
the updated policy, and runs PPO on that cohort before discarding it.

The offline phases share a standard Keras AdamW optimizer and schedule; PPO
has separate AdamW state. Both use weight decay `1e-4` and per-variable
`clipnorm=1.0`. Distributed loss reductions use TensorFlow autodiff.
Final AWR freezes the post-PPO actor as a reference and penalizes forward
KL with `--final-awr-kl` (default `0.1`). The moving policy prior is controlled
separately by `--prior-weight` (default `1`).

| Option | Controls |
| --- | --- |
| `--block-exposures`, `--cycle-blocks` | Length of each offline segment. |
| `--games`, `--collection-workers` | Self-play collection. |
| `--cohort-exposures`, `--ppo-epochs` | Maximum reuse of each collected cohort. |
| `--validation-data` | Optional fixed validation subset. |

For offline AWR only, set foundation to zero, set equal AWR-prefix and AWR-total
budgets, and use `--ppo-exposure-ratio 0`.
Use `python -m model.training --help` for all options.

## Monitor and recover

Paths below are relative to `--output`:

| Path | Contents |
| --- | --- |
| `plan.json` | Settings, budgets and replica count. |
| `status.json`, `progress.jsonl` | Current progress and metric history. |
| `tensorboard/` | Training and validation events. |
| `recovery/` | Two retained full checkpoints. |
| `latest.weights.h5` | Joint weights written with the latest recovery checkpoint. |
| `ppo-budget/` | Checkpoint at PPO-budget exhaustion. |
| `middle.weights.h5` | Joint weights before final AWR. |
| `final.weights.h5` | Joint weights after all phases finish. |

TensorBoard focuses on held-out policy agreement and NLL, sampled policy updates,
AWR weight concentration, PPO cohort drift, self-play outcomes and throughput.
Aggregate payment and belief losses are retained as validation health checks.
Self-play projected returns use ST3 divided by 100; they are not opponent
benchmark results. All detailed metrics remain in `progress.jsonl` and
`validation-*.json`, including stable auxiliary losses. The horizontal axis
counts total optimizer exposures across phases.

To resume an interrupted phasic run, repeat its command with `--resume` and the
same output directory. This restores optimizers, counters, schedule and elapsed
training time. Work after the last checkpoint is repeated.

To start fresh, use a new output directory and pass the previous run's
`latest.weights.h5` as `--initial`. This starts new optimizers, counters,
schedules and TensorBoard history. Weights must match the current architecture.

## Export and evaluate

```sh
python -m model.export \
  --weights /path/to/run/candidate/final.weights.h5 \
  --data /path/to/dataset/data --output /path/to/run/policy

python -m model.evaluate \
  --new /path/to/run/policy/saved_model \
  --old /path/to/opponent/saved_model \
  --output /path/to/run/evaluation \
  --games 5000 --workers 1 --devices GPU-UUID
```

Export needs a dataset with a training partition. It verifies finite outputs,
normalized distributions and SavedModel round-trip parity, and saves matching
source beside the model.

Each evaluation player uses its own frozen features. The opponent therefore
needs its matching `source/model` beside or above its SavedModel. Matches rotate
each seed through four seats, so the game count must be divisible by four.
Rerunning an unchanged evaluation resumes completed games. `summary.json`
contains paired estimates and confidence intervals.

Use the [service guide](../service/README.md) to serve the resulting export.
