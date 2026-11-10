# Candidate training

## Prerequisites

- Complete [setup](../../README.md#setup).
- Build a [dataset](../../log_dataset/README.md); use its `data/` directory.
- Download compatible [weights](../../weights/README.md).
- Reserve one GPU for collection and at least one other for training.

Find GPU UUIDs with `nvidia-smi -L`. Every visible GPU except
`--collection-device` trains the model. The batch size must divide evenly
across training replicas.

## 1. Configure evaluation

Save this argument list as `evaluation-command.json` and replace the paths and
GPU UUID. The runner fills in `{weights}` and `{output}`.

```json
["python", "-m", "training.candidate.evaluation",
 "--weights", "{weights}", "--output", "{output}",
 "--opponent", "/path/to/release/saved_model",
 "--games", "10000", "--seed", "941", "--devices", "GPU-TRAIN"]
```

Evaluation runs between passes and may use a training GPU. Keep opponent and
seeds fixed across passes; use fresh seeds for independent confirmation.

## 2. Launch

This example uses the selected objective with smaller collection budgets.
Choose batch and pool sizes for the available hardware.

```sh
CUDA_VISIBLE_DEVICES=GPU-TRAIN,GPU-COLLECT python -m training.candidate \
  --initial /path/to/release/final.weights.h5 \
  --awr-reference /path/to/release/final.weights.h5 \
  --data /path/to/dataset/data --output /path/to/new-run \
  --collection-device GPU-COLLECT \
  --evaluation-command evaluation-command.json \
  --record-fractions 0.8,0.6,0.4,0.2,0.2 \
  --awr-learning-rate 5e-6 --awr-final-learning-rate 3e-6 \
  --awr-temperature 10 --awr-pull 0.3 --awr-q-std 1 \
  --root-copies 8 --target-fraction 0.4 \
  --batch-size 128 --games 32 --collection-workers 4 \
  --fresh-positions 8192 --cohort-exposures 32768 \
  --replay-positions 65536
```

## Pass schedule

Each pass consumes as many update positions as the recorded training corpus.
Reusing a replay position counts toward this budget.

| Pass | Recorded | Self-play | Learning rate in the example |
| --- | --- | --- | --- |
| Recorded AWR | 100% | 0% | 5e-6 |
| Mixed 1 | 80% | 20% | 4.5e-6 |
| Mixed 2 | 60% | 40% | 4e-6 |
| Mixed 3 | 40% | 60% | 3.5e-6 |
| Mixed 4 | 20% | 80% | 3e-6 |
| Mixed 5 | 20% | 80% | 3e-6 |

Learning rate stays constant within a pass. `--mixed-only` skips the initial
recorded pass. The explicit flags above select this recipe; CLI defaults are
not a substitute for saving a complete run command.

## Main controls

| Option | Meaning |
| --- | --- |
| `--awr-reference`, `--awr-pull` | Fixed policy and KL coefficient; never refreshed between cohorts |
| `--awr-q-std` | Standard-deviation penalty in the actor's Q-based weight |
| `--awr-temperature` | Scale of the detached AWR exponent, in ST3 points |
| `--fresh-positions` | New generated positions per cohort |
| `--cohort-exposures` | Replay positions consumed per cohort |
| `--replay-positions` | Capacity of each disk-backed replay pool |
| `--root-copies` | Continuations per selected action |

See [objective](../OBJECTIVE.md) and [sampling](SAMPLING.md) for exact semantics.
Run `python -m training.candidate --help` for all options.

## Monitor

```sh
tensorboard --logdir /path/to/new-run/tensorboard
```

| Output | Contents |
| --- | --- |
| `plan.json` | Settings and initial/reference weight hashes |
| `status.json`, `tensorboard/` | Progress, losses, gradients, learning rate and throughput |
| `checkpoints/<exposures>.weights.h5` | Immutable cohort checkpoints |
| `latest.weights.h5` | Link to the latest checkpoint |
| `<phase>.weights.h5` | Completed-pass weights |
| `<phase>-evaluation/` | Export and match results |

## Restart

Use a saved checkpoint as `--initial` and a new output directory. Keep
`--awr-reference` fixed independently of the restart checkpoint. Optimizers,
replay, counters, schedules and TensorBoard history start fresh. Candidate
checkpoints contain weights only.

## Files

| File | Responsibility |
| --- | --- |
| [run.py](run.py) | Arguments, replay and pass scheduling |
| [collection.py](collection.py) | Frozen collector and batched root scoring |
| [monitor.py](monitor.py) | Progress and TensorBoard |
| [state.py](state.py) | Atomic JSON metadata writes |
| [evaluation.py](evaluation.py) | Export and evaluation command |
