# Candidate training

## Prerequisites

- Complete [setup](../../README.md#setup).
- Build a [dataset](../../log_dataset/README.md) and its
  [greedy distill policy sidecars](../HYBRID.md).
- Download compatible [weights](../../weights/README.md).
- Reserve one GPU for collection and at least one other for training.

Find GPU UUIDs with `nvidia-smi -L`. Every visible GPU except
`--collection-device` trains the model. The batch size must divide evenly
across training replicas.

## 1. Configure evaluation

Save this argument list as `evaluation-command.json` and replace the paths and
GPU UUIDs. The runner fills in `{weights}` and `{output}`.

```json
["env", "CUDA_VISIBLE_DEVICES=GPU-COLLECT",
 "python", "-m", "training.candidate.evaluation",
 "--weights", "{weights}", "--output", "{output}",
 "--opponent", "/path/to/release/saved_model",
 "--games", "10000", "--seed", "941", "--workers", "2",
 "--devices", "GPU-TRAIN,GPU-COLLECT"]
```

Evaluation runs between passes and may use a training GPU. Keep the opponent
fixed. The command must specify `--seed` and `--games`; games must be positive
and divisible by four. The seed is the first pass's starting seed. Each later
pass advances it by `games / 4`, giving disjoint wall seeds with four seat
rotations each. Retrying the same pass retains its seeds.

After the recorded and mixed 1–4 checkpoint matches, the runner fits the
critic for two complete passes over all recoverable decisions from all four
seats. These updates use the observed settlement objective, including gradients
through the shared encoder, with no actor objective.

Each of those match results describes the checkpoint before Q fitting;
encoder updates can subsequently change policy outputs. The mixed 5 checkpoint
match is the final evaluation. There is no final Q fit or separate confirmation
match. `final.weights.h5` contains those final evaluated policy weights.

See the [match-record contract](../MATCH_RECORDS.md) for recorded actions and
target construction. Each Q phase starts a fresh optimizer at the learning
rate of its preceding policy pass. Its two
passes retain every row, including the final partial batch. Policy exposure
budgets and Q exposure counters are tracked separately.

## 2. Launch

The objective and six-pass schedule are fixed in source. This example uses
smaller collection budgets.
Choose batch and pool sizes for the available hardware.

```sh
CUDA_VISIBLE_DEVICES=GPU-TRAIN,GPU-COLLECT python -m training.candidate \
  --initial /path/to/compatible/initial.weights.h5 \
  --reference /path/to/release/final.weights.h5 \
  --distill-labels /path/to/distill-labels --output /path/to/new-run \
  --collection-device GPU-COLLECT \
  --evaluation-command evaluation-command.json \
  --batch-size 128 --games 32 --collection-workers 4 \
  --fresh-positions 8192 --cohort-exposures 32768 \
  --replay-positions 65536
```

## Pass schedule

Each pass consumes as many update positions as the recorded training corpus.
Reusing a replay position counts toward this budget.

| Pass | Recorded | Self-play | Learning rate |
| --- | --- | --- | --- |
| Recorded imitation | 100% | 0% | 2e-5 |
| Mixed 1 | 80% | 20% | 1.625e-5 |
| Mixed 2 | 60% | 40% | 1.25e-5 |
| Mixed 3 | 40% | 60% | 8.75e-6 |
| Mixed 4 | 20% | 80% | 5e-6 |
| Mixed 5 | 20% | 80% | 5e-6 |

Learning rate decreases linearly with the scheduled self-play share, from
2e-5 at 0% to 5e-6 at 80%. It stays constant within each pass. Changing the
rate preserves policy optimizer moments. Every run starts with recorded
imitation, then five mixed passes capped at 80% self-play. Alternative recipes belong
on experiment branches, not launch flags.

## Fixed recipe

| Setting | Value |
| --- | --- |
| Fixed-reference KL coefficient | 0.3 |
| Excess losing-hand downside penalty coefficient | 0.1 |
| Positive-only expert CE coefficient | 0.2 riichi; 0.2 calls |
| Riichi hand-rate calibration target | 20% |
| Recorded imitation | 50% expert / 50% greedy distill policy Q-weighted CE; separate adaptive beta targeting 70% ESS |
| Previous-policy mirror KL coefficient | 0.3 |
| Continuations per selected action | 1 |
| Targeted sampling fraction | 50% |

## Capacity controls

Training replicas have an 18 GiB TensorFlow allocation limit; collection has
8 GiB. Evaluation pauses learning and can use one worker per GPU. Each
SavedModel worker loads two policies with a 1.5 GiB TensorFlow limit each;
allow additional space for CUDA contexts and kernels. Keep export on the
collection GPU through `CUDA_VISIBLE_DEVICES`, and list the worker GPUs in
`--devices`. Check peak memory with the learner resident before increasing
the worker count or game batch size.

| Option | Meaning |
| --- | --- |
| `--batch-size` | Update positions per batch; divisible by training replicas |
| `--games`, `--collection-workers` | Collection batch and simulator concurrency |
| `--fresh-positions` | New generated positions per cohort |
| `--cohort-exposures` | Replay positions consumed per cohort |
| `--replay-positions` | Capacity of each disk-backed replay pool |

Every restart loads all weights from `--initial`, including actor and attention
weights. `--reference` remains the fixed production magnet. Optimizers,
exposure/cohort counters, schedules, replay and TensorBoard start fresh in a
new output directory. The previous-policy snapshot starts from the initializer.

Initial weights and fixed-reference weights remain separate required paths.
`--seed` controls reproducibility. The saved plan includes the fixed recipe and
capacity settings.

See [objective](../OBJECTIVE.md) and [sampling](SAMPLING.md) for exact semantics.
Run `python -m training.candidate --help` for all options.

Replay pools reserve disk space when first populated. Leave room for all three
pools, checkpoints and evaluation exports; allocation fails before replay rows
are written if the filesystem cannot reserve a pool.

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
| `<phase>.weights.h5` | Policy-pass weights evaluated before Q fitting |
| `<phase>-evaluation/` | Export and match results |
| `<phase>-q-data/` | Current match records used for critic fitting |
| `<phase>-q-1.weights.h5`, `<phase>-q-2.weights.h5` | Q-pass checkpoints |
| `final.weights.h5` | Final policy model; no final Q fit |

Step timing tags separate recorded root scoring from input/replay work and
synchronized optimizer updates:

| Tag | Work measured |
| --- | --- |
| `runtime/input_seconds` | Input preparation and replay; mixed steps include pool replenishment |
| `runtime/root_seconds` | Recorded disagreement-root scoring |
| `runtime/update_seconds` | Sharding, transfers, forward/backward pass, optimizer and metric synchronization |

Initial updates include compilation. Collection and evaluation waits remain
separate from optimizer throughput.

## Restart

Use a saved checkpoint as `--initial` and a new output directory. Keep
`--reference` fixed independently of the restart checkpoint. Optimizers,
replay, counters, schedules and TensorBoard history start fresh. Candidate
checkpoints contain weights only. All loaded model weights are retained;
the policy is not reset from the fixed reference. The previous-policy snapshot
refreshes after Q fitting before policy updates or collection resume.

## Files

| File | Responsibility |
| --- | --- |
| [run.py](run.py) | Launch settings, pass scheduling and checkpoints |
| [replay.py](replay.py) | Record decoding, replay buffers and batch sampling |
| [collection.py](collection.py) | Frozen collector and batched root scoring |
| [monitor.py](monitor.py) | Progress, atomic metadata writes and TensorBoard |
| [evaluation.py](evaluation.py) | Export and evaluation command |
| [q_fit.py](q_fit.py) | Match conversion and two full critic passes |
