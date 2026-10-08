# Training pipeline

[`train.py`](../train.py) coordinates training, verified export and duplicate
matches. See the [root command](../README.md#dataset-and-training). Selected GPU UUIDs own
training and evaluation; each stage runs in a separate process. Source
publication and training completion do not deploy services.

| Output | Contents |
| --- | --- |
| `pipeline.json`, `source/` | Fixed settings and source snapshot. |
| `candidate/` | Recovery checkpoints, final weights, validation and TensorBoard. |
| `policy/` | Verified full SavedModel and matching frozen source. |
| `evaluation/` | Opponent/candidate snapshots, game logs and paired summary. |
| `pipeline_status.json`, `*.log` | Current stage and per-stage logs. |

Rerunning the same command resumes the candidate and completed match games.
Changed settings require a new output directory. A verified export is reused;
an interrupted export is rebuilt in a temporary directory before installation.
The opponent must have a matching `source/model` tree beside or above its
SavedModel. Each player executes its own frozen feature producer.
Exports provide `serving_default` for full diagnostics and `policy` for the
five move-selection heads. Export verification compares the policy signature
against the full graph. Frozen evaluation uses `policy` when supplied, preserving
the default signature for retained opponents, and does not pad duplicate rows.
Opponent snapshots include dataset helpers stored alongside older source trees.

`--initial` starts a new run from joint weights with a fresh optimizer;
otherwise the actor and critic start fresh and `--beliefs` initializes the
shared encoder and belief heads. `--mode smoke` runs four updates;
`experiment` defaults to 12,000. `--phase-batches N` changes the total budget
for these short modes. Full mode uses 770,181,120 positions; the batch size
must divide that budget and the number of replicas. Default batch size is 9,216.

AWR defaults to `--awr-mix 1 --awr-temperature 3`; use mix zero for imitation
only. `--rollouts STORE.sqlite` enables cached one-hand settlement supervision
and confidence-shrunk paired policy win regret,
with a default 10% replay admission budget and 0.5 stale-sample weight.
`python -m model.collect_rollouts` marks disagreements from cached reviews and
runs separately budgeted CPU collection; see the
[collection contract](PAYMENT_CRITIC.md#training-and-replay).

Matches default to 10,000 games, seed zero, one worker and batches of eight.
`--games` must be divisible by four: each seed plays the candidate in every
seat. `--match-workers` and `--batch-games` control evaluation concurrency.
`summary.json` contains paired estimates and confidence intervals. Playing
strength and promotion decisions require interpreting that result.

## Implementation

`features` and `feature_vector` encode visible inputs; `critic_full` owns joint
training, recovery and validation. `export` verifies the serving graph;
`evaluate` and `frozen_policy` run duplicate games with isolated saved sources.
`belief_training` supports separate belief initialization. Dataset builders,
labels and sample readers belong to [log_dataset](../log_dataset/README.md).

[PAYMENT_CRITIC.md](PAYMENT_CRITIC.md) defines the architecture, objectives,
scoring and continuation contracts. Individual stages support `python -m`
for maintenance; `train.py` owns the full job.

## Phasic self-play

`python -m model.phasic` starts from existing joint weights and alternates:

1. Complete south games with a fixed, stochastic policy in all four seats.
2. PPO epochs over those fresh decisions, with clipped probability ratios,
   entropy, and a learned scalar value baseline.
3. Offline AWR, belief, specialist and settlement updates with forward KL
   from the frozen post-PPO action distribution on the collected positions.

The collector samples the joint call/discard and riichi/discard distributions,
records their exact native candidate masks and behavior probabilities, and
executes the sampled continuation. Forced wins and existing forced actions
retain the action bridge's behavior. Only visible features enter the actor.
PPO uses completed-game ST3 returns divided by 100, with no discounting or
bootstrapping; this is a Monte Carlo baseline implementation, not GAE. The
separate scalar value head is training-only. Privileged paired rollout targets
are not used as PPO experience.

A second forward KL anchors both phases to the initial actor (`--human-kl`,
default 0.01). This preserves proximity to the chosen human-derived initializer;
it is distinct from the moving post-PPO anchor (`--auxiliary-kl`, default 1).
These coefficients are experimental. KL penalties are soft constraints and do
not certify human likeness, strength, or exploitability. `--target-kl` stops
additional PPO epochs after the measured behavior KL crosses its threshold;
it does not undo an epoch or impose a hard bound.

Select a training GPU by UUID in `CUDA_VISIBLE_DEVICES` before starting the
command. This initial implementation uses one GPU and current-policy self-play.
Historical opponent populations and distributed collection are not implemented.
For example, after selecting the device:

```sh
python -m model.phasic \
  --data /path/to/data \
  --initial /path/to/final.weights.h5 \
  --output /path/to/phasic-pilot \
  --cycles 10 --games 8 --batch-size 128 \
  --ppo-epochs 4 --auxiliary-updates 32
```

`--games` is complete games per cycle, not a decision exposure budget. Both
optimizers, the joint network, scalar baseline and completed-cycle counter are
checkpointed. Restarting the identical command restores the last complete
cycle and recollects any interrupted cycle; old PPO batches are never replayed
across cycles. TensorBoard and `status.json` report phase losses, behavior KL,
post-auxiliary snapshot KL and decision counts. `final.weights.h5` uses the
existing joint-weight export contract; evaluation and export are separate.
The existing `train.py` offline pipeline remains available.
