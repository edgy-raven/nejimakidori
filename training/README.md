# Train, export and evaluate

Start with [setup](../README.md#setup) and a [built dataset](../log_dataset/README.md).
Use its `data/` directory in the commands below. The [architecture guide](../model/ARCHITECTURE.md)
explains the policy, beliefs and settlement model.

Training requires CUDA GPUs visible to TensorFlow. Select devices by UUID with
`CUDA_VISIBLE_DEVICES`; batch and phase allocations must divide evenly across
the replicas. Initial weights must match the current architecture and feature
schema. Use the [released distillation weights](../weights/README.md) as the
starting checkpoint for candidate AWR.

## Candidate replay schedule

Every completed cohort preserves weights in
`checkpoints/<cumulative-exposures>.weights.h5`. Existing snapshots cannot be
overwritten; `latest.weights.h5` is an atomically replaced relative symlink.
Stage snapshots and stage-boundary evaluations remain separate. These are
weights-only checkpoints, not optimizer or replay snapshots for mid-AWR resume.

AWR reports `gradient/local_norm`, the local replica's pre-clipping gradient
norm (averaged across replicas in distributed reports). Non-finite norms fail
before the optimizer update. Existing per-tensor clipping is unchanged; this
metric is not the norm of the globally aggregated gradient.

`training.candidate_training` warm-starts joint weights with fresh optimizer state
and fresh counters. It runs six passes; `N` is the exact number of positions
in training partitions 0–6, including the final partial batch:

| Pass | Objective | Recorded / generated positions |
| --- | --- | --- |
| 1 | Shared AWR actor, V and settlement objective | 100% / 0% |
| 2–6 | Shared AWR over mixed replay | 80/20, 60/40, 40/60, 20/80, 0/100 |

The first pass visits every training record once. Each later pass is an
`N`-exposure budget, including sampled replay; it is not a promise to visit each
generated position once. `--record-fractions` configures five nonincreasing
record shares in [0, 1]; it may end above zero to cap self play below 100%.
The defaults above and `--target-fraction 0.4` are starting
settings, not experimentally selected optima. No PPO or expert anchor is added
to this candidate schedule.

Include a dedicated collection GPU in `CUDA_VISIBLE_DEVICES` and pass its UUID
as `--collection-device`. The learner uses the other visible GPUs. Collection
uses a separate frozen joint model and prefetches at most one cohort while AWR
updates run. After the first cohort of each pass, generated data is one cohort
behind the learner; actor, V and action-utility weights stay fixed together
throughout each collection job. Evaluation remains a pass boundary barrier.

Within **each** source, 40% of positions are drawn from its targeted pool and
60% from its ordinary stream/pool. This is an oversampling allocation: ordinary
sampling can also encounter targeted states. Cumulative quotas preserve both
source and target allocations across partial batches. All four strata enter
one shuffled minibatch and one globally normalized AWR update/optimizer.

Targets use the same observable predicates for both sources:

- Keiten: actual shanten ≤1 with at most 16 live-wall tiles.
- Tenpai: actual shanten ≤0. Stored shanten fields encode actual shanten +1.
- Calls: optional response and own-kan decisions, including pass choices.
  The corpus focused sampler now derives this same eligibility from decision
  phase; stored fast-call metadata does not narrow the sampled opportunities.
- Disagreement: sampled discard states comparing the max-Q action with the
  legal action whose Q is closest to V. Both use current-hand projected ST3.
  Distinct choices qualify when their Q gap is at least 0.25 ST3 or the
  closest action is still at least 8 ST3 from V. Identical choices do not
  form an action pair. Score 1/64 of discard states to bound inference cost.
  Critic preferences select experiments, not correctness labels.

Generated roots are captured during fresh self-play. Reservoir sampling keeps
one eligible position per natural game, then scores those roots in batches.
Each root is assigned an explicit path. Calls compare the policy-preferred
legal call with pass (own kan with decline). Tenpai compares keeping tenpai
with breaking it; keiten compares preserving the best available shanten with
worsening it. In each hand group, choose the highest-policy-probability legal
action, including riichi/discard variants. Skip roots without both groups.
Only disagreement compares max-Q with closest-to-V, skipping identical choices.
Calls retain their own path when they overlap hand categories; overlapping
hand/disagreement categories are sampled uniformly before reservoir selection.
`--root-copies` is the number of repeated rollouts **per action**, default 1:
one branch for each action, spending the branch budget on more distinct roots.
Larger repeat counts remain available for explicit experiments; no mean-target
or confidence-based action selection is introduced by this setting.
The two actions use the same fork seeds: concealed hands and the observed
prefix remain fixed, and each repeat resamples the unseen wall while retaining
revealed indicators. Calls are attempted before ordinary priority resolution.

The number of selected roots is reduced approximately in proportion to
`--root-copies`, targeting the former two continuation-games per natural game
when enough distinct pairs exist. The fixed fresh-position budget and replay
source/target shares remain unchanged. Branch counts and position counts are
not interchangeable, and repeated samples are correlated within each root.

Only the selected initial action is forced. `paired_root` marks that row and
`root_path` records keiten=1, tenpai=2, calls=4 or disagreement=8 (zero for
ordinary rows). Its behavior distribution records the intervention; continuation actions are
sampled normally. Realized current-hand settlements supervise the critic and
AWR. V remains a replay-mixture baseline, including forced-root examples;
these are not unbiased current-policy V observations. Paired focused collection
is AWR-only. Natural trajectories remain eligible, and only suffix decisions
enter replay. This path does not reconstruct roots from recorded features.
Independent root streams share inference batches and retain their own sampling
streams and settlement targets. Multiprocess collection restores feature rows
to trajectory order before assembling replay labels.

Candidate replay retains individual realized hand returns, not an average over
the repeated root action. Root markers are collection metadata and are not
stored in compact replay; no paired return estimator currently enters AWR.
Branches continue to match end, adding ordinary later-hand training examples
alongside the forced root. Matching wall seeds therefore does not imply that
training consumes a paired action comparison or eight-runout mean target.

Ordinary recorded examples stream through the corpus rather than a fixed small
cache. A bounded targeted-record reservoir is built during pass 1, then refreshed
from the ordinary record stream. Generated replay uses bounded recent pools,
shuffled within collection chunks so a chunk's final fork cannot dominate simply
because it was appended last. `--replay-positions` (default 524,288) is capacity
**per pool**; three memory-mapped pools live under the run output. Use artifact
storage for the run. `--fresh-positions` controls
new collection before each cohort; `--cohort-exposures` controls replay updates.
The final partial cohort scales its fresh-position requirement proportionally.
Collection shrinks its last game batch using observed positions per game and
continues until that requirement is met, avoiding an unnecessary full batch.

A required evaluation argv JSON runs immediately after every saved pass and
must succeed before the next pass starts. For example, `evaluation-command.json`:

```json
["python", "-m", "training.candidate_evaluation", "--weights", "{weights}",
 "--output", "{output}", "--opponent", "/path/to/incumbent/export",
 "--games", "512", "--seed", "2031010000", "--devices", "GPU-UUID"]
```

Use the same fixed incumbent and evaluation seeds across phases, with fresh
reserved seeds for a subsequent confirmation. The hook exports each checkpoint
and plays rotated-seat duplicate matches. It does not automatically promote a
screening winner or guarantee that the candidate improves.

```sh
CUDA_VISIBLE_DEVICES=GPU-TRAIN0,GPU-TRAIN1,GPU-TRAIN2,GPU-TRAIN3,GPU-COLLECT \
python -m training.candidate_training --collection-device GPU-COLLECT \
  --games 408 \
  --initial /path/to/release/final.weights.h5 \
  --data /path/to/dataset/data --output /path/to/artifacts/candidate \
  --evaluation-command /path/to/evaluation-command.json
```

Evaluation workers use `--devices` GPU UUIDs in round-robin order.
`training.candidate_evaluation --batch-games` forwards the number of concurrent
games per worker (default 8). Benchmark worker allocation and batch size
separately from optimizer throughput; checkpoint evaluations pause training.

Every candidate run begins at recorded AWR from `--initial`, using a new
output directory, fresh optimizers, zero counters and fresh TensorBoard history.
The released distillation checkpoint is the starting model; candidate training
does not load another model or recreate that checkpoint. An existing output
directory is rejected. Candidate runs have no optimizer-state resume path.

Each AWR pass uses one constant rate. Rates interpolate by self-play share from
`--awr-learning-rate` at the first pass's share to `--awr-final-learning-rate`
at the final share. Repeated shares use the same rate; if every share is equal,
the initial rate is used throughout. Recorded AWR followed by 20%, 40%, 60%,
80%, 80% self-play, with endpoints 5e-6 and 3e-6, gives 5e-6, 4.5e-6, 4e-6,
3.5e-6, 3e-6, 3e-6. Optimizer state persists across pass boundaries.
`--awr-pull` adds a nonnegative coefficient times KL(current || previous)
on the legal action distribution in every AWR update. The previous actor is
frozen for each cohort and refreshed at the next boundary; recorded AWR uses
`--cohort-exposures` boundaries too. Pull diagnostics are `mirror_kl` and
`weighted_mirror_loss`. A zero coefficient disables the reference forward pass.
V fits the sampled replay return distribution; it is not guaranteed to equal current-policy
full-match value. Returns retain observed hand settlement plus fixed ST3
continuation. V and actor are updated jointly, with detached advantage weights;
this differs from original AWR's separate value fitting and policy regression.
Shared encoder changes can move V and the settlement predictor during policy
training. Inspect value MSE/bias, payment NLL, weight clipping and effective
sample fraction alongside matched-game results at each boundary.

For a fresh mixed-AWR run from an existing checkpoint, use `--mixed-only`
with `--initial`. This skips record-only training, initializes
fresh optimizers and replay, and starts exposure counters and TensorBoard at
zero. The five nonincreasing
`--record-fractions` may end above zero: `0.8,0.6,0.4,0.2,0.2` caps self play
at 80% while retaining the five-pass exposure budget. Collection loads the
initial architecture and receives the current learner weights before each
prefetch cohort.

## Phasic training

`training.trainer` combines expert records with fresh self-play. This small example
starts from joint weights and exercises every phase:

```sh
CUDA_VISIBLE_DEVICES=GPU-UUID python -m training.trainer \
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
The rollout objective is selectable with `--rollout-objective`: `ppo`
(default), `ppo-awr` (alternate PPO and AWR passes on the same cohort), or
`awr` (all cohort passes use AWR). With `--ppo-epochs 2`, the latter two
compare PPO then AWR against two AWR passes at equal exposure budgets.
Recorded and generated replay use one AWR actor/value/settlement update and
one AdamW optimizer, separate from PPO and supervised training. Its learning
rate currently uses `--ppo-learning-rate`; the supervised cosine schedule counts
only foundation updates. The
recorded-data and native-rollout adapters only construct its inputs. AWR uses
detached `(observed return - current V) / temperature` in ST3 units, capped
positive regression weights, and settlement supervision on chosen actions.
`--awr-temperature` controls both replay sources and defaults to 3 ST3 units.
The former dataset-specific advantage helper and rollout-specific AWR loss,
optimizer name and temperature option have been removed. Old optimizer-state
checkpoints are not a supported resume format; warm-start with model weights.

AWR does not train hindsight belief/completion auxiliary losses or consume
separate auxiliary replay. Supervised foundation updates retain those losses.
The AWR source mix therefore cannot silently change the actor objective.
AWR-only cohorts skip PPO baseline scoring, standardized advantages and
behavior-KL scans. PPO retains its fixed, standardized cohort advantages.

Existing `ppo` budget, checkpoint and metric names count rollout updates
for all three objectives; the plan records the selected objective. Cohorts
are discarded after their configured passes; there is no cross-cohort replay.

The AWR total includes its prefix. PPO's budget is
`(awr_exposures - awr_only_exposures) × ppo_exposure_ratio`.
Positive finite ratios may exceed 1; with fixed cohort size and passes,
this collects more fresh cohorts. Auxiliary expert replay accompanies every
PPO update, so increasing this ratio also increases auxiliary replay.
The final AWR budget is additional, giving `6N` overall when foundation is zero.

`--expert-sampling`, `--replay-sampling`, and `--final-awr-sampling`
select expert-record distributions independently for middle/prefix AWR,
PPO auxiliary replay, and final AWR. Each accepts `mixed` (default),
`natural`, `focused`, or `placement_gap`. These options do not select
self-play starting states.

In this phasic entry point, each cycle finishes recorded AWR, collects games with
the updated policy, and runs PPO on that cohort before discarding it.

Both phases use a state-only value head on the shared pre-action encoder.
Expert records carry the actual post-hand scores, ranks, next dealer/hand,
terminal flag and outstanding deposits. AWR and PPO apply the same fixed
ST3 continuation forecast to these observed endings. This is a hand-return
target with fixed continuation, not a full-match Monte Carlo or TD(lambda)
target. The value head predicts ST3 / 100 and learns by squared error against
these returns: expert replay during offline updates, fresh returns during PPO.

AWR uses detached `(observed_return - state_value) * 100 / temperature`
for every decision, including responses and win/pass. Exponential weights are
capped at 20 before batch normalization. The default temperature remains 3 ST3.
The actor cannot backpropagate through these weights. Value fitting and actor
regression share an update and encoder; AWR weights are recomputed each batch,
not frozen after a separate value-fitting phase as in original AWR. PPO fixes
the cohort's baseline and standardized advantages before its update epochs.

The action-conditioned settlement predictor remains supervised separately
within the joint loss, but no longer supplies AWR or PPO baselines. Its derived
action utilities need not match the replay value head. Monitor held-out value
MSE and signed bias, payment NLL and policy metrics across phase boundaries;
gradient detachment does not prevent drift through shared encoder updates.

The dataset must contain `critic/settlement` and checkpoints must include the
state-value head. The existing observed-return corpus already satisfies this
contract; no full feature rebuild is needed.
There is no missing-field fallback or automatic partial checkpoint loading.

Only supervised foundation uses `CosineDecay`: the first
2% of updates warm up from 10% to the peak rate, then cosine decay reaches
10% of the peak at the end. AWR and PPO use separate constant-rate AdamW
optimizers. All optimizers use weight
decay `1e-4` and per-variable `clipnorm=1.0`. Repeated tile-embedding gradients
are summed locally before clipping and synchronization, matching a dense
one-hot projection. Distributed loss reductions use
TensorFlow autodiff.
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
Use `python -m training.trainer --help` for all options.

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

python -m training.evaluate \
  --new /path/to/run/policy/saved_model \
  --old /path/to/opponent/saved_model \
  --output /path/to/run/evaluation \
  --games 5000 --workers 1 --devices GPU-UUID
```

Export needs a dataset with a training partition. It verifies finite outputs,
normalized distributions and SavedModel round-trip parity, and saves matching
source beside the model.

Each evaluation player uses its own frozen features. The opponent therefore
needs its matching source snapshot beside or above its SavedModel. New exports
include `source/model`, `source/training` and `source/log_dataset`; retained
production snapshots keep their original worker location. Matches rotate
each seed through four seats, so the game count must be divisible by four.
Rerunning an unchanged evaluation resumes completed games. `summary.json`
contains paired estimates and confidence intervals.

Use the [service guide](../service/README.md) to serve the resulting export.

For schedule comparisons, `--cycle-blocks 1` alternates expert AWR blocks
with self-play blocks. Setting it to the entire middle-stage block count
runs all expert AWR first, then self-play. Both consume one continuous
expert stream; PPO still collects fresh cohorts at `--cohort-exposures`.

## Self-play-only training

`python -m training.online_training --initial WEIGHTS --output NEW_DIRECTORY
--exposures N` trains from fresh games without reading expert records.
Model setup uses fresh native simulator inputs. PPO/AWR returns and payment
critic labels come from self-play; expert belief replay is disabled. Existing
weights may initialize the model, with new optimizer state and counters.

`--objectives ppo`, `awr`, `ppo,awr`, or `awr,ppo` selects cohort objectives;
`--phase-cohorts` determines how many cohorts each objective uses. `ppo-awr`
alternates objectives within cohort passes. `--ppo-epochs` controls reuse.
These returns still use the existing fixed hand-end ST3 continuation forecast.
`--focus natural`, `keiten`, `tenpai`, `calls`, `disagreement`, or `targeted`
selects collection. `targeted` takes the union of the four target categories.
Comma-separated focus sequences advance with `--phase-cohorts`, alongside the
objective sequence. Focused collection uses the paired repeated-rollout
contract in the candidate schedule above; `--root-copies` counts repeats per
action. All non-natural focus modes require AWR. Keiten selects actual shanten
<=1 with <=16 live-wall tiles; tenpai selects actual shanten <=0 (stored
thresholds 2 and 1). Calls select response and own-kan decisions. Disagreement
uses max-Q versus closest-to-V, with the same 0.25/8 ST3 thresholds as recorded
sampling. Report unique roots, repeated branches and training positions
separately.

Export without records using `python -m model.export --weights WEIGHTS
--self-play-seed 941 --output OUTPUT`.

## Value experiments

See [value experiments](EXPERIMENTS.md) for repeated-continuation collection,
value-head fitting, architecture comparisons and independent confirmation.
