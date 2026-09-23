# Nejimakidori model card

This card describes the current source model and its fixed execution recipe.
It is documentation, not an editable run configuration. Fixed values live in
the root feature-vector contract and model implementation. The only execution
mode switch is `--experiment`: a fast smoke test
instead of the full pipeline. Both training entry points also expose
`--dealership-advice-strength` for controlled advice comparisons. Neither mode loads a settings registry.

## Intended use and model

Four-player Japanese riichi mahjong action selection from the player's hand
and public game state. The large structured suit ConvNet encodes shared
candidate hands; attack and defense experts feed a learned final decision
pass. Rivers combine tile embeddings and discard metadata through three
64-channel residual temporal convolutions (dilations 1/2/4), retaining ordered
positions and masked global maxima.

Legal chi/pon compositions consume an available red five. Physical red tile
identity remains relevant to scoring and the browser's call chooser. Native
legal win and nine-terminals offers are currently accepted deterministically.

## Fixed training recipe

| Setting | Value |
| --- | --- |
| Training devices | GPUs 1–4; GPU 0 remains production |
| Expert exposure | 1,540,362,240 requested position presentations |
| Expert batch | 5,120 total, 1,280 per GPU |
| Memory cap | 23,552 MiB per GPU |
| Training phase order and exposure | Natural 25% → focused 50% → natural 25% |
| Optimizer | AdamW; peak learning rate 0.0002; 2% warmup, hold through 75%, final-quarter cosine decay to 0.00002 |
| Precision | Existing mixed bfloat16 policy |
| Main actor mixture | (BC + payoff coefficient × blended placement/advice AWR) / (1 + payoff coefficient) |
| Payoff coefficient | 0 through 10% of updates, ramps to 2 at 60% |
| AWR | Temperature 1; weight cap 20; global weight normalization |
| Expert actor mixture | Half BC, half AWR, plus structural regret |
| Final action-value critic MSE | 0.1; training-only head; joint gradients through shared normalized actor features |
| Policy entropy coefficient | 0.01; negative entropy on active legal-action distributions |
| Dataset workers / games per shard | 40 / 32 |
| Evaluation | 10,000 seat-balanced games, 8 workers, 8 games per batch |

Budgets round down to complete batches and focused blocks. The run's actual
exposure is recorded in `candidate/run_config.json`. Auxiliary objectives, sampling, native search
bounds and architecture stay defined in code; this card does not introduce
runtime switches for them.

The shared board trunk projects inputs to 512 units, then applies four
pre-normalized residual MLP blocks (512 → 2,400 GELU → 512), followed by a
512-unit GELU board-state layer and the existing output normalization.
The actor has 23,150,027 parameters, within 0.13% of the reference's
23,121,122. The input projection uses 6,641,664 parameters, reallocating
capacity from the previous wide projection into depth. Candidate board
context remains 192 units and the shared candidate projection is 384 units.
This architecture applies to fresh training; frozen runs retain their
original architecture. Correctness checks do not establish playing strength.
On 2026-09-22, six targeted tests passed, including nonzero finite gradients
through all eight residual-block dense layers and a train/export/reload
smoke test. A CPU-only bfloat16 forward benchmark (32 repeated real decision
features, four intra-op threads, three warmups and 30 timed batches) measured
median latency of 17.44 ms versus 16.00 ms for the previous architecture.
This is about 9% slower on that CPU fixture; GPU latency and trained quality
remain unmeasured. Artifacts: `nejimakidori-audits/depth-20260922/`.
Payment predictions have separate ron, tsumo, draw and bank channels.
Payment/time/risk structural priors and their extra forward probes remain
removed. Final-placement supervision uses the score-aware shrinkage below.

## Data and objectives

Expert data comes from eligible archived 2024/2025 Houou games. AFK detection
excludes the whole affected hand. Focused sampling and natural finishing use
one current record contract.

Training combines imitation, separate Monte Carlo placement and hand-point
critics, AWR-style actor regression, and structured auxiliary supervision.
Future or concealed opponent information provides labels, never policy inputs.

Q and V each have two zero-initialized outputs: final-placement residual and
signed-log remaining-hand utility. Placement uses actual final ranks with utility
[1, 0.6578947, 0.3157895, -1], subtracting the shared score-aware prior below.
This critic does not use the auxiliary rank head's smoothed labels. Points
use the existing typed terminal-payment labels, including remaining bank
transfers and excluding honba, consistently for observations and alternatives.
Each component has its own known-label mask, critic loss and baseline loss.
Their losses are averaged to retain the previous total critic coefficients.
The point utility is `sign(p) * log1p(abs(p) / 10000)`, applied after netting
payments and removing honba, including known ron and abort alternatives.
Both point Q and V predict expected transformed utility; advice uses their
transformed-space difference. This deliberately compresses payout preferences
and must not be interpreted as an unbiased estimate of expected raw points.
For example, 8,000 and 12,000 map to 0.588 and 0.788; losses use negative values.

Board payment supervision uses the same signed-log amount axis for both
unconditional and conditional CDF-distance losses. Categorical amount labels,
cross-entropy, transfer occurrence, payment-type channels and raw inference
amounts retain their original definitions. The transform changes training
loss geometry, not the payment tensor's conservation or currency units.
The run configuration records both transforms; no corpus regeneration is needed.

Known immediate ron payments train point alternatives even when final placement
is unknown. Exact terminal placement is decoded from existing ron utility
labels only where their placement share is nonzero. Known abort deposits train
point alternatives at every dealership count. Unknown continuation placement
remains masked; no score-prior estimate is labeled as an observed outcome.
The existing record schema and generated corpus remain valid and unchanged.

The main AWR advantage is placement Q-residual minus placement V-residual;
the common prior cancels. A separate advice AWR uses point Q minus point V,
scaled by 1 / 0.5 / 0 for two / one / zero future dealerships, excluding the
current dealership. Both weight calculations are detached from the critics,
cap exponential weights at 20, and supervise only recorded decisions.
The advice strength is fixed per run: default 0.25, configurable through
`--dealership-advice-strength` (0 disables it). The payoff actor loss is
`(placement_AWR + strength * advice_AWR) / (1 + strength)`, retaining the
existing BC/payoff ramp. With zero point advice the advice term becomes BC.
This default is an experiment setting, not a calibrated optimum. Separate
component errors, alternative counts, and both AWR effective sample fractions
are logged; the strength and component definitions are saved in run_config.

Specialist receipts, deal-in losses, immediate danger and their baseline
reward components also use signed-log values. Tenpai probabilities, bounded
hand-quality targets and regret keep their existing definitions. Conditional
quality heads are trained only on reached-tenpai hands; their state baselines
use the joint reach-times-quality target, zero on failed hands and masked when
unknown, matching the predicted reach-probability-times-quality estimate.

Loss groups have explicit budgets: main policy mixture 1, board supervision
0.5 times its seven-task mean, specialist policy 0.5 times the two-expert mean,
specialist supervision 0.1 times its sum, specialist baseline 0.05, outcome
critics 0.1 times their mean, and outcome baselines 0.1 times their mean.
Main-policy entropy enters as `-0.01 * H`, using active legal-action heads.
Each weighted contribution is logged separately; their sum is the total loss.
Entropy and all group weights are recorded in run_config.

These are provisional balancing defaults. A CPU diagnostic using 256 recorded
positions from eight shards and 16 updates found the previous board gradient
about 1.7 times the sum of main-policy gradient norms at the final shared
residual block, with entropy only about 0.14 percent. This motivated reducing auxiliary
budgets and increasing entropy; it does not establish mature-run balance or
playing strength. Repeating with revised weights gave ratios about 0.75 and
1.5 percent respectively. Audit artifacts are under
`nejimakidori-audits/loss-balance-20260922` alongside the repository.

These changes apply to new training, not the frozen evaluation candidate.
Correctness checks and a short train/export smoke do not establish stronger
play; compare advice strengths on held-out data before expensive game runs.

Reach-tenpai reward is 1, with no explicit tenpai-speed objective. Attack tenpai-value
quality at most 0.1, defense holding-burden cost at most 0.1, and bounded local
regret at most 0.025. Every never-tenpai hand receives the reach failure signal.
Actual receipts, ron losses and placement keep their separate coefficients.
New reach/value heads require fresh training and compatible records.

## Final-placement shrinkage

Final-placement cross-entropy and squared CDF loss share the soft target
`(1 - alpha) * observed_rank + alpha * prior`, with
`alpha = 0.95 * clip(scheduled_hands_remaining, 0, 7) / 7`.
The prior is the actor's rank marginal over all 24 Plackett-Luce seat orderings,
using scores divided by `4720 * sqrt(remaining + 1)` points. Equal scores give
a uniform prior; score gaps break symmetry. Its scale was fit on training games
only. It deliberately ignores private hand information and tie-order effects;
the observed-rank component still supplies their supervision.

Keep loss weights and valid-label denominators unchanged. Unknown final ranks
remain masked. Use scheduled hands remaining for East-only games, repeats and
extensions. South 4 and extensions have zero shrinkage, which does not force
the predicted entropy to zero. Current-hand placement, recorded-payoff critics,
AWR returns and payment supervision are unchanged. TFRecords retain observed
ranks; no regeneration is required. The prior runs only in training, without
an extra actor forward pass or an inference-time temperature.

The early-horizon criterion is whether deviations from this prior reliably
predict unseen outcomes, rather than optimizing aggregate match NLL. In the
initial East 1 hand, use 95% prior and 5% observed rank, retaining a small
learning signal for hand-specific developments. A large lead in an East 1 repeat still
changes the score-aware prior. The learned output is encouraged toward this
target, not hard-clamped at inference; other training signals and approximation
error can still produce deviations. Later shrinkage falls linearly, restoring
ordinary observed-rank supervision at zero scheduled hands remaining.

An initial 18-model CPU sweep selected 0.75 shrinkage on aggregate tuning NLL;
that criterion is superseded. A follow-up separated initial East 1 (honba zero)
from repeats. None of four retained-residual strengths showed a reliably lower
tuning NLL than the broad prior in initial East 1 (game-bootstrap intervals).
Six CPU proxy models then compared no shrinkage with full early shrinkage,
using the same 2,781 training and 952 tuning games. Confirmation used 48 fresh
shards containing 1,348 games, disjoint from every prior experiment split.

On fresh initial East 1 observations (1,342 games), the unshrunk predictor's
NLL minus prior NLL was +0.00021 with 95% game-bootstrap interval
[-0.00220, +0.00283]: no demonstrated benefit from the deviations. Full early
shrinkage reduced mean maximum rank probability from 27.35% to 26.08% and mean
absolute probability deviation from the prior from 1.49 to 0.70 percentage
points. It did not demonstrate improved initial-East-1 NLL; the purpose is
restraint where added predictive detail is unsupported. Overall fresh NLL was
1.12009 without shrinkage and 1.11647 with it. These small public-context
predictors do not establish the full shared actor's calibration or strength.
Artifacts are retained with the corresponding private run output.
A subsequent fixed-checkpoint gradient diagnostic selected 95% instead of
100% early shrinkage to preserve some outcome signal. At checkpoint step
71,500, 16 paired microbatches of eight independent games per cohort measured
`board_state.kernel` gradients through the complete saved actor. The placement
CE + CDF term kept its actual 1/7 auxiliary coefficient. In initial East 1,
placement-gradient variance fell 98.85% at 95% shrinkage versus 99.13% at 100%.
For imitation plus placement alone, variance fell 1.76%; other objectives were
not included in that combined measure. Gradient alignment with imitation
remained near zero. This supports a quieter placement contribution, not a
claim of lower variance for the complete optimizer update or stronger play.
The diagnostic uses one frozen checkpoint and small CPU batches, not the
5,120-position production batch. Gradient artifacts and paired bootstrap intervals are retained with the corresponding private run output.
The active frozen packed-reduction run retains its unsmoothed targets.

## Opponent ukeire shrinkage

Opponent structural-ukeire BCE targets blend the observed binary label with
an empirical shanten-conditioned tile prior. Prior weight is
`0.5 * live_wall_count / 70`: 50% at a full wall, 25% at 35 tiles and zero at
wall exhaustion. This is within-hand progression, independent of East/South
or repeats. It is a regularization schedule, not a claim that a depleted wall
reveals the opponent's exact hand. Own-hand ukeire and action-efficiency labels
are unaffected. Conditional shanten branches still use true shanten only for
supervision, never as an actor input.

The four shanten cohorts use seven tile classes: suited 1/9, 2/8, 3/7, 4/6,
5, winds, dragons. Suits and reflected ranks share fitted probabilities; do
not smooth all tiles toward 50%. Constants in `objectives.ukeire_loss` were
estimated on training games only and rounded to six decimals. Keep the 9:1
conditional tenpai/non-tenpai weighting and all unknown-label masks. Report
wait recall, precision and Brier scores against original binary labels, not
softened targets. Preserve exhausted/fifth-copy structural information: this
head is structural ukeire, not physically available legal-ron danger. No extra
forward pass, inference-time smoothing or record regeneration is introduced.

Fast calibration sampled 64 original record shards, retaining one deterministic
random decision per game/hand/player. Game-disjoint splits: 1,038 training,
363 tuning, 358 test games. Nine CPU predictors (three seeds, strengths 0,
0.5, 0.95) used public visible-tile counts, opponents' rivers/melds, riichi,
wall and winds, with four conditional shanten branches. Epoch and strength
were selected by tuning 9:1 weighted BCE. Test weighted BCE improved from
0.222538 to 0.221256; paired game-bootstrap change interval
[-0.001549, -0.001002]. Early-hand tenpai top-5 recall rose from 24.31% to
25.32%; late-hand recall was essentially unchanged (38.33% to 38.22%).
The 95% variant had worse weighted BCE and late recall than 50%. These are
small-model results, not evidence of stronger full-actor play. Artifacts are retained with the corresponding private run output.
Active frozen training retains its original hard ukeire targets.

## Execution

Run it from the repository root with `python train.py --root RUN --reference REFERENCE --data DATA`, or replace `--data DATA` with `--archives ARCHIVES`.

The pipeline takes an output directory and either archives or prepared records.
Without `--experiment` it runs the full recipe above. With `--experiment` it
uses two archived games (when building records), one record worker, GPU 1,
batches of four, two natural updates, four focused updates and two final natural updates, followed by
four evaluation games with one worker. Both modes use the same feature
calculation, architecture, losses, optimizer, phase order and export checks.
The smoke test checks execution, not playing strength.

Every run starts a fresh model and optimizer. There are no warm-start,
component-transfer, resume or hyperparameter options. Runs freeze this
Markdown card with their source. `plan.json` records source/reference identity
while `games/evaluation.json` records the automatically chosen seed.
`candidate/run_config.json` records actual exposure and optimizer details. The final actor is saved as `model.keras`
and exported as `saved_model/` before game evaluation.

## Evaluation and limitations

The candidate and reference retain their own frozen feature producers and
native runtime. Evaluation uses RiichiEnv's native duplicate south format:
each seed rotates the sole candidate through all four seats against three
references, with fixed per-seat draws and exactly eight scheduled hands.
Calls cannot redirect another seat's draws; kan consumes the caller's queue.
Dora/ura stay on the fixed dead wall. There are no dealer repeats, early match
endings or extensions; honba follows normal win/draw progression while points
and riichi deposits carry forward. A dealer-renchan outcome pays 400 all, then
the dealer advances. The pinned RiichiEnv dependency provides this duplicate format.

Report placement, rank points, score and
paired seed-cluster uncertainty only from complete rotations. Artifacts record
`riichienv_duplicate_south` and cannot resume ordinary shared-wall results.
This modified riichi benchmark is separate from ordinary match strength.
Report results from the run's game artifacts. Correctness checks are not
playing-strength measurements, and current source changes have no implied quality result until
trained and evaluated. Training does not automatically promote a model to
production. Existing frozen runs retain their original recipe and code.

Top checkpoints use exact complete-action imitation accuracy over disjoint
100-update windows (the last window in a phase may be shorter). Accuracy
records are tracked separately for natural_start, focused and natural_finish;
a global 30-minute cooldown limits saves. Improvements during cooldown update
the record but are not queued for later saving. A later save needs a new record.
Every saved actor remains under `candidate/top/<phase>-<step>/model.keras`
with `checkpoint.json` recording the selection metric and step. These are
training-accuracy checkpoints, not playing-strength selections or optimizer
resume states. The final model/export are saved unconditionally as before.

Restart consistency audit (2026-09-22): final learner weights, including the
critics, are retained in `candidate/learner.weights.h5`. The run snapshot and
evaluation snapshots include feature-vector and log-dataset dependencies.
Duplicate evaluation routes each seat exclusively to its assigned policy and
retains the fixed reference's continuation adapter.
