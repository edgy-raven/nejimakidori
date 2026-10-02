# Joint policy and payment critic

`python train_joint.py` trains imitation and the current-hand payment critic
through one shared board/hand encoder. No self-play or frozen-teacher cache
is required. Full runs use half the production position exposure; rebuilt
training-partition counts determine the phase proportions.

## Architecture and losses

`model.critic` owns the public and hand context layers, payment distribution,
joint policy wrapper, and payment losses. `model.payment_scoring` owns the
han/fu tables and scoring constraints; `model.payment_bonus` produces observed
scoring-scenario labels. The top-level `export_policy.py` owns the serving
wrapper and export verification. Specialist sampling is part of
`model.training_targets`; shared wait and completion scoring lives in
`log_dataset.scoring`.

`JointPolicyPayment` has two board residual blocks, candidate-hand and river
encoders, 256-unit attack/defense branches and a 256-unit final pass. The
pre/post-riichi rivers remain separate padded tensors. Policy inference uses
visible features only and does not evaluate the payment decoder.

Candidate encoding explicitly includes a haitei-draw flag (excluding rinshan),
a houtei-response flag, and two four-seat indicators for the projected last
ordinary draw before and after the candidate action. Projections assume no
further calls or hand termination. Response pass continues after the discarder;
chi/pon changes the next draw to the seat after the caller; kan removes one
live-wall tile for the replacement draw. No remaining ordinary draw gives an
all-zero seat indicator. These ten channels are derived on device from existing
inputs, so serialized records and serving inputs do not change. They feed both
specialists, the final policy and the shared payment-critic candidate encoding.
The last-draw channels add ten projection inputs. Combined with removing the
action-origin flag, the old 780×384 kernel becomes 789×384. A one-time weight
splice drops old row 586, retains row 587 (call option), inserts ten zero rows
at [587,597), then copies the old board context rows [588,780). All other
critic arrays remain unchanged. Zero initialization makes the new channels
initially neutral; training must learn their use. Removing the old origin
contribution and merging policy classes intentionally changes the old policy.
Frozen runs and existing releases retain their original architecture.

There is one action-conditioned payment predictor. It separates scoring
bonuses and han/fu scenarios, then produces payment distributions. Point means
come from that distribution; net variance additionally needs payment-cell
correlations and held-out calibration. There is no duplicate point-value head
or scalar final-rank baseline.

The current joint loss is:

- Legal-action imitation and existing entropy regularization; guided updates
  blend imitation with AWR as described below.
- Payment critic loss, weight 0.1: payment likelihood, signed-log point error,
  bonus and
  scoring supervision, immediate danger and discard ranking.
- Existing board and specialist auxiliaries at their existing weights,
  including shanten, hand-end/final placement and attack/defense supervision.

Point error is `mean((s(predicted_mean) - s(realized_net_payment))**2)`,
where `s(p) = sign(p) * log1p(abs(p) / 10000)` in raw point units.
It is logged as `point_log_mse`; `point_mse` remains a raw-scale diagnostic
(in squared units of 10,000 points). This transforms each value before taking
the error, not the error afterward. Payments, scoring, and AWR utility stay
in their existing units. The point term emphasizes relative discrepancies
at large magnitudes; it no longer has the arithmetic mean as its standalone
optimal predictor. Payment likelihood still supervises the full distribution.

Composition lives in `critic_full.objective_terms`; critic loss composition
lives in `critic.objective_metrics`. Both policy and critic gradients
reach the shared encoder. These weights are retained settings, not established
optimal choices.

**Uncertainty-aware ST3 AWR** is implemented in `critic_utility.py` and
`critic_full.objective_terms`. Full training requires `--utility settings.json`.
The settings are `every: 1`, `reference_weight`, `reference_every`,
`temperature` (ST3 units), `placement_scale` (points per square root of remaining
hands plus one), `variance_scale`, and the calibrated 19-by-19 actor-payment
`correlation` matrix. Correlations and scale use training/calibration data.

The selected candidate uses temperature 3 and placement scale 4,720. At each
legal action, the critic supplies actor net-payment mean and variance. A
17-node Gaussian quadrature projects those uncertain transfers onto current
scores, sharing the opposite transfer equally across the three opponents.
The shared `objectives.placement_prior` adds Gumbel continuation uncertainty
and derives coherent rank probabilities. Utility is expected point gain/1,000
plus rank bonuses `(125, 60, -5, -255)`. Current-score constants cancel from the
advantage. There is no extra constant variance penalty or first-place tilt.

This is an approximate ST3 utility, not an exact game transition model. It
omits exact ceiling/tie/continuation rules, individual future dealerships, and
payer/recipient structure beyond actor moments. Immediate danger and draws
remain supervised and contribute through payment forecasts. A broad outcome
distribution is allowed; its uncertainty changes risk preference with scores.

Every update after warmup scores the active legal alternatives in fixed chunks
of 512. Direct first/second moments of the han/fu and bonus mixture avoid full
histograms for the 19 actor-relevant cells. Recorded-action supervision keeps
all 40 payment distributions. AWR reuses the candidate context already computed
for supervision. Scoring and utility reduction are compiled on the GPU and run
outside gradient-tape recording. The current policy expectation over the same
legal actions is the baseline; centered, temperature-scaled advantages are
stopped before forming positive weights capped at 20.

The reference coefficient stays zero for 10% of updates and rises linearly to
`reference_weight` at 60%. Convert that coefficient `w` to
`w / (reference_every + (reference_every - 1)*w)`, then blend imitation and AWR
as `(CE + coefficient*AWR)/(1+coefficient)`. With reference weight 2 and cadence
20, the plateau is 1/29, giving a 1/30 AWR blend every update. This preserves
the reference's average unified blend; it does not preserve optimizer dynamics
or imply equivalence of the new reward. Critic and auxiliary weights stay fixed.
Logging remains every 20 updates. Diagnostic modes may omit utility for controls.


Specialist AWR and its baseline have been removed. Attack/defense use
allocated imitation, global legal-action entropy (0.01), and global regret
(0.025), retaining the existing specialist group weights. Allocation labels
also route final-policy imitation gradients through each specialist branch,
excluding indirect final-policy AWR from those branch parameters.
Shared encoder and final-policy training retain their ordinary coverage.
Unified critic AWR remains independent of this allocation.

Uke regret differentiates legal actions only for eventually-tenpai hands.
Never-tenpai hands receive flat regret 2, above the normalized maximum 1;
this constant has no action-preference gradient. Unknown labels are excluded.
Sakigiri uses a flat never-tenpai target 14,689, above the conservative
136 × 36 × 3 = 14,688 holding-cost bound. Its supervised target is
`log1p(cost) / log1p(14689)`, preserving the strict gap in float32.

The existing final-placement head remains supervised and trains the shared
encoder. It is state-conditioned: final-rank loss does not directly train the
payment decoder or turn these point advantages into final-rank action values.
No duplicate final-rank scalar head was added. The ST3 prior is a fixed utility
construction using predicted payments; the learned placement decoder remains
a separate research candidate.

Final-rank placement points `(125, 60, -5, -255)` remain the evaluation goal;
full ST3 also includes the final-score contribution and ceiling. The Gaussian/Gumbel
projection remains a surrogate for this exact evaluation payoff.

## Scoring lookup contract

Ordinary scoring has 65 states: han 1–4 crossed with the fourteen fu values,
then one fu-independent state for each han 5–13. Fu zero marks those limit-hand
states; the label producer canonicalizes actual fu at five or more han. Seven
separate true-yakuman states retain allocated payments, including liability.
The decoder therefore has 72 outputs rather than 189.

Static tables enforce fu constraints using ron/tsumo, open/closed scoring mode,
and exposed meld fu, then map states to rounded payments using dealer/payer
status. Closed-hand 20-fu pinfu tsumo and 25-fu chiitoitsu exceptions remain;
five-plus-han settlement has no fu constraint. The existing bonus-scenario
mask still enforces open-hand restrictions. These are scoring constraints,
not a complete enumeration of attainable hands or yaku combinations.

Critic dataset summaries and build plans carry `critic_scoring_states`; the
reader requires an exact match. Rebuild or explicitly remap old score labels
before training. Initialization must also match the smaller projection. An
experimental teacher projection fits the old masked aggregate logits using
training positions only; it approximates the old forecast rather than exactly
preserving it. Runtime speed does not establish equal predictive quality.

## Data, initialization and runtime

The historical corpus contains 128,230,481 positions across 363,056 games.
Rebuild records and selection metadata before training: specialist allocation,
focused eligibility, never-tenpai sakigiri targets, and passed-tile safety inputs
changed. Old summaries are rejected. Historical weights require an explicit
one-time projection conversion before strict loading. Earlier zero-column
expansions preserved the old feature contributions; the current origin-removal
and last-draw splice has the narrower preservation contract described above.
There is no inference compatibility path.

| Mode | Natural start + focused + natural finish | Purpose |
| --- | --- | --- |
| smoke | 1 + 2 + 1 updates | Runtime and correctness checks |
| experiment | 3,000 + 6,000 + 3,000 updates | Comparative training experiment |
| full | Natural/focused/natural in 0.5/4/0.5 pass proportions | Half production exposure |

A pass uses partitions 0–6 only. Production used 300,852 updates at batch 5,120,
or 1,540,362,240 exposures. Full training uses exactly 770,181,120 exposures:
133,712 updates at the default batch 5,760 (or 150,426 at batch 5,120). For natural count N and focused count F, allocate
the fixed update budget in proportions `N/2 : 4F : N/2`, rounding the two equal
natural phases and assigning the remainder to focused training. The earlier
literal half/four/half pass budget is superseded by this exposure requirement.
`--batch-size` must divide the full exposure budget and the five replicas.

At the default batch, five GPUs process 1,152 positions each per update,
without accumulation. The encoder uses BF16, heads FP32, and TF32 is enabled.
One AdamW optimizer covers the encoder and heads, using the same global
schedule: 2% warmup from 2e-5 to 2e-4, hold through 75%, then cosine decay to
2e-5. There are no phase resets or natural-phase LR multipliers.

Historical policy initialization restores 15 policy/board layers; the removed
specialist baseline is excluded. The new `discard_origin_scores` auxiliary
starts with zero logits and is excluded from historical final-placement
restoration despite sharing its 512×4 weight shape. Persistent gradient tapes route branch
imitation separately. Detached critic scoring runs outside tape recording
to avoid retaining inference activations. Current throughput evidence lives in
the project training review; historical timings used less frequent guidance.
The payment forecast, river encoder, and suit blocks are compiled only after
the first forward initializes real
Keras variables; compiling during shape inference can capture temporary values
and disconnect gradients. Expanded payment diagnostics stay outside the
compiled forecast. The trainer rejects disconnected parameter gradients.

Inputs travel in one buffer per dtype and are unpacked on the GPU. An int32
bitcast avoids host-only splitting without changing any bits or the TFRecord
contract. Numeric fields are concatenated before compression, dense gradient
clipping is compiled together, and risk/ranking statistics share the packed
loss reduction. Dense AdamW updates use the fused TensorFlow kernel. Sparse
embedding gradients reduce clipped per-row sums and sums of squares before
communication, preserving Keras' treatment of duplicate indices. Payment
moments share one normalized reduction for probability mass, mean, and second
moment. Fu legality and settlement use precomputed tables, scenario
losses skip unlabeled cells, and hand batches use 512-sized buckets (4,096 for compiled suit blocks). Detached
AWR scores actions in fixed chunks of 512, discarding padded results.

The historical launcher `/home/fen/projects/codex_env/codex_tools/run_critic.py` writes
`critic-current/joint-training`, historically using `reference-full/data` and
historical critic/policy initialization. It uses a frozen source snapshot; do not use it to launch the new contract.
Invoke the current trainer with rebuilt data, current-contract initialization,
and the ST3 utility settings. Historical initialization exposure to holdouts is not established; audit it before candidate selection. Previous checkpoints
and reports remain in their original directories.

`best-policy.weights.h5` selects imitation NLL only. It does not establish
critic quality or playing strength. Compare fixed-budget checkpoints using
held-out mean error, distribution calibration, confidently wrong predictions,
imitation, and paired reference evaluation. Keep the fresh reserve closed
until candidate selection.

## Batch budgets and resuming

Future runs accept `python train_joint.py --phase-batches N F N`, specifying total
natural-start, focused, and natural-finish batch budgets. For example,
`--phase-batches 3000 6000 3000` uses 12,000 optimizer updates. Zero-length
phases are allowed; the total must be positive. Full-mode overrides must still
meet the exact half-production exposure budget. Defaults retain the selected
mode's schedule. Use the original budgets when resuming to preserve its LR
and AWR schedules; these are total budgets, not additional updates.

Recovery restores model weights, optimizer slots, and global update count.
Completed phases are skipped arithmetically. Remaining phases start a shuffled
stream seeded by their starting global update, without reading/discarding prior
batches. This preserves the sampling distribution, not exact uninterrupted
example order: previously seen examples may recur. Validation is unchanged.

This source change does not alter the historical frozen snapshot under
`critic-current/joint-training/source`.

## Action agreement metrics

Discard policies use 37 tile classes; riichi policies use 74 classes, with
separate dama/declaration branches. Red fives remain distinct. Karagiri and
tsumogiri share one candidate throughout imitation, entropy, regret, critic,
AWR, serving, and top-1/top-3 metrics. There is no candidate origin flag.
`discard_origin_logits` is a separate [2, 2] auxiliary: no-declaration versus
declaration branches, each predicting recorded tedashi/tsumogiri. Its masked
cross entropy belongs to board supervision and is eligible only when the
recorded discard matches a valid current draw and the hand contains another
copy of that same tile code. Red and ordinary fives remain distinct. Other
origin labels remain stored for focus selection but receive no auxiliary loss.
Its output does not feed policy or critic values. Actual river origin
observations remain input evidence. Eligibility is computed from existing
records at training time; this loss correction requires no corpus rebuild.

The canonical candidate ranges are discard [0,37), riichi [37,111), response
without daiminkan [111,260), and kan [260,295). Response daiminkan aliases
261 + trigger tile. Tile one-hot features still have 38 entries: the extra
entry means no tile, not tsumogiri.

Records now store tile-coded discard/continuation labels, 37-entry regret,
and a separate `discard_origin` label (-1 when inapplicable). Rebuild labels,
selection, and TFRecords before training; keep frozen old corpora with their
own source. Architecture and output shapes require a new matching checkpoint
and export. This change does not resume training or deploy a model.
