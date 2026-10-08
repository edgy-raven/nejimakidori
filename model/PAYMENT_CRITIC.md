# Joint policy and settlement contract

The [training job](README.md) uses `critic.JointPolicyPayment`. The policy,
payment critic and opponent-belief heads share candidate-hand, board and river
encoders. Pre/post-riichi rivers remain separate. Two board residual blocks
feed directly supervised attack/defense branches and the final policy pass.
Their predictions remain features for the final policy.

## Actions and beliefs

Visible features produce 37 discard and 74 dama/riichi choices, response and
kan policies. Legal masks apply during training and export. Drawn/held copies
of one tile class share a policy action. There is no discard-origin head.
Opening symmetries pool equivalent legal choices and fade with opponent
discards; [decision_layers.py](decision_layers.py) is authoritative.

Opponent shanten, conditional structural ukeire and hand counts have explicit
supervised heads. Their probabilities remain differentiable into the policy
and critic; one optimizer owns all shared parameters. Ukeire uses hard
structural 0/1 labels. Structural waits and legal ron are distinct: the critic's
conditional ron eligibility is multiplied by wait probability and hard public
safety masks.

The standalone belief trainer, joint trainer and exporter project masked count
marginals before both policy and critic consumption. It minimizes weighted
KL to the learned marginals, using the existing count-supervision weights
(dora, red fives and yakuhai), subject to public concealed hand sizes and shared
physical copy limits. The numerical solver targets 1e-5 count residuals,
uses at most 2,048 iterations, and raises if the final residual exceeds 1e-4.
Training reports hand-size and shared-copy errors. These are expected-count constraints, not a joint legal
hand sampler. The default constructors enable the projection; gradients from
count supervision, the actor and critic pass through it. Explicit
`project_counts=False` is reserved for matched ablations. Training plans record
the projection and exports use the same default graph.

`passed_unchanged_to_seat` retains passed-tile evidence through tsumogiri and
clears it on hand changes. `riichi_push_count` counts non-tsumogiri discards
that were unsafe against each accepted riichi at the time. Both are learned
public evidence, not hard proofs of tenpai or safety.

`last_tedashi_tile` is a `[4, 37]` one-hot input in actor-relative seat order,
including self. It identifies each player's latest non-tsumogiri discard,
including a hand-discarded riichi declaration. Red fives have separate codes;
an all-zero row means no tedashi this hand. Tsumogiri and calls preserve the
historical tile. Selection uses the full river, not the truncated river tensors.
The shared encoder consumes this categorical input for beliefs, actor and critic.
New training records and model exports must use the updated input schema;
existing frozen datasets and checkpoints do not contain this feature.

## Settlement and scoring

[critic.py](critic.py) predicts hand-ending events and scored hand values.
[immediate.py](immediate.py) owns a separate immediate-outcome network with its
own trunk, ron eligibility, han/fu amount decoder and exhaustive-draw mask head.
It shares the visible encoder, not the continuation decoder's payment factors.
[joint_settlement.py](joint_settlement.py) normalizes compatible physical-hand
outcomes, deposits and resulting scores into one joint law. Hand-end ranks
and expected payments are marginals of that law. Ties use first-dealer order;
riichi deposits, carried pots and honba are settled once. An immediate deal-in
ends the hand before future deposits. A chi/pon transition first accepts any
pending opponent declaration, so that deposit precedes ron on the call discard.

One candidate transition supplies post-action open status, fixed-meld fu,
riichi requirements and ippatsu eligibility to recorded-action training, AWR,
rollout replay and export. Opening calls cannot retain closed/riichi scoring.
A proposed declaration requires its deposit in the surviving-hand branch;
ron and triple-ron abort occur before acceptance. Exhaustive and ordinary
abortive draws accept pending declarations but cannot invent future deposits.
Future exhaustive masks require tenpai for prior riichi and every newly
accepted deposit. Event and deposit probabilities share this legality mask.
Score support includes guaranteed han from the scoring mode, bonuses and
ron/tsumo conditions; interval-valued limit tiers retain their entire range.
The 20-fu and 25-fu states also enforce their pinfu and chiitoitsu contributions.
These constraints do not sum marginal yaku predictions.
Robbed kans retain pre-resolution ippatsu; completed calls/kans cancel it.

`ron_legal[3,37]` separates legal-ron labels from payment availability. Native
scoring labels every legal discard, chi/pon follow-up discard, and concealed/
added-kan robbery candidate. Calls cancel ippatsu and account for passed-tile
furiten. Missing ura or responsibility amounts censor amount supervision, not
legal-ron supervision. Concealed-kan robbery is restricted to kokushi; any
public meld, including a concealed kan, rules out an opponent claim.
The independent immediate amount head receives known honba-free ron amounts;
it does not reuse the future winner's eventual hand value.

Visible `discard_abort[2,37]` encodes four-riichi, four-kan and four-wind abort
conditions. Wall exhaustion gates the draw head, whose sixteen masks retain
all-tenpai and nobody-tenpai separately. Current opponent shanten supplies
supervision only at that immediate boundary; it is not a future-tenpai label.
The actor's exact readiness and accepted riichi constrain those masks. Native
legal own wins and nine-terminals aborts remain resolved by the action bridge.
Three ron claim slots may coexist. The immediate decoder composes their
Bernoulli probabilities without competing away any marginal; this assumes
conditional independence of immediate claims. The future event head directly
predicts all seven nonempty claim sets per discarder, four tsumo winners,
sixteen exhaustive-tenpai masks and an ordinary abort. Double ron pays both
winners; native triple ron aborts. Only the closest ron winner receives honba
and the riichi pot. Any winning dealer repeats, including a second ron winner.
Responsibility and nagashi allocations outside this ordinary support remain
censored in settlement likelihood and reported by coverage metrics.
Call forecasts are conditional on the call taking effect; competing ron on its
triggering discard precedes that transition and remains outside this branch.

[payment_scoring.py](payment_scoring.py) owns han/fu classes and payment
constraints; [payment_bonus.py](../log_dataset/payment_bonus.py) labels observed and
counterfactual bonus scenarios. Limits collapse fu-independent scores.
Closed 20-fu pinfu tsumo, 25-fu chiitoitsu, public meld fu, red/dora bonuses,
ippatsu, chankan and responsibility payments retain their scoring rules.
These constraints do not enumerate every attainable hand.

The learned score states are independent of payer shares. Twelve ron claimant
cells and four tsumo winners supply latent hand values; all three tsumo shares
use the same value. Rule-based settlement derives rounded monetary transfers.
Bank movement follows an event-conditional sixteen-mask deposit distribution,
accepted-deposit timing and pot priority. Records store raw int32 remaining
honba-free transfers, including bank payments. Readers map bank labels to the
public pot plus zero through three other deposits, avoiding a fixed pot cap.
Supervision matches observed categorical events, shared hand scores and bank
masks directly. It multiplies separate ron-claim score likelihoods without
forming their Cartesian product. Ambiguous zero-transfer labels marginalize
nobody-tenpai, all-tenpai and abortive draws; these outcomes are not relabeled.
Exact double-win combinations remain in AWR and serving valuation.

Training summaries declare the record and scoring contracts; consumers require
an exact match. The serving payment mixture is honba-free, while projected
settlement values include the public hand context.

## Yaku and scored value

For every seat, `round_completion_yaku` and `round_completion_score` describe
one shared takame completion of the next observed tenpai shape. Shapes already
in tenpai use their current waits. Ron and tsumo have separate targets, since
their best completions, yaku and fu can differ. Select the highest total scored
points among legal, publicly unexhausted waits; break ties by native wait order,
with normal before red. Do not maximize han and fu independently or take a union
of yaku across winning tiles. Seats that never reach a supported tenpai have
unknown (-1) labels, rather than zero value. Non-winning tenpai seats are labeled.

The native scorer supplies both targets from the selected completion. Conditions
use the recorded open/closed hand, winds, known dora and ordinary riichi when
declared. Ura, ippatsu, double-riichi's extra han and special-event bonuses are
not assumed. Ron respects replay furiten. Availability excludes exhausted waits
but does not weight the maximum by tile count. This is hand potential, not
expected payment, a win probability, or a claim of optimal future play.

Yaku labels have shape `[4, 2, 23]`; score labels have shape `[4, 2]`, with mode
order ron then tsumo. Scores use canonical hand-value classes: paired han/fu below five han,
mangan/haneman/baiman/sanbaiman/kazoe tiers, and true yakuman
encoded by multiplicity (one through four). Both heads use masked auxiliary supervision at the
shared 0.01 scale. Their predictions and candidate attack-route probabilities
condition the joint critic; labels never enter inference. The realized-payment
scorer and its score NLL remain separate. The score head masks impossible
ron/tsumo fu combinations and values below the public meld-fu minimum.
Currently closed hands may open before future tenpai; declared riichi cannot.

Marginal yaku probabilities are never summed into han. The shared witness makes
supervision consistent, but separate predictions do not enforce an exact joint
law over yaku and score. Scoring remains the authoritative definition of value.
Regenerate TFRecords and train new weights for the changed label/output shapes;
there is one current contract, without an old-record compatibility path.

[yaku_constraints.py](yaku_constraints.py) removes predictions contradicted by
irreversible melds, for both four-seat completion forecasts and candidate routes.
Examples include tanyao after an honor pon, flushes after melds in two suits,
closed-only yaku after opening, and seven pairs after any fixed meld. Concealed
future tiles remain unknown; these are necessary conditions, not a joint yaku
distribution or a conversion of marginal yaku probabilities into han.

The small candidate-route head remains a structural feature: how much of the
visible shortest-route mass supports tanyao, flush or outside hands. It is not
another forecast of realized yaku or a standalone value reward. AWR learns when
these routes are worth pursuing from the scored settlement objective.

## Objectives and continuation

[critic_full.objective_terms](critic_full.py) composes the objectives:

- Legal-action imitation with active-policy entropy regularization.
- Optional detached projected-ST3 advantage-weighted imitation of the recorded
  action. Its share ramps from 10% to 60% of training progress. ESS is diagnostic.
- Joint payment likelihood, derived hand-end placement likelihood and immediate
  legal-ron binary cross-entropy, known immediate amount likelihood and
  immediate exhaustive-mask likelihood. These share the existing critic scale;
  valid candidate entries average within decisions first. Unsupported settlement
  labels are masked.
- The mean of the three opponent-belief losses, scored-winner yaku loss and
  conditional joint han/fu likelihood.
- Attack candidate yaku affinity, defense legal-ron
  prediction, and early sakigiri order supervision.

Scored-winner yaku, han/fu and [specialist losses](specialists.py) share one auxiliary
coefficient, 0.01. This reuses the existing completion-yaku scale; it is a
conservative setting, not a tuned optimum. The attack
branch predicts the visible shortest-route fractions for tanyao, flush
(honitsu/chinitsu) and outside hands (chanta/junchan), for each legal action's
post-action hand. These are copy-weighted structural route fractions, not win
probabilities. Incomplete bounded searches are unknown (-1), never negatives.
Fixed melds, public availability and all structural winning tiles constrain the
labels. Each yaku family has equal BCE weight, with no extra tanyao multiplier
or second attack imitation objective. Tanyao affinity is learned as route
support; AWR determines whether pursuing it is a good decision. Defense predicts
binary legal ron per opponent, without payment size.
Candidate prediction losses first average valid candidate/head entries within
each decision, then average labeled decisions. Unknown routes contribute no
normalization mass. Both consume explicit legal-ron labels, including known ron
with unknown payment. Defense excludes triple-ron aborts from deal-in targets;
the immediate model retains the three legal claims to form its abort branch.

Sakigiri uses native-verified exchanges of two discard orders. Both early cuts
must be legal and safe, the later tile continuously retained, every intermediate
hand no worse in shanten, and the final hand, river multiset and live legal wait
summary identical. Calls/kans and intervening accepted own riichi are excluded.
Binary endpoint ron exposure supplies the preferred early cut; triple-ron and
no-live-win cases are excluded. Each endpoint's comparisons share unit weight.
Only ordinary early discard records receive `sakigiri_pair` and
`sakigiri_weight`; future hands never enter the inputs. Defense scores receive
`0.01 * softplus(other_score - preferred_score)`. The final policy consumes the
learned specialist features and is trained by AWR, without a second pair loss.
This is conditional recorded-continuation supervision, not CFR or an unbiased
counterfactual return. The legacy scalar holding cost and never-tenpai sentinel
are not policy rewards.

Records, checkpoints, exports and rollout caches must match the current input
and label contracts. Rebuild records and train compatible weights when these
contracts change; retained releases execute their own frozen source.

The actor receives no direct learned Q or final-rank regression. ST3 uses the
joint current-hand settlement and fixed score/dealership continuation.
Terminal branches use exact final scores and ranks; outstanding terminal pots
go to the leader. Exhaustive draws repeat when the dealer is tenpai, including
all-tenpai zero-payment draws. Nobody-tenpai draws advance. Abortive draws
repeat without dealer-top termination. Payment-only labels cannot distinguish
these zero-transfer events; their likelihood retains the compatible mixture.
Current shanten is not an observed terminal-tenpai label.
Serving integrates continuation rank probabilities over
settlement branches and active policy actions to expose
`final_placement_probabilities`.

The shared auxiliary scale is in [critic_full.py](critic_full.py); policy
entropy and the payoff ramp are in [objectives.py](objectives.py).
Masked statistics reduce by global valid counts across replicas. Detached AWR
advantages cannot train the critic through the actor objective.

## Training and replay

One AdamW optimizer updates the shared encoder and heads. The default schedule
warms from 2e-5 to 2e-4, holds through 75% of progress, then decays to 2e-5.
The encoder uses BF16 and heads FP32. Recovery includes optimizer slots and
requires the same plan. Joint checkpoints also save reusable belief weights.
Future-settlement supervision uses every policy position. Plans record equal
policy and critic exposure budgets. Compact likelihoods preserve the observed
payment objective; the auxiliary derived hand-end rank loss is removed.
Counterfactual han/fu, beliefs, specialists and immediate-outcome supervision
remain active. Replay uses the same compact observed likelihood.

The CPU input pipeline compacts 130 counterfactual score targets into 27
same-score groups plus their occurrence counts. Counts preserve each cell's
original scenario mean and its valid-label weight. Unused observed-bonus and
counterfactual payment labels are not decoded or transferred to the GPU.
The raw TFRecord contract is unchanged; existing datasets need no rewrite.

Settlement supervision recomputes backward passes in chunks of 512 positions.
Detached alternative-action AWR uses chunks of 1,024 actions. Continuation
payoffs are computed once per distinct public context within each replica
batch, in chunks of 128, and gathered for its candidate actions. Keys include
all four scores, dealer, first-dealer tie priority, honba, pot and hand number. Scores, honba, bank deposits, dealer repeats and
terminal ranks are shared; action-specific event probabilities and scoring
constraints remain separate. Plans record all three chunk sizes. This reuses
exact payoffs without changing the AWR objective or settlement support.
AWR evaluates the probability law directly, without constructing serving
rank and payment diagnostics for each action.
Immediate-ron and continuation scoring share a sparse score-to-payment mapping,
avoiding a candidate-by-score-by-payment expansion in both passes.
Reported diagnostics are detached before the custom backward pass; only the
observed categorical settlement and han/fu objectives propagate through its
statistics. Payment-moment and rank diagnostics remain available from the
serving decoder but are not evaluated by each supervised training step.
Each backward chunk compiles recomputation, differentiation and gradient
accumulation together, keeping saved intermediates inside that operation.
Its differentiable forward body is inlined into this compilation; nested
compiled gradients otherwise specialize on changing forward tensor values.

Training interleaves three natural records with one focused record. Focus
categories receive the weights in [focus_sampling.py](../log_dataset/focus_sampling.py);
validation uses a fixed 1/16 decision-hash sample across the entire natural
validation partition. Dedicated validation shards avoid rescanning training
records. All sampled records, including the final partial batch, contribute;
sparse objectives combine valid-label numerators and denominators across
batches and replicas before calculating losses. Smoke mode uses two batches.
The full budget and job controls are described once in [README.md](README.md).

Optional one-hand replay uses [rollout_store.py](../log_dataset/rollout_store.py) and
[rollout_training.py](rollout_training.py). Sample stores are read-only
during training. Cached root features must match the current input names,
shapes and dtypes; regenerate incompatible caches from recorded root states
before training. Eligible roots include actor/critic/expert or point-EV
disagreements at any hand stage, tenpai, or
one-shanten with fewer than 16 live-wall tiles, across standard, chiitoitsu
and kokushi families. Complete paired screen trials supply critic labels;
confirmation trials and held-out games do not enter training. Settlement replay
detaches the encoder and trains settlement heads. A separate paired win-regret
term trains the policy and its shared encoder. Both discount stale samples.
For each nominated pair, count trials where only one action wins, including
each winner of multiple ron independently. A 95% Wilson interval estimates the
direction among these discordant trials; the nearest endpoint to equality gives
a conservative signed gap, scaled by the discordant fraction of all trials.
An interval crossing equality gives zero preference. The loss is the absolute
gap times the policy's conditional probability of the worse action within the
pair. Ties do not push the policy toward uniformity, and payment magnitude does
not enter this term. Gaps are computed once when opening the completed store.
This is conditional win-opportunity evidence under recorded hidden hands and a
frozen continuation policy, not CFR or an on-policy value estimate. Confidence
shrinkage is a training heuristic, not a multiple-comparison significance claim.
Existing settlement/AWR supervision still supplies value and draw objectives.
Admission accounts for setup and whole replay-batch time under the configured
wall-time fraction. Stored identities, offsets and partitions prevent duplicate
trials and train/holdout leakage.

`python -m model.collect_rollouts mark --store STORE.sqlite --model SAVED_MODEL
--revision REVIEW_REVISION REVIEW.json.gz ...` queues cached complete reviews.
Use the continuation actor and revision that generated those reviews; the cached
review files do not contain their revision, so it must be supplied explicitly.
The store retains model file hashes, phase, pre-decision MJAI prefix, current
feature tensors, expert choice and predicted values. Unknown opponent hands
cannot be collected with this conditional sampler. Live inference emits marks;
complete recorded hands supply recoverable roots when reviews are ingested.

`python -m model.collect_rollouts collect --store STORE.sqlite --seconds 60
--trials 32` runs a separate CPU worker. Use one collector per store. It
prioritizes fewer completed trials, then larger predicted ST3 gaps. Each trial
branches all distinct nominated actions against the same physical wall and
frozen continuation actor, stopping at the first hand settlement. Tenpai/late
one-shanten positions with unanimous nominations probe the next-best ST3 action.
Riichi roots always nominate the best projected-ST3 riichi and dama choices,
plus the legal opposite declaration for each nominated discard. With fewer
than 16 live-wall tiles, discard and response roots also compare the best
projected-ST3 tenpai choice with the noten choice having the lowest summed
immediate ron probabilities (ST3 breaks ties). This includes call versus pass;
kan replacement-draw choices are not classified as keiten from current shanten.
These are comparison candidates, not forced policy preferences. New trials use
different wall seeds; alternatives within each trial share the same wall.
The collector skips policy inference for the nominated root action and its
specified follow-up discard, and checks only new events for settlement.
Independent wall trials and their action branches advance together and batch
policy requests. `--trial-batch-size` defaults to 16 walls per position;
`--trials` still caps total trials for that position. Each wall/branch/seat has
a separate inference stream, preserving replay state and pending call plans.
Completed branches leave the batch; every nominated branch finishes before its
trial is committed. Collection admission reserves time for the whole batch,
which can overrun its estimate. Use `--trial-batch-size 1` for finer time-budget
control. The collector retains a log cursor per branch and evaluates the four
seats' settlement continuation values together. Inference uses only unresolved
policy decisions: the worker padding floor is one, so forced actions do not
cause duplicate feature rows to be evaluated merely to fill a request batch.
Root wall constraints, recorded decisions and expected-state reconstruction are
prepared once per batch and reused across wall seeds. Each wall still replays
the prefix against its own shuffled future. Logged training features and queued
root features are already persisted; features for newly simulated continuations
are built by the frozen actor's matching feature producer.
Native special-draw deltas can retain a previously deducted riichi deposit.
Collection reconstructs these transfers from boundary scores, separating
deposits and any final-game bank award. Nagashi allocations remain draw transfers
and are outside ordinary settlement supervision; coverage reports that exclusion.
Existing queued marks retain their original nominations; mark a fresh collection
store to apply the expanded coverage to previously queued reviews.
Riichi/dama roots precede declaration and deposit; call/pass roots precede the
call. Both preserve their nominated follow-up discard. Kan roots preserve the
replacement-draw boundary. No future recorded draws enter wall construction.

Collection time includes startup and complete trials; admission reserves at
least `--reserve-seconds` (default 30), increased to twice the largest observed
trial duration. A newly admitted trial can overrun its estimate; it finishes
and commits all pairs atomically. This budget is separate from training's 10%
replay budget. Interrupted trials leave no partial sample. Re-running resumes
the queue, and repeated marks do not duplicate roots. New review roots split by
game ID using the same raw-ID SHA-256 bucket as the main corpus; generated
historical roots without a source ID retain their start-state split. Training
loads the completed store at startup, decodes and rotates payment targets once,
samples positions uniformly and keeps confirmation and held-out samples out.
Unused rank labels are not constructed during replay sampling.
