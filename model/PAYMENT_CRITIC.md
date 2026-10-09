# Architecture and training objectives

Nejimakidori learns a legal-action policy alongside predictions that explain
the hand: opponents' concealed tiles, attack routes, immediate deal-in risk,
and eventual settlement. The settlement critic supplies the value estimates
used by advantage-weighted regression (AWR).

This page defines the current model. For commands and recovery, see
[training](README.md); for record generation, see [datasets](../log_dataset/README.md).

## Shared representation and decisions

[JointPolicyPayment](critic.py) shares candidate-hand, board and river encoders
between the policy, beliefs and critic. Rivers before and after riichi are
encoded separately. Attack and defense branches receive their own labels,
and their predictions become features for the final policy.

| Output | Meaning |
| --- | --- |
| Discard policy | 37 tile classes, distinguishing red fives. |
| Riichi policy | 74 choices: dama or riichi, each paired with a discard. |
| Response and kan policies | Legal pass, call and kan choices. |
| Opponent beliefs | Shanten, shanten-conditional improving tiles and tile counts. |
| Completion forecasts | Yaku and scored value for each seat, separately for ron and tsumo. |
| Immediate outcomes | Claims and draws ending the hand at this action. |
| Future settlement | Hand-ending event, hand values and riichi deposits. |

Drawn and held copies of the same tile class share a policy action; the action
bridge preserves red identity and forced-discard legality. Legal masks apply
in training and export. Opening symmetries pool equivalent legal actions and
fade with opponents' discards; [decision_layers.py](decision_layers.py)
defines that pooling.

All inputs are visible to the acting player. They include own hand, candidate
actions, rivers, melds, dora, score context and public safety evidence.
`last_tedashi_tile[4,37]` records each seat's latest hand-discarded tile in
actor-relative order, including self; zero means no tedashi this hand.
Tsumogiri and calls preserve it. Passed-tile evidence persists through an
unchanged hand and clears on hand changes. A separate count records unsafe
tedashi against each accepted riichi. These are learned clues, not proofs
that an opponent is tenpai or that a tile is safe.

## Opponent beliefs

The belief heads learn concealed tile counts, shanten and structural ukeire
(tiles that improve the hand). Structural waits and legal ron are distinct:
a complete shape may lack yaku or be furiten. Immediate ron combines predicted
wait probability, conditional ron eligibility and public safety masks.

Before the policy or critic consumes count marginals, a differentiable
projection enforces expected concealed hand sizes and shared physical copy
limits. It minimizes weighted KL divergence from the learned marginals, using
the count loss's dora, red-five and yakuhai weights. The solver targets a
1e-5 residual, allows at most 2,048 iterations and fails above 1e-4.

The same projection runs in belief training, joint training and export.
Gradients from count supervision, policy and critic pass through it.
`project_counts=False` is available for matched ablations. These constraints
apply to expected counts; they do not define a sampler of complete legal hands.

## Immediate outcomes and future settlement

An action can end the hand immediately or leave it to continue. A separate
[immediate network](immediate.py) predicts immediate ron eligibility, ron value
and exhaustive-draw tenpai masks. It shares the visible encoder with the
future critic but has its own trunk and decoders.

[The settlement decoder](joint_settlement.py) combines these forecasts with
scoring rules into compatible events and score transfers. Expected payments,
hand-end ranks and continuation values are derived from this distribution.

| Event | Representation and settlement |
| --- | --- |
| Ron | Three possible claimants per discarder. Double ron pays both winners. |
| Triple ron | An abort under the implemented rules; all three legal claims remain represented. |
| Tsumo | One winner's hand value determines all three payer shares. |
| Exhaustive draw | Sixteen tenpai masks distinguish all readiness combinations. |
| Abortive draw | Separate from an exhaustive draw, including zero transfers. |
| Riichi deposits | An event-conditional sixteen-mask distribution, settled with the existing pot. |

The immediate decoder combines independent Bernoulli ron claims. The future
event head directly predicts seven nonempty claim sets per discarder, four
tsumo winners, sixteen exhaustive masks and an ordinary abort. Immediate
claim independence is a modeling assumption.

Han/fu classes encode hand value below the limit tiers; mangan, haneman,
baiman, sanbaiman, kazoe and true yakuman multiplicities encode higher values.
Twelve ron claimant cells and four tsumo winner cells supply the values.
[payment_scoring.py](payment_scoring.py) converts them to rounded transfers
with dealer, honba and bank rules. Marginal yaku probabilities are never
summed to obtain han.

The decoder enforces these timing and legality rules:

- Immediate ron ends the hand before future deposits. A proposed riichi is
  deposited only if it survives ron or triple-ron abort. Pending declarations
  are accepted before ordinary exhaustive or abortive draws.
- Chi/pon forecasts use the post-call hand, including its opening status,
  fixed-meld fu and cancelled ippatsu. Competing ron on the triggering discard
  happens before the call and is outside that conditional forecast.
- Robbed kans retain pre-resolution ippatsu; completed calls and kans cancel
  it. Concealed-kan robbery is restricted to kokushi.
- Score support respects guaranteed han, public meld fu, bonuses and scoring
  mode, including pinfu's 20-fu and chiitoitsu's 25-fu cases. These constraints
  do not enumerate every attainable hand.
- Only the closest ron winner receives honba and the riichi pot. Any winning
  dealer repeats, including a second ron winner. Ties use first-dealer order.
- Accepted riichi requires tenpai in exhaustive masks. All-tenpai and
  nobody-tenpai remain separate despite both having zero noten transfers.
  Abortive draws repeat; exhaustive draws repeat when the dealer is tenpai.

Legal own wins and nine-terminals aborts are handled by the native action
bridge. Responsibility and nagashi payments outside ordinary settlement
support are excluded from its likelihood and counted in coverage metrics.

### Settlement labels

`ron_legal[3,37]` labels candidate discards, chi/pon follow-up discards and kan
robbery using native legality/scoring. Calls account for cancelled ippatsu and
passed-tile furiten. Unknown ura or responsibility amounts can mask the amount
label while leaving legal-ron supervision valid. The immediate amount head
learns the immediate ron amount, not the future winner's eventual value.

Visible abort features identify four-riichi, four-kan and four-wind cases.
Wall exhaustion gates the immediate draw head. At that boundary only, actual
shanten supplies tenpai supervision; it is not a label for future readiness.

Records retain honba-free transfers, including bank movement. Observed event,
score and deposit labels train a categorical likelihood. Separate ron-claim
score likelihoods multiply without constructing a Cartesian product of score
classes. If zero transfers cannot distinguish all-tenpai, nobody-tenpai and
abort, the likelihood sums compatible outcomes. It does not invent a noten
label. Serving and AWR retain exact double-win combinations for valuation.

The serving payment mixture is honba-free. Projected settlement values add
the actual honba, scores, deposits and pot from the public context.

## Completion yaku and value

Every seat has paired yaku and score targets for its next observed tenpai
shape, or its current shape when already tenpai. Non-winning tenpai hands
also receive labels. Ron and tsumo each select their own **takame**: the legal,
publicly unexhausted completion with the highest total scored points. Ties
follow native wait order, normal five before red.

Both targets come from that one completion. Han and fu are not maximized
independently, and yaku from different waits are not combined. Scoring uses
the recorded melds, winds, known dora and ordinary declared riichi. It does not
assume ura, ippatsu, double-riichi's extra han or special-event bonuses. Ron
respects replay furiten. Seats that never reach supported tenpai are unknown
(`-1`), not zero-value examples.

`round_completion_yaku` has shape `[4,2,23]`; `round_completion_score` has shape
`[4,2]`, with ron before tsumo. These describe hand potential, not win
probability or expected payment. Their predictions condition the critic.
Separate heads do not enforce a joint yaku/score distribution, even though the
labels share a completion.

[yaku_constraints.py](yaku_constraints.py) masks routes contradicted by fixed
melds, such as tanyao after an honor pon, closed-only yaku after opening or
seven pairs after any meld. Future concealed tiles remain uncertain.

## Attack, defense and sakigiri

The attack head learns copy-weighted fractions of shortest structural routes
supporting tanyao, flush hands (honitsu/chinitsu), and outside hands
(chanta/junchan). Each legal action supplies its post-action hand. Melds,
public availability and structural winning tiles constrain these labels;
incomplete bounded searches are unknown. The three families have equal BCE
weight. There is no additional tanyao reward or attack imitation objective.

The defense head learns binary legal ron exposure per opponent, without
payment magnitude. Triple-ron aborts are excluded from deal-in labels, while
the immediate network retains all three legal claims for its abort branch.

Sakigiri labels compare exchanging two early discards. Both early choices
must be legal and safe; the later tile must remain held; intervening hands
must not worsen in shanten. The final hand, river multiset and live legal wait
summary must match. Calls, kans and intervening accepted own riichi are
excluded. If one order exposes a legal ron at the later endpoint, the other
is preferred. Triple-ron and no-live-win endpoints are excluded.

Each endpoint's comparisons share unit weight. The defense score learns
`0.01 * softplus(other_score - preferred_score)`. The final policy consumes
those specialist features but has no second sakigiri pair loss. This is
supervision under a recorded continuation, not CFR or an unbiased return.

## Joint objective and AWR

One AdamW optimizer trains the shared encoder and heads. The encoder uses
BF16 and prediction heads FP32. The learning rate warms from 2e-5 to 2e-4,
holds until 75% progress, then decays to 2e-5.

[critic_full.objective_terms](critic_full.py) combines:

| Term | Role and scale |
| --- | --- |
| Policy | Recorded-action cross-entropy, blended with AWR. |
| Entropy | Active-policy entropy bonus, coefficient 0.01. |
| Critic | Settlement and immediate-outcome supervision, coefficient 0.1. |
| Beliefs | Mean of count, shanten and conditional ukeire losses. |
| Completion | Yaku BCE and paired score NLL, coefficient 0.01. |
| Specialists | Attack, defense and sakigiri terms, coefficient 0.01. |
| Scoring | Counterfactual score NLL, coefficient 0.01. |

There is no auxiliary hand-end rank loss or direct learned final-rank
regression in the current training objective. Hand-end ranks remain derived
serving diagnostics. Unknown targets contribute neither loss nor denominator.
Candidate losses average valid entries within a decision before averaging
labeled decisions; distributed statistics use global valid counts.

For AWR, let `Q(a)` be projected ST3 rank points and `pi(a)` the current legal
policy. The recorded action receives advantage
`(Q(recorded) - sum_a pi(a) Q(a)) / temperature`. Its regression weight is
`exp(min(advantage, log(20)))`, normalized over eligible records. The weight
is detached: the actor loss cannot train the critic by changing its advantage.
The AWR blend ramps from zero at 10% progress to `--awr-mix` at 60%.

ST3 uses [the implemented Saint 3 Jade South utility](outcomes.py). The critic
integrates current-hand settlement against a fixed continuation model using
scores, remaining hands and dealerships. Terminal branches use exact final
scores and ranks, with outstanding pots awarded to the leader. Final-placement
probabilities integrate those continuation probabilities over settlement
branches and active policy actions.

Settlement supervision covers every policy position. To bound memory, the
trainer recomputes settlement backward passes in 512-position chunks and
scores alternative AWR actions in 1,024-action chunks. Public-context payoffs
are shared across actions and computed in 128-context chunks. CPU readers
compact equivalent score labels before GPU transfer. These are computation
choices, not changes to the objective.

## Rollout supervision

[The collector](collect_rollouts.py) reconstructs complete recorded roots,
retains their hidden hands and shuffles future walls consistent with the
observed prefix. It does not draw hidden hands from belief marginals.
Different trials use independent wall seeds; all nominated alternatives
within one trial share a wall and a frozen continuation policy. Each branch
stops at the first hand settlement.

Nominations cover policy/critic/expert or point-value disagreements, tenpai,
and late one-shanten positions. Riichi roots explicitly compare riichi and
dama. With fewer than 16 live-wall tiles, discard and response roots compare
the best projected-ST3 tenpai choice with the noten choice having the lowest
summed immediate ron probability. Calls can be compared with pass. Kan
replacement draws are not classified as keiten from current shanten.

Complete screen trials provide settlement labels to the critic, with encoder
gradients stopped. A separate paired win-opportunity term trains the policy
and encoder. For each action pair, count trials where exactly one action wins;
multiple-ron winners are counted separately. A 95% Wilson interval on these
discordant trials gives a conservative preference: take the endpoint nearest
equality, then scale by the discordant fraction of all trials. An interval
crossing equality gives no preference. The loss is the absolute gap times
the conditional probability of the worse action within the pair.

Payment magnitude does not enter that paired term; settlement/AWR still
supply value and draw objectives. Confidence shrinkage is a training
heuristic, not a multiple-comparison significance claim. Both replay terms
discount stale samples and share the configured replay time budget.

Only complete training-partition screen trials are admitted. Confirmation
trials and held-out games are excluded. Stores retain root identities,
partitions, model hashes and cached features; interrupted trials do not leave
partial pairs. One collector owns a store. The trainer opens completed data
read-only. Cached inputs must match the current feature schema.
