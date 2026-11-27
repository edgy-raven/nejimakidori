# Training objective

The model learns from observed current-hand outcomes and recorded decisions.
Its actor and settlement critic share a representation with seven supervised
belief predictions. [Training](README.md) describes the commands;
[model constraints](../model/CONSTRAINTS.md) describe legal support.

## Decision eligibility

Padding, single-legal-action rows and discards committed by riichi contribute
zero loss. Voluntary riichi/dama decisions and the ordinary discard after
staying dama are eligible. Unknown belief targets are masked per coordinate.
An entirely excluded global batch does not update optimizer state.

A policy target must also lie within the actor's effective support. A legal
teacher discard excluded by the final-wall safe-cut rule receives no policy
cross-entropy or normalization mass, while its known predictive targets
remain eligible. `policy_target_coverage` reports this fraction. Q-weight
calibration uses all eligible rows.

## Expert initialization

The initializer has exactly three corpus-sized passes of Q-weighted
cross-entropy against the recorded human action. Each pass uses a fresh
optimizer at learning rate `1e-5`.

Every pass also fits observed settlement likelihood, critic constraints and
all seven supervised beliefs, each at unit weight. The settlement target
belongs to the human action actually played. Unplayed Mortal actions receive
no copied human outcome labels. Human CE has no MMD penalty.

Settlement sufficient statistics and their gradients use 128-row chunks,
with one loop iteration in flight to bound the settlement workspace.
Padding has zero valid mass; chunk totals are reduced over the original
eligible rows. The backward pass recomputes each chunk instead of retaining
all settlement distributions.

Before each human exposure batch, select focused roots paired one-for-one
with unfiltered roots from the natural-source half of the batch. Collect
and fit their Monte Carlo outcomes together at each 16-batch window boundary
and at pass end. The six categories are Q disagreement, tenpai with threat,
keiten, strictly shanten-advancing calls, riichi/dama and future defensive
sakigiri, using the [shared category rules](corrections/README.md#root-selection).
Every advancing
legal call-plus-discard is included with pass, regardless of the policy's
preferred action. Sakigiri includes every qualifying earlier alternative and
the recorded cut. Other categories contribute their existing action pair;
overlapping actions and categories at a position are deduplicated.

Each 16-batch collection window targets one actual simulated decision per
human exposure. Every requested decision across all seats counts, including
forced roots and descendants not retained for training. Padding and scoring
recorded human candidates do not count.

Each human batch selects balanced focused/normal seed pairs using an estimate
of decisions per continuation: initially 32, then the previous window's actual
global decisions divided by its continuations. Pair selection stops when its
estimated cost reaches the batch's human count. Eligibility, selected actions
and category rules are unchanged by lazy scoring: visit all
eligible positions in a seeded random order, score them in small chunks, and
stop once enough qualifying pairs have been selected. Restricting a uniform
random order to qualifying positions preserves uniform targeted selection;
there is no fixed search cap. Score normal partners together after selection.
Compact metadata, selected actions, wall seeds and root affinities are retained;
full human batches are
not buffered. Human fitting continues each batch. Every 16 batches, and at a
short final pass window, all accumulated continuations enter one collector
with up to 128 active streams per learner. Each action runs once. Started
hands finish, and all their root labels are fitted after collection. Actual
counts can differ from the estimate; there is no debt or repayment. Root
starts remain 50/50, with no placement-based sampling or passes.

Seed choices use the policy at selection time; continuations use the current
policy frozen at the end of the collection window. This introduces up to
16 human updates between selecting a seed and running it.

Focused alternatives share one wall seed; normal roots draw two policy actions
with replacement and independent walls. The learner is frozen during collection and used in
all seats. Original opponent hands remain fixed; unseen wall tiles, including
unrevealed indicators, are reshuffled. Human roots load directly from saved
simulator states whose features were verified during dataset preparation.
Held-out games are rejected. A batch with no eligible natural-source partner
does not collect an unbalanced focused-only set.

Each continuation supplies its own root settlement and eligible belief labels.
Root likelihoods are averaged over valid samples. Mean payments are never treated as a settlement.
Only roots are retained for initializer updates; descendants are used to
produce outcomes and labels. These critic/belief updates share the stage's
optimizer with human updates. They do not train policy CE or turn forced
actions into imitation labels. Human expert updates compute Q weights from
the learner at that update, including prior collection-window updates. All three initializer passes use this
collection/fitting sequence; their human exposure budgets remain unchanged.

Each MC window then trains the actor with the same MMD objective as the
main loop: detached expected Q, previous-policy KL 0.3, fixed-reference KL
0.3, actor structural constraints 0.01, and all seven direct belief losses at
unit weight. Snapshot the learner actor immediately before fitting that
window's critic targets; keep this mirror fixed through both fits. The magnet
uses the immutable reference model and its own frozen replay features at the
root before any forced action. Reference inference batches independent roots
together; alternatives at the same root share its reference distribution.
MMD has its own optimizer at 1e-5, fresh each initializer pass and retained
across windows within that pass. All collected root rows receive one critic
pass followed by one actor pass. Forced actions are not imitation labels.

For the expert passes, both action values come from the current learner:

```text
gap = stop_gradient(Q(human) - Q(Mortal))
raw_weight = min(20, exp(gap / beta))
weight = raw_weight / mean_valid(raw_weight)
loss = mean_valid(weight * human_cross_entropy)
     + payment_nll + critic_constraint_loss + sum(belief_losses)
```

Each global batch calibrates beta in `[0.0001, 1.28]` to target 75% effective
sample size. The solver chooses the highest-beta crossing; when the target
is unattainable, it reports the closest grid value. Weights are detached and
apply only to policy cross-entropy. Recorded Mortal choices come from the
prepared corpus and are mapped into the critic's candidate-action space.

Each pass draws 50% natural and 50% focused records, with
replacement. Its total exposure budget equals the number of training records;
validation remains natural and disjoint. Focused sampling emphasizes calls,
yaku routes, visible threats and late decisions. No placement-based sampling
stratum or pass remains. Sampling priorities are metadata, not model inputs; their
rules are defined in [focus_sampling.py](../log_dataset/focus_sampling.py).
Each learner shard must contain positive focused mass. Initializer MC and
main-loop root collection both pair focused and unfiltered normal starts
50/50. This is a root-start balance: all generated eligible training rows
remain supervised, so differing action counts or trajectory lengths need not
produce equal loss-row counts.

## Current-hand critic

```text
J = projected_ST3 / 100 + P(first) - P(fourth)
Q(action) = sum(settlement_probability * J)
critic_loss = payment_nll + critic_constraint_loss + sum(belief_losses)
```

The critic fits observed payments for the action played, including forced
alternatives with their own simulated outcomes. Alternatives without an
observed or simulated outcome receive no settlement target. Each possible
settlement passes through the fixed match-continuation mapping before taking
its expectation. Terminal
values use realized placement and rounded ST3; nonterminal values use
calibrated placement probabilities. Final-match returns are not targets.

The likelihood marginalizes compatible total han/fu outcomes. Public scoring
evidence conditions the score network; future bonuses are learned within its
single total-score distribution, with no separate added-han prediction.
Belief and critic losses train their own readouts and shared representations.
A call candidate includes its contingent discard; settlement forecasts account
for the possibility that another player preempts the attempted call.

## Magnetic mirror descent

Each match/root phase fits the critic once, then trains the actor once:

```text
actor_loss = -sum((P(current) - P(previous)) * stop_gradient(Q))
           + 0.3 * KL(current || previous)
           + 0.3 * KL(current || fixed_reference)
           + 0.01 * structural_constraint_loss
           + sum(belief_losses)
```

The previous policy is frozen at the start of each phase, before critic
fitting. The fixed reference remains unchanged throughout the run. Both
references and Q values are detached. All legal actions enter the objective;
forcing an action at a rollout root does not make it an imitation target.
The previous-policy value term is a constant baseline.

Reference probabilities are evaluated with the reference export's own frozen
feature producer, then stored with each training decision. The actor and critic
use separate AdamW optimizers at `1e-5`. Optimizer state persists across the
match/root phases. Gradients are summed globally, clipped per variable at norm
1, and checked for finite values before an update.

## Gradient ownership

| Objective | Shared representation | Own prediction layers | Actor layers |
| --- | --- | --- | --- |
| Direct belief supervision | Yes | Yes | No |
| Settlement likelihood and critic constraints | Yes | Yes | No |
| Policy cross-entropy, Q improvement and KL | No | No | Yes |

Every consumed belief prediction is detached, including predictions consumed
by another belief. Actor inputs also detach the shared representation. Critic
losses retain their representation gradients. Consequently, shared features
can change during an MMD actor pass through direct belief supervision, while
Q itself remains outside the actor's gradient path.

Monitor likelihood, belief quality, policy entropy, KL penalties and gradient
norms alongside [independent match results](EVALUATION.md). J includes placement
bonuses, so multiplying it by 100 does not recover ST3. Lower training loss
alone does not establish stronger play.
