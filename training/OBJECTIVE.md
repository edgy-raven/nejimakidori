# Candidate objective

## Objective ownership and phase coverage

Start here when auditing a missing label or an incorrect target. Model outputs
and training evidence have separate owners:

| Responsibility | Implementation |
| --- | --- |
| Settlement probabilities and hard rule support | [joint_settlement.py](../model/joint_settlement.py) |
| Streaming loss statistics and shared Q objective | [critic.py](../model/critic.py) |
| Phase-specific losses and gradient application | [trainer.py](trainer.py) |
| Belief projections and direct belief losses | [beliefs.py](../model/beliefs.py) |
| Recorded / generated / match-fit target transport | [replay](candidate/replay.py), [self play](self_play.py), [Q fit](candidate/q_fit.py) |

| Supervision | Policy training | Q-only fit |
| --- | --- | --- |
| Observed settlement NLL | Yes, critic weight 0.1 | Yes, weight 1 |
| Observed projected-ST3 MSE | No | No |
| Hard and directional critic penalties | Yes, critic weight 0.1 | Yes, weight 1 |
| Direct shanten / wait / count labels | Yes, mean of three losses, weight 1 | No |
| Raw-to-constrained belief penalty | Yes, shared constraint weight 0.01 | No separate belief penalty |
| Shared encoder and belief gradients from Q | Yes | Yes |
| Actor objectives | Yes | No |

The absence of direct belief supervision during Q-only fitting is explicit;
indirect Q gradients are not equivalent to training on hidden-hand labels.
Neither phase freezes the shared encoder merely by omitting actor loss.
Distributed training transports the complete nested replay batch; it has no
second field whitelist that can silently discard new targets.

## Observed Q loss

Policy training and Q-only fitting use the same settlement objective:

```text
Q = sum(settlement_probability * projected_settlement_ST3)
Q_loss = payment_nll + critic_constraint_loss
```

There is no auxiliary ST3 mean-loss term. Q remains the expectation of the
current-hand settlement distribution under the fixed continuation mapping.
Recorded `projected_st3` labels remain available for offline diagnostics;
training does not enumerate settlements to regress this mean. Padded rows
contribute neither loss counts nor gradients.

## Outcome evidence

Both stages fit settlements from played actions. Unplayed alternatives receive
no settlement loss, including selectively known terminal outcomes. Candidate
Q estimates remain detached inputs to the actor objectives.

## Current-hand values

The critic models observed current-hand settlements and projects each possible
outcome through the fixed match-continuation mapping. Q is the expected
projected ST3, not the projection of an averaged settlement. Later realized
match outcomes are not critic targets. There is no state-value head or loss.

## Recorded expert / distill policy imitation

```text
w(a) = stop_gradient(min(exp((Q(a) - Q(reference)) / beta_source), 20))
expert_loss = 0.5 * weighted_mean(-log P(expert), w(expert))
            + 0.5 * weighted_mean(-log P(distill policy), w(distill policy))
```

Expert and distill policy temperatures adapt separately toward 70% ESS. Each update
samples at most 64 rows per replica, retaining only valid recorded rows. Every
32 updates, calibrate on the last 128 sampled batches (at least 1,024 rows),
including the actual 20x cap. Search beta in [0.01, 128] and limit each change
to x0.8–x1.25. Start both at 16. If ties or clipping make 70% unattainable,
report the minimum attainable ESS rather than claiming convergence. Sampling
and temperature selection are detached; self-play and anchors are unchanged.

Both actions label the same recorded position. distill policy labels are cached once,
phase-conditioned and checked against the legal mask. Normalize globally
across recorded rows and replicas separately for each label source. The
reference is the fixed production policy; each label uses its own detached
Q gap. Both sources retain equal objective mass. Original
settlement and belief labels remain tied to the recorded game; an unplayed
distill policy alternative does not inherit the expert action's outcome.

The label travels with each recorded row through natural and targeted replay.
Generated rows carry a sentinel instead of a distill policy label and never enter
this imitation objective. See [annotation](HYBRID.md).

## Self-play improvement

All legal action candidates are evaluated with detached current-critic values.
Generated actions, including forced branch actions, are not imitation labels.

```text
risk(a) = 0.1 * max(downside(a) - downside(reference action), 0)
score(a) = Q(a) - Q(reference action) - risk(a)
self_play_loss = -sum(P(a) * score(a))
```

Downside is RMS below-mean ST3 shortfall on net losing-hand outcomes. Utility
uses raw ST3 units, with no division by 100. All legal
candidates retain probability mass; a sampled pair does not replace the full
legal action distribution. WIN has a single shared critic forecast, so its
value term supplies no accept-versus-decline preference.

## Mirror and magnetic anchors

Both recorded and generated rows receive these penalties exactly once:

```text
0.3 * KL(current policy || previous policy)
+ 0.3 * KL(current policy || fixed reference policy)
```

The previous policy freezes within each cohort-sized recorded window and each
mixed cohort; it refreshes from the current actor at the next boundary.
`--reference` supplies the fixed magnet, which never refreshes. These terms
implement an anchored policy objective optimized with gradient steps, not an
exact tabular closed-form MMD update. Coefficients are initial settings, not
newly strength-tuned values.

Reference weights use the current model and feature contract. Changed hard
constraints can therefore change its outputs relative to an older release's
frozen executable. The production match opponent retains its own frozen source.

Batch size changes the number of optimizer steps in a frozen-policy window.
Refreshing the mirror before every single gradient step would make its KL
gradient zero at that step's starting policy. Tune refresh cadence together
with the coefficient, batch size and self-play utility scale.

## Positive expert prior

The reference is unmodified. A separate positive-only expert term encourages
qualifying riichi and call actions:

```text
mean_over_all_decisions(coefficient * -log(sum P(eligible actions)))
coefficient = 1.0 for riichi, 0.2 for calls
```

Rows without eligible actions contribute zero. This retains a useful gradient
when eligible probability is tiny; it never supplies a dama or pass target.
First-riichi eligibility requires no threat according to the existing native
visible-threat detector:

- Favor riichi with at least six remaining winning tiles, or at least one
  additional han besides riichi, including visible dora and red dora.
- Stay neutral for a bad wait (fewer than six tiles) with dama value at
  least 5200 and a valid yaku.
- At dama value 7700 or more, favor riichi only with at least six tiles in
  row 1 (turns 1–6), or ten tiles in row 2 (turns 7–12). Otherwise stay neutral.
- At dama haneman or above, stay neutral regardless of wait or turn.

Values are non-dealer ron points without honba, ura or ippatsu. The scorer
uses the minimum value and extra han across remaining winning-tile variants;
a no-yaku variant contributes zero dama value. Wait counts are available
copies, not wait shapes. Turn means the upcoming own discard number.

For chi/pon, also require no outside threats. With a closed hand, encourage
calls reaching tenpai with a valid open yaku when post-call non-dealer ron
value is at least 3900 **or** hypothetical closed-riichi value of that same
post-call hand is below 7700. With an already open hand, encourage calls
reducing shanten. Other calls remain neutral. There is no kan prior.
Hypothetical closed value conceals chi/pon melds and treats kans as concealed;
actual seat/round winds still determine yaku. Values include visible/red dora.

These priors change only the training objective, never legality, inference
logits, Q forecasts or recorded-action weights. The selected calibration target is
riichi on 20% of hands, measured in games; it is not an enforced quota.

## Combined loss and gradients

The actor loss mixes expert loss and self-play loss by their actual source
fractions. Both anchors, positive expert CE (1.0 riichi, 0.2 calls),
0.1 settlement likelihood and
critic constraints, and 0.01 structural constraints then enter once. There is no
separate entropy bonus. Direct belief supervision adds the mean of the
shanten, improving-tile/wait and held-tile-count losses with coefficient 1,
matching the former supervised belief objective. Both recorded and generated
replay carry hidden-state labels separately from public model inputs.
Unknown labels and padded rows contribute no supervision. Belief losses and
wait calibration/recall metrics use the constrained predictions; detached
raw-to-constrained penalties remain in the 0.01 structural term, once per row.

Actor losses cannot backpropagate through Q scores or either frozen anchor.
Exact constraints train raw predictions through detached targets; see
[model constraints](../model/CONSTRAINTS.md).

Candidate forecasts use fixed-size chunks and padding for detached policy
values. They receive no additional outcome targets or backward pass.

## Calls and action priority

- Forecasts condition on attempting a call; preempted attempts remain examples.
- Accepted-call features describe a hypothetical candidate.
- Hard scoring constraints use guaranteed pre-call facts.
- A planned discard is contingent on acceptance.

## Diagnostics

Checkpoint matches are followed by two critic-only passes over their recorded
decisions. This phase minimizes the observed Q loss above at 2e-5;
it updates the shared encoder without actor objectives or policy-owned weight
updates. It uses a fresh optimizer per Q phase. These fit metrics are in-sample.
Q fitting follows the recorded pass and mixed passes 1–4 only. The final
mixed pass has no Q fit. Its checkpoint match is the final evaluation; there is no separate
confirmation match.

Track both recorded cross-entropies and Q gaps, expected self-play gain in ST3, both KLs,
settlement likelihood, gradient norms and independent match results. Non-finite
gradients fail before an optimizer update. Local gradient norm is not the
norm of the globally aggregated gradient. Lower fitting losses alone do not
establish stronger play.
