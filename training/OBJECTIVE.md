# Training objective

The active [match-then-root MMD loop](corrections/README.md) learns from complete
matches against Mortal, then balanced eligible/normal roots rerun in self-play.
All four seats contribute observed actions and current-hand outcomes. Root
selection changes the sampled states; forced choices and Mortal choices are
experience, not imitation labels. Every retained descendant trains the actor.

## Current-hand critic

```text
J = projected_settlement_ST3 / 100 + P(first) - P(fourth)
Q = sum(settlement_probability * J)
critic_loss = payment_nll + critic_constraint_loss
```

The critic fits observed payments for played actions. Unplayed alternatives
receive no settlement target. Each possible current-hand outcome is projected
through the fixed match-continuation mapping before taking expectations.
Terminal J uses rounded ST3/100 and the realized placement bonus; nonterminal
J uses calibrated single-Gumbel placement probabilities. Later realized match
outcomes are not targets. There is no state-value head or ST3 mean-loss term.
Padded rows contribute neither loss nor gradients.

[CriticTrainer](trainer.py) updates the shared encoder and critic from these
observed payments. Hidden-hand belief labels are not directly supervised in
this phase; shared beliefs receive indirect gradients through the critic.

## Magnetic mirror descent

```text
actor_loss = -sum((P(current) - P(previous)) * stop_gradient(Q))
           + 0.3 * KL(current || previous)
           + 0.3 * KL(current || fixed production)
           + 0.01 * structural_constraint_loss
```

The previous actor is frozen at each match/root phase boundary. The production
magnet stays fixed across the run. Both references and Q values are detached.
The previous-policy value term is a constant baseline. All legal actions enter
the objective; selected root pairs do not replace the legal distribution.
MMD uses gradient steps on this objective. There is no separate entropy bonus,
Q-weighted imitation, positive-action prior or downside-risk penalty.

Each phase fits the critic once and then trains the actor once. The shared
encoder can change in both stages. Actor updates do not fit critic weights
through Q, although their shared-feature changes can affect later forecasts.

| Supervision | Critic pass | MMD actor pass |
| --- | --- | --- |
| Observed payment likelihood and critic constraints | Yes | No |
| Detached expected legal-action Q | No | Yes |
| Previous and fixed-production KL | No | Yes |
| Actor structural constraints | No | Yes, weight 0.01 |
| Direct hidden-hand labels or imitation | No | No |

## Restoration and root eligibility

Supervised policy and belief losses and the [expert label decoder](restoration.py)
remain available for restoring prior structures. They do not run as improvement
stages. Sakigiri witnesses select eligible roots only; the policy has neither
a dedicated sakigiri scalar nor a separate sakigiri loss.

Calls must strictly reduce shanten to qualify as targeted roots; a pass is
eligible when an improving call exists. Disagreement roots compare policy
choice against max-Q with a 0.0025 objective gap. Normal roots balance targeted
roots. Native legality and normal trajectories retain all legal choices.

Forecasts condition on attempting a call, including attempts preempted by other
players. Accepted-call features describe a hypothetical candidate, while hard
scoring constraints use guaranteed pre-call facts. Planned discards are
contingent on acceptance.

## Validation and ownership

[Settlement support](../model/joint_settlement.py) owns hard outcome rules;
[critic statistics](../model/critic.py) own streaming likelihood and Q;
[model constraints](../model/CONSTRAINTS.md) own detached structural targets;
[MMD training](corrections/trainer.py) owns actor losses.

Check settlement likelihood, legal-policy entropy, both KLs, gradient norms
and independent match results. Raw ST3 reporting stays in its original units;
J includes placement bonuses and cannot be converted back by multiplying by
100. Non-finite gradients fail before updates. Lower training losses alone do
not establish stronger play.
