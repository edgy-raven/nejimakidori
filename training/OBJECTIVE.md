# Candidate objective

## Units and targets

| Quantity | Definition | Units |
| --- | --- | --- |
| Observed return | Actual current-hand settlement projected through fixed match continuation | ST3 / 100 |
| V | State-only prediction of that observed return | ST3 / 100 |
| Q mean | Critic's expected projected current-hand settlement for the chosen action | ST3 |
| Q standard deviation | Spread of those projected settlements | ST3 |

The target uses the current hand only. Later realized match outcomes are not
labels for V. Both observations and critic forecasts use the same continuation
mapping. Because it is nonlinear, average projected outcomes rather than
projecting an averaged settlement.

## Actor weighting

With `--awr-q-std 1`, the actor's unnormalized weight is:

```text
min(20, exp((Q_mean - Q_std - 100 * V) / temperature))
```

Weights are detached and batch-normalized across recorded and generated rows.
Without `--awr-q-std`, the exponent uses the observed return minus V instead.
Predicted spread represents outcome variability, not uncertainty about the
model's parameters. This penalty does not create new rollout labels.

## Fixed policy pull

`--awr-pull` weights `KL(current policy || reference policy)`.
`--awr-reference` supplies frozen reference weights independently of
`--initial`. The reference never refreshes at cohort or pass boundaries.

## What receives gradients

| Component | Supervision |
| --- | --- |
| Actor | Detached AWR weights, entropy regularization and fixed-reference KL |
| V | Squared error against observed current-hand return |
| Chosen-action critic | Observed settlement likelihood and constraints |
| Shared encoder | Combined gradients from these objectives |

Q-based actor weights do not replace the observed targets for V or the critic.
AWR does not consume hindsight belief/completion losses or separate auxiliary
replay. Exact constraints still train raw predictions through detached targets;
see [model constraints](../model/CONSTRAINTS.md).

## Calls and action priority

- Forecasts condition on attempting a call; preempted attempts remain examples.
- Accepted-call features describe a hypothetical candidate.
- Hard scoring constraints use guaranteed pre-call facts.
- A planned discard is contingent on acceptance.

## Diagnostics

| Metric | Use |
| --- | --- |
| Value MSE and signed bias | Calibration against observed hand returns |
| Settlement likelihood | Chosen-action distribution fit |
| AWR weight concentration | Effective actor sample weighting |
| Reference KL | Drift from the fixed policy |
| `gradient/local_norm` | Local pre-clipping norm, averaged across replicas |
| Match ST3 and paired interval | Playing-strength comparison |

Non-finite gradients fail before an optimizer update. Local gradient norm is
not the norm of the globally aggregated gradient. Lower fitting losses alone
do not establish stronger play.
