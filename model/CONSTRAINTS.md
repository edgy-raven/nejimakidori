# Constraints and gradients

[Architecture](ARCHITECTURE.md) · [Candidate objective](../training/OBJECTIVE.md)

## Shared representation

Forward masks enforce legality. Detached constrained targets also train raw
predictions so the heads and shared encoder learn the constraints.

| Output | Constraint loss |
| --- | --- |
| Categorical heads | Forward KL to constrained targets |
| Yaku and route marginals | Bernoulli KL |
| Critic settlements | Payment, score and deposit constraints |

Shared-head constraint weight is 0.01. Critic constraints join the critic
objective at weight 0.1. Unknown feasible yaku do not become positive labels.
Padded rows do not contribute.

## Policy exceptions

| Case | Treatment |
| --- | --- |
| Uncomputed action slots | Constant padding and hard legality mask |
| No legal win opportunity | Win head masked; not a decline label |
| Legal ron or tsumo | Learn accept versus decline |
| Heuristic policy priors | Separate from constraint losses |

Padding must not penalize the arbitrary common offset of learned action scores.
Non-winning positions must not teach the model to refuse wins. Neither case
receives a raw-to-masked policy penalty.

## Calls

Call forecasts condition on attempting the action, including preemption by
other claims. Hypothetical accepted-call features describe the candidate;
constraints use facts guaranteed before acceptance. The follow-up discard
remains contingent. Passing also waits for priority resolution, including
pending riichi declarations and last-wall boundaries.

## Data compatibility

Feature layouts, labels and score vocabularies must agree across datasets,
weights and exports. Consumers validate the dataset summary against the current
schema. Regenerate records after a contract change; editing their summary does
not convert them. Frozen exports retain their matching source.
