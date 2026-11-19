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
objective at weight 1.0. Unknown feasible yaku do not become positive labels.
Padded rows do not contribute.

Opponent held-count marginals respect known meld copies, concealed hand sizes
and shared unseen-copy limits. Declared riichi or four fixed melds (hadaka
tanki) fix the opponent's shanten distribution to tenpai. Both require tenpai
in exhaustive-draw settlements; four melds do not imply a riichi deposit.
Both projections teach their raw heads through detached-target KL penalties.
Direct belief supervision uses the projected outputs.

Structural wait probability is tenpai probability times the conditional wait
probability. There is one settlement head, with no separate immediate-ron
branch. Last-discard and abort boundaries restrict event support, preserving
ron priority and declaration timing. Future wins are not capped by present
waits because hands can change before winning.

Directional penalties compare half/double belief-odds probes with all other
inputs fixed. Higher opponent tenpai should increase that opponent's winning
probability and readiness conditional on exhaustive draw. Higher conditional
wait probability on the candidate tile should increase that opponent's ron
against the actor. Triple-ron abort is excluded from winning marginals.
Violating probe pairs train toward a detached equal-probability target;
probe inputs are detached so the critic, not the belief labels, absorbs this
penalty. The terms enter critic constraints once. They encourage directional
consistency but do not guarantee global monotonicity. Known readiness and
no-tile actions naturally have no applicable probe change.

## Policy exceptions

| Case | Treatment |
| --- | --- |
| Uncomputed action slots | Constant padding and hard legality mask |
| No legal win opportunity | Win head masked; not a decline label |
| Legal ron or tsumo | Restoration learns accept/decline; MMD shares one forecast |
| Heuristic policy priors | Restoration only; excluded from MMD |

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
