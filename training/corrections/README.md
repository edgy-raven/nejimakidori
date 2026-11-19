# Targeted decision corrections

This experiment replaces broad imitation and self-play updates with measured
corrections at selected decisions. The existing candidate runner is a separate
historical recipe; use `python -m training.corrections` for this experiment.

## Data flow

1. Find eligible decisions in match replays and frozen-policy discovery games.
2. Select one action pair at a root, before observing continuation outcomes.
3. Fork the same simulator state with the same unseen-wall seed for both actions.
4. Force each proposed action once, then use the frozen continuation policy until
   the current hand settles.
5. Update from the two root outcomes. Discard ordinary and continuation decisions
   from the training batch.

Opponent concealed hands remain fixed within a pair. They vary across source
positions; this is not a posterior sampler for alternate opponent hands at one
fixed public position. Calls remain attempts subject to simultaneous claims and
priority resolution.

## Root selection

| Category | Comparison |
| --- | --- |
| Q disagreement | Captured policy discard versus max-Q discard, with at least 0.0025 normalized objective gap |
| Tenpai | Policy choice versus a choice across the keep/break-shanten boundary |
| Keiten | Same structural comparison within one shanten and at most 16 live tiles |
| Calls | Strictly shanten-improving call or kan versus declining |
| Riichi | Declaration versus dama, preferring the same discard |
| Opponent threat | Policy choice versus a different genbutsu safety level against threatening seats |

Threat selection uses public riichi declarations or at least two open melds as
proxies. It does not assert that those opponents are tenpai. When the policy
already chooses a safer tile, the alternative tests pushing; targeting is not
restricted to suspected excessive aggression.

Discovery uses equal numbers of logged and fresh games. Logged games are
sampled without replacement across rounds. The recorded action only advances
discovery and never becomes an imitation target.

Use `--matches` for completed simulator matches: their original seeds and rules
reconstruct the trajectory, with exact event validation. Use `--expert-corpus`
for expert ZIP archives. Every archived game is eligible for sampling. Expert
roots reconstruct the current hand from its observed prefix, including initial
hands, draws, calls and public indicators. The simulator executes that prefix
to recover transition state; root features must match the log replay exactly.
Unknown wall tiles are sampled from the remaining physical tiles, without
conditioning on future logged draws or outcomes. Both alternatives use the
same sampled wall. Discovery does not add expert imitation labels.

Categories receive balanced discovery-game assignments. A uniform reservoir
keeps at most one eligible root per game. Disagreement games retain up to
eight candidate positions for batched Q scoring, then choose one qualifying
root; this adds search coverage, not repeated continuations. Equal assignments do not guarantee
equal selected-root counts; report assigned, eligible and selected counts for
each category. The captured policy action remains the comparison baseline; Q scoring does
not reselect it from a second actor inference. Q disagreement currently covers
discard decisions; the other
categories do not require Q disagreement. Riichi and call baseline choices use
the same marginal branch selection as serving.

## Objective

For each branch, project the observed current-hand settlement through the fixed
continuation mapping:

`J = ST3 / 100 + P(first) - P(fourth)`

Let `a` be the frozen policy choice and `b` the proposed alternative. Define
`delta = J(b) - J(a)` and `p = pi(b) / (pi(a) + pi(b))`.

The root objective is:

`-p * delta + 0.3 * KL(pi || fixed_policy)`

The KL covers the full legal action distribution at the root. Both outcomes
count, including negative differences and ties; there is no winning-branch
filter or imitation target. Q proposes some roots but supplies no actor reward.
The observed payment likelihood and critic constraints train on both branches,
with the existing 0.1 coefficient. Model constraints retain coefficient 0.01.

There is no expert/distill imitation, adaptive beta, mirror policy, entropy
bonus, or separate downside shaping in this experiment. Expensive losses enter
through their observed settlement. This optimizes expected J, not a separate
tail-risk measure. Shared encoder updates can change behavior away from selected
roots; the anchor does not guarantee exact preservation there.

## Execution and evidence

Weights are fixed throughout each discovery/continuation round. Fresh games
and continuations use serving-style greedy choices, including marginal riichi
and call branch selection. Each collected
pair receives one update exposure; subsequent rounds use fresh positions and the
updated policy. One continuation per action keeps the budget spread across
positions. Random walls and trajectories remain noisy, so neither an individual
pair nor its training loss establishes that a move is better.

The runner uses one explicitly selected GPU, learning rate 5e-6 and a fresh
optimizer. It atomically saves every completed collection chunk as
`round-N-source-S-BBBB-pairs.npz`, so a later failure retains earlier outcomes.
It writes separate
TensorBoard history and checkpoints, and evaluates each round with fresh,
nonoverlapping four-seat wall-seed groups. There is no imitation warmup and no
separate post-match critic fit.

Create `STOP` in the output directory to stop after the current collection chunk
or update batch. The runner saves the current weights before exiting. This is a
weights checkpoint, not an optimizer-resume contract.

To continue after a completed round, initialize from its saved weights in a
new output directory and set `--start-round` to the next round. Keep the same
seed, match inputs, game budget and total `--rounds`; earlier source slices and
evaluation seeds are skipped. Optimizer, exposure counters and TensorBoard
history start fresh.

| File | Responsibility |
| --- | --- |
| [collection.py](collection.py) | Public root categories and paired outcomes |
| [trainer.py](trainer.py) | Measured paired policy loss, fixed anchor and critic fitting |
| [run.py](run.py) | Bounded collection/update rounds, checkpoints and evaluation |

```sh
CUDA_VISIBLE_DEVICES=GPU-TRAIN python -m training.corrections \
  --initial /path/to/latest.weights.h5 \
  --reference /path/to/fixed/saved_model \
  --matches /path/to/completed/matches \
  --output /path/to/new-run/candidate \
  --evaluation-command evaluation-command.json \
  --rounds 5 --games 2048 --batch-size 64 --seed 2084000
```

For expert logs, replace `--matches` with:

```sh
--expert-corpus /path/to/2024.zip /path/to/2025.zip
```

These source options are mutually exclusive. `--rounds` and `--games` set the
sample budget; supplying the full corpus does not imply a complete sweep.

`--games` counts discovery games per round, half logged and half fresh. The
match directories must contain completed evaluator manifests and original
replays. `match-sources.json` identifies every logged source; paired artifacts
retain source indices, categories and wall seeds. Budget at least
`rounds * floor(games / 2)` distinct logged games.

Use `python -m training.corrections --help` for launch paths and budget controls.
