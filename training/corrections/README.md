# Match and root magnetic mirror descent

## Source layout

Read [run.py](run.py) for the complete training sequence. The package separates
that sequence from root selection, artifact storage and replica coordination.

| Module | Responsibility |
| --- | --- |
| [__main__.py](__main__.py) | Command-line arguments and failure reporting |
| [run.py](run.py) | Launch validation, checkpoints and match/root training stages |
| [matches.py](matches.py) | Validate match manifests and replay recorded decisions |
| [collection.py](collection.py) | Select balanced roots and simulate their continuations |
| [rollouts.py](rollouts.py) | Collect a round in chunks, write receipts and read packed shards |
| [trainer.py](trainer.py) | MMD actor objective and updates |
| [distributed.py](distributed.py) | Replica updates, stop coordination and stage publication |

The shared [critic trainer](../trainer.py), [match records](../match_records.py)
and [self-play engine](../self_play.py) live one level up. Collection returns
decision batches; rollout storage owns filenames and serialization. The runner
coordinates stages and keeps the optimizers alive across rounds.

## Training sequence

The loop is:

1. Play 10,000 standard South games against the fixed Mortal opponent, using
   2,500 wall seeds with four learner-seat rotations each.
2. Fit the critic on all recoverable decisions from **all four seats**, including
   Mortal's, then run one MMD actor pass over those positions. Observed actions
   and settlements supervise the critic; the actor uses estimated action values,
   not cross-entropy targets copying Mortal's choices.
3. Select targeted starts from the match logs and balance them with an equal
   number of uniformly sampled ordinary starts. Rerun these starts in
   **self-play**, using the policy after match training.
4. Fit the critic on every decision in those reruns, then run one MMD actor pass
   over the same decisions, including forced roots and all descendants.
5. Evaluate the resulting checkpoint in a fresh 10,000-game Mortal match. That
   match supplies the next iteration; the last match is not subsequently trained.

Imitation is reserved for
restoring prior structures in a separate restoration step; it is not part of the
MMD improvement loop. There is no outcome-winning filter, separate sakigiri loss,
or entropy bonus.

Mortal trajectories are off-policy experience: their settlements reflect Mortal's
subsequent play and the original seat-specific opponents. They are not unbiased
estimates of our policy's returns. Self-play reruns provide new outcome evidence
under our policy. The same MMD objective applies to positions from either source.

## Shared learning rule

Both actor phases minimize the same neural MMD objective:

```text
-E_pi[stop_gradient(Q)]
+ 0.3 KL(pi || previous_policy)
+ 0.3 KL(pi || fixed_production)
+ 0.01 constraint_loss
```

Subtracting the previous policy's expected Q provides a constant baseline.
Q uses `J = ST3/100 + P(first) - P(fourth)`, with scale 1 in this runner.
The previous policy refreshes at the start of each match/root learning phase,
then remains frozen through critic fitting and actor updates. The production
magnet never refreshes. Actor loss does not backpropagate through Q or update
critic-owned weights.
The critic and actor have separate AdamW optimizers, both at learning rate 5e-6.

This is a gradient-based neural approximation to the statewise
[magnetic mirror descent objective](https://arxiv.org/abs/2206.05825), not an
exact tabular solve or a four-player equilibrium guarantee. The actor evaluates
all legal actions using the learned critic; forcing a root action does not turn
that action into an imitation target. Win/pass still shares the model's one
current-hand forecast and therefore gets no value-based accept/decline gradient.

Critic supervision remains observed **current-hand settlements**, with shared
encoder gradients and the existing payment likelihood and structural constraints.
The fixed continuation mapping projects those settlements to J. Match training
covers all four seats throughout each game; it does not introduce final-game return
regression or a state-value head. Root reruns currently stop at hand settlement.
Root archives retain payment cells; unused twelve-value continuation-state
metadata is no longer serialized. Generate fresh archives for the current
contract without rewriting retained historical artifacts.
The critic fits each phase before its actor pass, so new outcomes can affect
policy improvement even when a pair has identical realized settlements.

## Root selection

Positions from all four seats in the Mortal match are eligible starts. One
category is assigned per game before scanning it:

| Category | Eligibility / exploration |
| --- | --- |
| Q disagreement | Policy discard versus max-Q discard, gap at least 0.0025 J |
| Tenpai | Opposite sides of the keep/break-shanten boundary |
| Keiten | Same structural comparison within one shanten and at most 16 live tiles |
| Calls | Strictly improving call/kan versus declining |
| Riichi | Declaration versus dama, preferring the same discard |
| Threat | Different genbutsu safety against riichi or two-open-meld seats |
| Sakigiri | Verified completion-preserving discard-order witnesses from the logged hand |

Each non-Q category uses a uniform reservoir over eligible decisions. Q
sampling keeps up to eight proposals, then selects a qualifying disagreement.
Sakigiri reuses the existing witness miner; its hindsight ron evidence selects
starts only. Its preferred/other ordering and weights never supervise the actor.
Both legal witness actions are rerun, and their new outcomes train the critic.
Sakigiri is a root-selection criterion only. The policy has no learned
`sakigiri_order` scalar or dedicated sakigiri loss.

Every selected target is accompanied by one uniform normal root from the same
source game. Normal means unfiltered, so it can also meet a targeted criterion.
The balance is **50/50 root starts**, not equal descendant counts. Both groups
receive two continuations. Targeted branches force the two proposed actions on
a shared unseen wall. Normal branches sample their first actions from the frozen
policy and use independent wall seeds. Subsequent actions sample the frozen
policy in all four seats. Opponent hands remain fixed within each fork; this is
not posterior resampling of hidden hands.

All decisions from every continuation train once per phase. Rollout files are
batched separately, and each update averages its valid rows; partial batches
therefore affect per-row step weighting. Longer trajectories provide more rows.
Mortal is used for match generation only; roots use frozen self-play.
No root selection depends on the new continuation outcomes.

## Run

The evaluator command must contain `--games 10000`, `--seed`, `--mortal-source`,
and `{weights}` / `{output}` placeholders. Save an argument list such as this
as `evaluation-command.json`:

```json
["python", "-m", "training.checkpoint_match",
 "--weights", "{weights}", "--output", "{output}",
 "--opponent", "/path/to/mortal-checkpoint.zip",
 "--mortal-source", "/path/to/Mortal",
 "--games", "10000", "--seed", "2100000000",
 "--workers", "2", "--devices", "GPU-TRAIN-A,GPU-TRAIN-B"]
```

```sh
CUDA_VISIBLE_DEVICES=GPU-LEARNER python -m training.corrections \
  --initial /path/to/initial.weights.h5 \
  --reference /path/to/fixed-production/saved_model \
  --output /path/to/new-run/candidate \
  --evaluation-command evaluation-command.json \
  --rounds 5 --root-games 10000 --batch-size 64 --workers 16 --seed 941
```

Each match advances its base seed by 2,500. `--root-games` is the number of games
sampled without replacement from the current match for root mining, not a
promise that every game yields a target. Match MMD always uses all 10,000 games.
Every invocation starts new optimizers, counters and output files; prior runs
are not resumed. Existing stopped-run files retain their historical meanings.

## Multiple learner workers

Set `TF_CONFIG` with the same worker list in every process and a distinct
worker index. Select one GPU per process with `CUDA_VISIBLE_DEVICES`. Worker 0
owns evaluation, conversion, root mining, checkpoints and progress files.
Other workers wait on CPU while those stages run. All workers create the model
and both optimizers once; optimizer state persists through every phase and
round. The mirror refreshes on every worker at each learning phase.

`--batch-size` is the global valid-row budget. Each replica receives a physical
batch of `ceil(batch_size / replicas)` rows, repeating the final input row for
padding. Valid counts exclude padding, including entirely padded replicas.
Masked statistics are normalized globally; both losses divide by replica count
for differentiation. Dense gradients are summed before per-variable clipping
at norm 1, followed by one AdamW update. The optimizer performs no second sum.
The reported norm is `gradient/global_norm`; `plan.json` records `replicas`.

When workers use separate filesystems, provide `--sync-command` pointing to a
JSON argument list with `{root}` and `{stage}` placeholders. The chief writes
`{stage}-ready.pending.json`, then runs that command. The command must copy the
stage's generated inputs and publish the pending JSON on the remote host as
`{stage}-ready.json` **after** all inputs arrive. On return, the chief renames
its own pending file to the ready file. Workers then enter a short barrier.

| Stage | Generated inputs to synchronize |
| --- | --- |
| `round-N-match` | `round-N-match-data/` |
| `round-N-roots` | `round-N-*-rollouts.npz` and `round-N-rollouts.json` |
| `complete` | Ready marker only |

A marker with `stopped: true` needs no input transfer. Checkpoint and match
provenance remain on the chief. Dataset shards and rollout filenames are
relative to each host's output root, so the two roots need not match. Workers
remain alive through the final evaluation and exit after the complete marker.
The launcher must stop all peers if a worker fails.

Artifacts include initial and phase checkpoints, completed match replays,
all-seat TFRecords, root selection receipts, and compressed rollout shards.
Rollouts use the existing lossless `PackedBatch` encoding with canonical critic
action IDs. Every shard is consumed including its final partial batch. Only
one shard is loaded at a time. Counters distinguish actor and critic exposures.
Rollout manifests list only games in completed mining chunks, including when
`STOP` interrupts a round.
Put run artifacts on the configured artifact volume, outside the repository.

Create `STOP` to stop after the current match, conversion, collection chunk or
update batch. Completed artifacts and the latest trained weights are preserved.
Match results describe the checkpoint **before** learning from that match;
post-match checkpoints require the next fresh match for strength evidence.
