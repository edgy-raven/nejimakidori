# Match and root magnetic mirror descent

This workflow improves an initialized policy using recorded outcomes from
Mortal matches and self-play continuations. See the
[training objective](../OBJECTIVE.md) for losses and gradient ownership.

## Training sequence

1. Play 10,000 standard South games against Mortal: 2,500 wall seeds, each
   rotated through all four learner seats.
2. Fit the critic once on eligible decisions from all four seats, then run
   one MMD actor pass over the same decisions.
3. Select targeted positions and an equal number of uniformly sampled normal
   positions from those games. Generate two self-play continuations per root.
4. Fit the critic once on eligible roots and descendants, then run one MMD
   actor pass over the same decisions.
5. Play a fresh 10,000-game Mortal match. It supplies the next iteration;
   the final match is not trained afterward.

The initializer's final checkpoint starts the first match. Mortal actions are
observed experience, not policy imitation labels. Their outcomes reflect the
original policies; the self-play continuations provide additional outcomes
under the learner. Continuations stop at the current hand's settlement.

The previous-policy reference refreshes before each critic/actor phase. The
fixed reference never refreshes. Both optimizers persist across phases.

## Root selection

Each source game is assigned a category before its positions are inspected:

| Category | Comparison |
| --- | --- |
| Q disagreement | Policy discard versus max-Q discard, with at least 0.0025 J gap |
| Tenpai with threat | Keep versus break the minimum-shanten route while facing a historical visible threat |
| Keiten | The same comparison within one shanten and 16 live tiles |
| Calls | Strictly shanten-improving call or kan versus declining |
| Riichi | Declare versus dama, preferring the same discard |
| Sakigiri | Earlier safe cuts before a future genbutsu/non-genbutsu dilemma at best shanten |

Eligible positions are sampled uniformly within each category. Q disagreement
scores a reservoir of up to eight proposals. Sakigiri evidence selects starts
only; its ordering never supervises the actor.

The visible-threat rule is shared with focused corpus sampling: riichi, at
least three open melds, at least two open melds after nine opponent discards,
or a public value estimate of at least three han. The value estimate counts
exposed yakuhai, visible/red dora, and a supported flush signal. Closed kans
do not count as open melds. Hidden readiness is not required.

Sakigiri finds the first pre-own-riichi decision where a threatening opponent
has both genbutsu and non-genbutsu legal cuts tied for the best immediate
shanten. It searches backward for earlier opportunities to cut those future
non-genbutsu tiles. The early alternative and recorded cut must both avoid
immediate ron and leave exactly equal immediate shanten. Endpoints are never
training roots. Calls and kans may intervene; later hands and waits may differ.
All qualifying alternatives are stored, followed by the recorded cut. Main-loop
paired roots sample one alternative uniformly with the controlled seed;
initializer MC evaluates all stored alternatives. Witnesses select roots only
and never become imitation labels.

Each targeted root is paired with an unfiltered normal root from the same
game. Targeted alternatives share an unseen wall; normal roots sample their
initial actions and independent walls from the frozen policy. Subsequent
choices use that policy in all seats. Hidden opponent hands remain fixed
within each continuation. No root selection uses the new outcomes.

The 50/50 balance applies to root starts, not descendant counts. All eligible
decisions train once per phase; longer continuations contribute more rows.

## Run

Save an evaluation argument list as `evaluation-command.json`. It must include
`--games 10000`, `--seed`, `--mortal-source`, and the `{weights}` / `{output}`
placeholders:

```json
["python", "-m", "training.checkpoint_match",
 "--weights", "{weights}", "--output", "{output}",
 "--opponent", "/path/to/mortal-checkpoint.zip",
 "--mortal-source", "/path/to/Mortal",
 "--games", "10000", "--seed", "941",
 "--workers", "1", "--devices", "-1"]
```

```sh
python -m training.corrections \
  --initial /path/to/initializer/final.weights.h5 \
  --reference /path/to/fixed-reference/saved_model \
  --output /path/to/new-run \
  --evaluation-command evaluation-command.json \
  --rounds 5 --root-games 10000 --workers 4 --seed 941 \
  --critic-batch-size 512 --actor-batch-size 512 --root-batch-size 512
```

Choose batch sizes and collector concurrency for the available hardware.
`--root-games` limits the source games inspected for roots; match training
always consumes the full match. Each evaluation advances its base seed by
2,500. Each training invocation requires a new output directory.

Outputs include checkpoints, match replays, training records, root-selection
receipts, rollout shards, progress JSON and TensorBoard events. Create `STOP`
to stop after the current collection chunk or update batch. Completed outputs
and the latest trained weights are preserved. Match results describe the
checkpoint evaluated before learning from that match.

## Multiple workers

Set `TF_CONFIG` to the same worker list in every process, with distinct worker
indices. Select one GPU per process using `CUDA_VISIBLE_DEVICES`. Worker 0
coordinates collection, checkpoints and progress; all workers participate in
synchronized learner updates.

Batch-size arguments are global row budgets. Replica padding carries no loss
mass, and losses and gradients are normalized globally. Each worker decodes
only its assigned records. Set `--gpu-memory-mb` when an explicit allocator
limit is needed.

For separate filesystems, `--sync-command` names a JSON argument list using
`{root}` and `{stage}` placeholders. The command copies the stage inputs and
publishes `{stage}-ready.pending.json` as `{stage}-ready.json` on each worker
only after all inputs arrive. Worker output roots may differ.

| Stage | Inputs to synchronize |
| --- | --- |
| `round-N-match` | `round-N-match-data/` |
| `round-N-roots` | `round-N-*-rollouts.npz` and `round-N-rollouts.json` |
| `complete` | Ready marker only |

A marker with `stopped: true` needs no data transfer. The launcher must stop
all peers if a worker fails.

## Source layout

| Module | Responsibility |
| --- | --- |
| [run.py](run.py) | Stage sequence and checkpoints |
| [collection.py](collection.py) | Root selection and continuations |
| [matches.py](matches.py) | Recorded-match discovery |
| [rollouts.py](rollouts.py) | Rollout storage and batches |
| [trainer.py](trainer.py) | MMD updates |
| [distributed.py](distributed.py) | Replica coordination |
