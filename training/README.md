# Training

Training starts with three expert passes, then improves the policy through
matches and self-play continuations. See [setup](../SETUP.md) for dependencies
and [dataset preparation](../log_dataset/README.md) for training records.

## Initialize from expert logs

The initializer runs three Q-weighted expert imitation passes. Each pass
samples one corpus-sized budget with 50% natural and 50% focused
examples. There is no placement-based sampling stratum. Validation
records are excluded. Each pass starts a fresh AdamW optimizer at `1e-5`.

All three passes train the observed-settlement critic and seven supervised
beliefs at unit weight. Human settlement targets belong to the recorded
human action. Recorded Mortal choices supply the Q-weight comparator. Targeted
alternatives also receive one simulated root settlement each. Compact balanced
seeds accumulate while human training continues each batch. Every 16 human
batches (or at pass end), their continuations run together, then their root
labels fit the critic, followed by an MMD actor pass. Starts are 50/50 focused/natural. The simulated-decision
target matches the window's human exposures, estimated using the previous
window's decisions per continuation (initially 32). Started hands finish;
there is no debt or repayment. Each selected action is simulated once.
The six shared categories
are Q disagreement, tenpai with threat, keiten, improving calls, riichi/dama,
and future defensive sakigiri; all shanten-advancing calls include pass. Rollout
actions do not become imitation labels. Policy losses train the actor; critic
and belief losses also train the shared representation.
The [objective reference](OBJECTIVE.md) defines the losses and gradient rules.

Prepare an [evaluation command](corrections/README.md#run), then run:

```sh
python -m training.expert \
  --data /path/to/expert-data \
  --initial /path/to/initial.weights.h5 \
  --reference /path/to/fixed-reference/saved_model \
  --output /path/to/new-initializer \
  --evaluation-command evaluation-command.json \
  --batch-size 512 --rollout-batch-size 512 --seed 941
```

Prepare the dataset's [persistent root states](../log_dataset/README.md#prepare-human-root-states)
before starting initialization. The collector loads verified simulator snapshots
directly; source archives are needed only during preparation. One wall seed
is shared across competing focused actions; normal branches use independent
walls. The learner stays frozen for each collection window and plays every seat.
Independent action streams share inference batches, with at most 128 active
streams. Only root rows are retained; future-tenpai scoring stops once each
requested decision has its first witness. Each stream retains its own RNG and
outcome labels. Visible candidate yaku-affinity labels are computed once per
root and shared across its actions and walls; future-hand and settlement labels
remain specific to each continuation.
Collection is additional work beyond the corpus-sized human exposure budget.

The output includes per-rank `imitation-rollouts/pass-N/window-*-rank-N/`
directories containing `roots.json`, `statistics.json` and, when eligible
roots exist, `outcomes.npz`. Receipts record original position identities,
head-local actions, targeted/normal kind, categories and per-action wall seeds. Outcome shards contain packed
root features, global critic actions, one settlement per selected action
and directly supervised belief labels. Store run outputs on artifact storage.

The output also includes a checkpoint for each pass, `final.weights.h5`, progress
JSON and TensorBoard events. `training-complete.json` marks successful
completion. Use the final checkpoint as `--initial` for the
[match/root training command](corrections/README.md#run). Its first match is
the first post-initialization evaluation; no match runs between expert passes.

## Improve through matches

The [MMD runner](corrections/README.md) alternates a Mortal match, critic and
actor fitting, and self-play continuations from selected positions. Every
phase trains on eligible decisions from all four seats. The final match
measures the final checkpoint and is not trained afterward.

Create a `STOP` file in a run's output directory to stop at the next safe
boundary. Training commands require a new output directory and start fresh
optimizers. Evaluation commands can resume unchanged matches.

## Eligibility and monitoring

Single-legal-action decisions and discards committed by riichi do not train
policy, critic or beliefs. Voluntary riichi/dama decisions and the ordinary
discard after staying dama remain eligible. Unknown belief targets are masked
independently. An entirely excluded batch leaves optimizer state unchanged.

`exposures` counts sampled records; `actor_exposures` and `critic_exposures`
count eligible positions processed by their respective phases. Progress files
include the `imitation-rollout` critic phase: its samples increase
`critic_exposures`, while `actor_exposures` and corpus `exposures` count human
records only. Fresh optimizer state is shared by both update types within
each pass. Replica padding contributes no loss, including ranks with no roots.
JSON and TensorBoard report losses, policy agreement, gradient norms and
throughput.
[Independent evaluation](EVALUATION.md) measures playing strength separately.

Use one visible GPU per learner process. Leave `TF_DETERMINISTIC_OPS` unset
or set it to `0` for normal GPU execution; data and game seeds are controlled
separately. [Distributed execution](corrections/README.md#multiple-workers)
uses global loss normalization and synchronized updates.

| Guide | Purpose |
| --- | --- |
| [Objectives](OBJECTIVE.md) | Supervision, values and gradient boundaries |
| [Match/root workflow](corrections/README.md) | Improvement loop and command-line setup |
| [Match records](MATCH_RECORDS.md) | Recorded decisions and training targets |
| [Evaluation](EVALUATION.md) | Export, action agreement and matches |
