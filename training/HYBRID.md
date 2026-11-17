# Q-weighted expert and distill policy imitation

The selected candidate uses the [six-pass recipe](candidate/README.md): one
recorded pass, then five mixed passes, checkpoint matches and Q fitting.
Its recorded actor objective is

```
w(a) = stop_gradient(min(exp((Q(a) - Q(reference)) / beta(source)), 20))
loss = 0.5 * weighted_mean(CE(expert), w(expert))
     + 0.5 * weighted_mean(CE(distill), w(distill))
```

Each source starts at beta 16 and adapts independently toward 70% rolling
ESS. See the [candidate recipe](candidate/README.md) for its update bounds.
Both terms use the same positions. Normalize each label source separately
across recorded rows and replicas, preserving equal objective shares. Q is
projected current-hand ST3 and the reference action comes from the fixed
production policy. Each label uses its own Q gap; weights do not backpropagate
into Q. The full recipe retains self-play improvement, both KL anchors,
positive priors and shared belief/critic objectives.
AdamW starts at 2e-5, decreasing linearly to 5e-6 as the scheduled self-play
share rises from 0% to 80%, with weight decay 1e-4 and clip norm 1. The actor and
its shared encoder train. The full recipe also supervises the critic and
beliefs. Existing outcome labels remain associated
with the original played actions, and are not assigned to distill policy alternatives.

The initializer must match the current model architecture. For production
weights predating tile attention, transfer every existing array unchanged and
initialize the new attention correction to zero before strict loading. Verify
initial-output parity and gradients through both attention blocks; do not use
partial weight loading or carry probe updates into the training initializer.

## One-time greedy annotation

`log_dataset.distill_labels` reads the original game archives identified by the
expert corpus's `games.jsonl`. It hides opponents' concealed hands and draws
before passing events to the frozen upstream distill policy adapter. The checkpoint,
upstream Python and native extension hashes are recorded in `plan.json`.

The distill policy's combined action space is conditioned on the student's decision phase.
Discard rows exclude riichi/kan/win; riichi rows compare ordinary discards with
riichi followed by its greedy discard; response rows compare pass, calls with
their greedy follow-up discard, and open kan. Kan rows compare kan with
continuing, and win rows compare winning with continuing. The upstream win
legality guard is retained. No Boltzmann sampling or stochastic latent is used.
These are greedy phase-conditioned labels, not a mixture sampled online.
Hypothetical call and riichi continuations are discarded before returning to
the recorded trajectory. Delayed dora information visible at a decision is
applied before its triggering draw.

Each original record gets one int16 action in an aligned `.npy` file. Its
manifest pins the source shard and label file by SHA256. The original corpus
is not rewritten. Every label is checked against the student's legal mask
before the manifest is published. Completed shards can be reused on restart;
worker shard offsets must be disjoint.

Training verifies each shard and label file once, then streams records and
aligned labels through `tf.data`. Partition checks and aligned label gathers
run in batches before records are individually shuffled; record-count
mismatches fail before a full pass can finish. Records do not
round-trip through Python individually. Both imitation terms share one
log-softmax. The candidate runner owns objective weights and optimization;
the label reader does not provide a separate training recipe.

```sh
python -m log_dataset.distill_labels \
  --data /path/to/expert/data --output /path/to/distill-labels \
  --teacher /path/to/frozen-distill --device cuda:0 --workers 32
```

`--teacher` is an evaluator's frozen distill policy `old/` bundle containing
`checkpoint` and `source/`. For multiple GPUs, set each worker process's
`CUDA_VISIBLE_DEVICES` to one training GPU UUID and use distinct
`--shard-offset` values with the same `--shard-stride`.
`--limit-shards 1` validates a pilot. Progress uses TensorBoard under the label
output, or `--tensorboard`. `--cuda-graphs` caches the float32 brain execution
for each actual batch shape; precision, masks and batch composition stay the
same. Multiple processes may share a GPU using distinct shard offsets.
Private events are serialized once per seat and hand; only the observed
prefix is passed to distill policy. ZIP indexes are opened before dispatching game
threads to avoid redundant concurrent index builds.
