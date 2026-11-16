# Replay and paired collection

[Launch a run](README.md) · [Training objective](../OBJECTIVE.md)

## Replay sources

| Source | Storage and selection |
| --- | --- |
| Ordinary recorded | Stream through the full corpus |
| Targeted recorded | Bounded reservoir, filled during the recorded pass and refreshed from the stream |
| Ordinary generated | Bounded recent replay pool |
| Targeted generated | Bounded recent replay pool |

A fixed 50% share allocates targeted positions within each source. Ordinary
sampling can also encounter targeted states. Cumulative quotas preserve source
and target shares across partial batches. All strata enter one shuffled
minibatch and share optimizer state; recorded weights normalize across recorded rows.

`--replay-positions` sets capacity per pool. Three memory-mapped pools live
under the run output; ordinary recorded data does not need a pool. Generated
chunks are shuffled so the last appended fork cannot dominate the pool.

Every replay row also retains int8 opponent shanten, improving-tile and held
tile-count labels, separate from packed public features. Native continuations
label their actual decision-time hidden hands, including forced root rows.
The existing TFRecords already contain these labels; replay pools must be
created anew with this contract.

Self-play inference transfers all five policy heads in one contiguous output
buffer, then returns NumPy views for host-side action sampling. Packed public
feature transport and replica ordering are unchanged.

## Target predicates

| Path | Eligibility | Action pair |
| --- | --- | --- |
| Calls | Optional response or own-kan with a strictly shanten-reducing call, including pass alternatives | Preferred shanten-reducing call versus pass; own kan versus decline |
| Tenpai | Actual shanten ≤0 | Keep tenpai versus break it |
| Keiten | Actual shanten ≤1 and live wall ≤16 | Preserve best shanten versus worsen it |
| Disagreement | Sample 1/64 of discard states; distinct max-Q/policy choices and sufficient Q gap | Max-Q versus policy-preferred legal action |

Shanten features store actual distance plus one. Hand-category pairs choose the
highest-policy-probability action in each group, including riichi/discard
variants. Roots without both groups are skipped. Only the disagreement path
uses max-Q versus the policy-preferred action.

Disagreement qualifies when the max-Q versus policy-action gap is at least
0.25 ST3. Identical choices never form a pair. These are selection rules, not labels
that declare either action correct.

Same-shanten or worse calls are excluded from targeted replay and root
action pairs, including own kan. Pass is targeted only if an improving call
exists. Natural replay and inference legality remain unchanged.

## Overlapping paths

Calls keep their own path when they overlap hand categories. Overlapping
hand/disagreement paths are sampled uniformly before reservoir selection.
Reservoir sampling retains at most one eligible root per natural game.
Recorded and generated target selection use observable state only.

## Matched continuations

Each selected action receives one continuation, paired with the other action
using the same fork seed.

1. Capture a root during a fresh game.
2. Keep the observed prefix and concealed hands fixed.
3. Resample unseen wall tiles while retaining revealed indicators.
4. Run each paired action with the same fork seed.
5. Force only the first action; sample later actions normally.

Calls are attempted before priority resolution. Independent root streams share
inference batches; feature rows return to trajectory order before labeling.

Each selected root produces two continuation games when both actions exist.
Fresh-position budgets and replay shares do not change. Report roots, branches
and training positions separately: the two branches from one root correlate.

## What enters replay

| Field or outcome | Treatment |
| --- | --- |
| `paired_root` | Marks the forced first action during collection |
| `root_path` | Keiten=1, tenpai=2, calls=4, disagreement=8 |
| Behavior distribution | Records the initial intervention |
| Current-hand settlement | Observed target for the critic |
| Later-hand decisions | Ordinary examples from the continuation |

Only suffix decisions enter replay. Root markers are not retained in compact
replay. Each realized return stays a separate target: matching seeds does not
make the actor consume a paired action-difference target. Generated rows train
expected-Q improvement, not imitation of forced actions.

## Collection timing

A frozen joint model collects at most one cohort ahead of the learner. Its
actor and action utilities stay fixed for the collection job. After the
first cohort of each pass, generated data is one cohort behind the learner.
Evaluation is a pass-boundary barrier.

A partial final cohort scales its fresh-position budget proportionally.
Collection estimates positions per game to shrink the last game batch and
continues until the budget is met.

## Outcome labels

Only the played action's observed settlement supervises Q. Replay retains its
existing hidden-label schema; unplayed terminal alternatives receive no loss.
Hidden labels remain separate from public model inputs.

Observed ST3 targets use raw `projected_st3` throughout replay. Recorded rows
project their stored `critic/settlement`; generated rows project their actual
`settlement` in the training controller before packing. Collection workers stay
TensorFlow-free. Padding repeats the target but contributes zero loss/count.
