# Replay and paired collection

[Launch a run](README.md) · [Training objective](../OBJECTIVE.md)

## Replay sources

| Source | Storage and selection |
| --- | --- |
| Ordinary recorded | Stream through the full corpus |
| Targeted recorded | Bounded reservoir, filled during the recorded pass and refreshed from the stream |
| Ordinary generated | Bounded recent replay pool |
| Targeted generated | Bounded recent replay pool |

`--target-fraction` allocates targeted positions within each source. Ordinary
sampling can also encounter targeted states. Cumulative quotas preserve source
and target shares across partial batches. All strata enter one shuffled
minibatch and share AWR weight normalization and optimizer state.

`--replay-positions` sets capacity per pool. Three memory-mapped pools live
under the run output; ordinary recorded data does not need a pool. Generated
chunks are shuffled so the last appended fork cannot dominate the pool.

## Target predicates

| Path | Eligibility | Action pair |
| --- | --- | --- |
| Calls | Optional response or own-kan decision, including passes | Preferred legal call versus pass; own kan versus decline |
| Tenpai | Actual shanten ≤0 | Keep tenpai versus break it |
| Keiten | Actual shanten ≤1 and live wall ≤16 | Preserve best shanten versus worsen it |
| Disagreement | Sample 1/64 of discard states; distinct choices and sufficient Q gap | Max-Q versus legal Q closest to V |

Shanten features store actual distance plus one. Hand-category pairs choose the
highest-policy-probability action in each group, including riichi/discard
variants. Roots without both groups are skipped. Only the disagreement path
uses max-Q versus closest-to-V.

Disagreement qualifies if the Q gap is at least 0.25 ST3, or the closest Q is
still at least 8 ST3 from V. Both quantities use current-hand projected ST3.
Identical choices never form a pair. These are selection rules, not labels
that declare either action correct.

## Overlapping paths

Calls keep their own path when they overlap hand categories. Overlapping
hand/disagreement paths are sampled uniformly before reservoir selection.
Reservoir sampling retains at most one eligible root per natural game.
Recorded and generated target selection use observable state only.

## Matched continuations

`--root-copies` counts repeats **per action**. The selected recipe uses eight.

1. Capture a root during a fresh game.
2. Keep the observed prefix and concealed hands fixed.
3. Resample unseen wall tiles while retaining revealed indicators.
4. Run each paired action with the same fork seed.
5. Force only the first action; sample later actions normally.

Calls are attempted before priority resolution. Independent root streams share
inference batches; feature rows return to trajectory order before labeling.

The root count decreases approximately in proportion to the repeat count,
targeting two continuation games per natural game when enough pairs exist.
Fresh-position budgets and replay shares do not change. Report roots, branches
and training positions separately: repeated branches from one root correlate.

## What enters replay

| Field or outcome | Treatment |
| --- | --- |
| `paired_root` | Marks the forced first action during collection |
| `root_path` | Keiten=1, tenpai=2, calls=4, disagreement=8 |
| Behavior distribution | Records the initial intervention |
| Current-hand settlement | Observed target for V and critic |
| Later-hand decisions | Ordinary examples from the continuation |

Only suffix decisions enter replay. Root markers are not retained in compact
replay. Each realized return stays a separate target: matching seeds does not
make AWR consume an eight-runout average or a paired action-difference target.
V is a replay-mixture baseline, including forced roots; these samples are not
unbiased observations under the unmodified current policy.

## Collection timing

A frozen joint model collects at most one cohort ahead of the learner. Its
actor, V and action utilities stay fixed for the collection job. After the
first cohort of each pass, generated data is one cohort behind the learner.
Evaluation is a pass-boundary barrier.

A partial final cohort scales its fresh-position budget proportionally.
Collection estimates positions per game to shrink the last game batch and
continues until the budget is met.
