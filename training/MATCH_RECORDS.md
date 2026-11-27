# Match records

Match records convert completed games into all-seat settlement and belief
supervision, together with the fixed reference probabilities used by MMD.

## Build

```sh
CUDA_VISIBLE_DEVICES=-1 python -m training.match_records \
  --matches /path/to/evaluation/matches \
  --reference /path/to/reference/saved_model \
  --output /path/to/new-match-records --workers 4
```

The input `games.jsonl` and `evaluation.json` identify the complete game set.
The output directory must be new. Conversion writes compressed TFRecord shards
and `dataset.json` with game counts, position counts and reference identities.
Keep generated records outside the source repository.

The reference export must include its matching frozen native feature producer.
Conversion replays each original event stream through that producer and stores
masked log probabilities for the active policy head. Reference inputs are
never reconstructed from the learner's feature tensors.

## Targets

Records include recoverable decisions from all four seats. Recorded actions
identify the observed settlement target; they are not MMD imitation targets.
Unrecorded losing claims are excluded rather than interpreted as passes.

Current-hand payments include accepted riichi deposits and use actor-relative
seats. Final-game outcomes are not targets. The seven belief targets are
reconstructed from each decision and its observed continuation; unavailable
targets remain masked. The trainer excludes forced decisions and riichi-locked
discards according to the [shared eligibility rule](OBJECTIVE.md#decision-eligibility).

## Batches

`match_records.batches(output, batch_size, seed)` makes one shuffled pass and
retains the final partial batch.

| Field | Meaning |
| --- | --- |
| `features` | Decision-time model inputs |
| `labels` | Seven supervised belief targets |
| `chosen` | Canonical critic action ID |
| `head`, `legal` | Active policy head and legal action mask |
| `reference_log_probability` | Frozen reference log probabilities, float32 `[150]` |
| `payment_points` | Observed actor-relative payments |

Root continuations use the same targets and reference probabilities. Their
compressed rollout shards retain every root and descendant decision. The
[runner](corrections/README.md) owns shuffling, padding and optimizer state.
