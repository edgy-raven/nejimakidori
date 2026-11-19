# Match records for critic fitting and MMD

## Build

Run conversion in a separate CPU process after the checkpoint match finishes:

```sh
CUDA_VISIBLE_DEVICES=-1 python -m training.match_records \
  --matches /path/to/checkpoint-evaluation/matches \
  --output /path/to/new-critic-records --workers 16
```

The evaluator's `games.jsonl` and `evaluation.json` identify the complete
input set. The output directory must be new. Conversion writes compressed
TFRecord shards and `dataset.json` with exact game and position counts.
Keep generated records outside the source repository.

## Positions and targets

- Include every recoverable policy decision from all four seats, including
  Mortal's. Match MMD uses these as off-policy experience, not imitation labels.
- Do not apply AFK exclusions, focused sampling or training partitions.
- Share the recorded-action mapping with the expert replay reader.
- Keep censored call intentions excluded; an unrecorded losing claim is not
  evidence of a pass. Forced actions remain outside the decision contract.
- Supervise recorded current-hand payments, including accepted riichi
  deposits, with the same relative-seat contract as ordinary replay.
- Later game results are not a critic target.
- Leave unused hidden-hand, hindsight score, bonus and ron labels unknown.
  Conversion never reconstructs counterfactual scores from a winning hand.

These records use the existing numeric training-record schema. They do not
introduce a second feature encoding or a legacy format fallback.

## Consumer

`match_records.batches(output, batch_size, seed)` makes one finite shuffled
pass and retains the final partial batch. Each batch contains:

| Field | Meaning |
| --- | --- |
| `features` | Decision-time model inputs |
| `chosen` | Canonical critic action ID |
| `head`, `legal` | Active policy head and legal action mask for MMD |
| `payment_points` | Observed relative-seat payment cells |

The trainer owns padding masks, optimizer state and the number of passes.
There are no imitation targets in this consumer. Match MMD uses detached
critic action values and KL penalties.
