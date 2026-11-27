# Export and evaluation

## Export weights

```sh
python -m model.export \
  --weights /path/to/run/final.weights.h5 \
  --data /path/to/dataset --output /path/to/export
```

To export without a corpus, replace `--data` with `--self-play-seed 941`.
Export validates finite outputs, normalized distributions and numerical parity
before and after saving.

| Output | Purpose |
| --- | --- |
| `saved_model/` | TensorFlow prediction signatures |
| `source/` | Matching feature and policy source |
| `export-verification.json` | Weight identity and validation results |

Keep the model and source together. The `policy` signature returns five action
heads; the full signature returns the
[serving tensors](../model/ARCHITECTURE.md#savedmodel-interface). Local applications
can use `model.inference.Predictor`.

## Recorded-action agreement

Compare a policy with both the human action and Mortal's recorded greedy action
on the reserved validation set:

```sh
python -m training.evaluate_imitation \
  --model /path/to/export/saved_model \
  --data /path/to/expert-data \
  --output /path/to/agreement.json
```

Use `--weights /path/to/checkpoint.weights.h5` instead of `--model` to evaluate
weights with matching source. The evaluator snapshots the checkpoint before
loading it and records its hash. `--max-decisions` limits evaluation to a
repeatable prefix of the validation sample.

The report contains exact match counts, disagreements, top-1 agreement and
supported-target NLL, overall and by decision type. Call agreement includes
the follow-up discard; branch agreement and discard agreement conditional on
the target's chi/pon are also reported. Use the same cohort when comparing
checkpoints. A small prefix may underrepresent uncommon decisions.

## Compare two models

```sh
python -m training.evaluate \
  --new /path/to/export/saved_model \
  --old /path/to/opponent/saved_model \
  --output /path/to/match \
  --games 10000 --seed 941 --workers 1 --devices -1
```

Each policy loads its own frozen feature and action source. Every wall seed
rotates the candidate through all four seats, so the game count must be a
positive multiple of four. Tile-copy aliases and exact policy ties use seeded
randomness keyed by wall, seat, event and decision phase. Neural outputs may
still differ slightly with inference batch shape.

Repeating an unchanged command resumes completed games. Changed model files
or settings require a new output directory. Use independent seeds for later
checkpoint comparisons.

| Output | Contents |
| --- | --- |
| `evaluation.json` | Model identities, rule format and seed range |
| `games.jsonl` and `replays/` | Completed game results and MJAI replays |
| `summary.json` | Placements, scores, ST3 points and paired bootstrap intervals |
| `tensorboard/` | Progress and comparison metrics |

Summaries include complete four-seat groups only. `new` identifies the
candidate and `old` its opponents. `--tensorboard` selects another event
directory. Lower training loss or higher action agreement alone does not
establish stronger play.

### Distributed match collection

Assign disjoint seed ranges to workers and keep each seed's four rotations
together. Use identical model, opponent and adapter files throughout.
`evaluate.merge_parts()` combines complete result lines from the worker
outputs. Finalization checks every requested seed/seat and replay, then copies
the replays into the merged match directory for training.

## Mortal opponent

The evaluator also accepts a Mortal checkpoint (`.pth`, or a ZIP containing
one checkpoint):

```sh
python -m training.evaluate \
  --new /path/to/export/saved_model \
  --old /path/to/mortal-checkpoint.zip \
  --mortal-source /path/to/Mortal \
  --output /path/to/mortal-match \
  --games 4 --seed 941 --workers 1 --devices -1
```

Install PyTorch in the evaluation environment. The adapter uses Mortal commit
`0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` and its matching `libriichi` extension.
Build with the same Python interpreter:

```sh
git clone https://github.com/Equim-chan/Mortal.git /path/to/Mortal
git -C /path/to/Mortal checkout 0cff2b52982be5b1163aa9a62fb01f03ce91e0d2
cd /path/to/Mortal
PYO3_PYTHON="$(command -v python)" cargo build --release --locked -p libriichi
python - <<'PY'
import pathlib
import shutil
import sysconfig

shutil.copy2(
    "target/release/libriichi.so",
    pathlib.Path(sysconfig.get_paths()["platlib"]) / "libriichi.so",
)
PY
```

Mortal receives each seat's private MJAI event stream, loads its checkpoint
strictly and uses greedy legal-action selection with the upstream win guard.
Mortal's source and weights remain external dependencies with their own
licenses. The match output records their identities for reproducibility.

Mortal matches use standard Tenhou South rules, recorded as
`riichienv_standard_south`. Model-versus-model matches use
`riichienv_duplicate_south`: per-seat draw queues, shared replacement draws and
14 tiles reserved at exhaustion. The rule formats are reported separately
and cannot be combined in one summary.

Start with four games. CPU inference uses `--devices -1`; a comma-separated
GPU UUID list selects GPU workers. `--batch-games` controls concurrent games.
Measure full-game throughput and memory before increasing concurrency.

## Evaluate a training checkpoint

`python -m training.checkpoint_match` exports the checkpoint under
`OUTPUT/policy/` and runs its comparison under `OUTPUT/matches/`. An existing
export must match the checkpoint hash. The
[training runner](corrections/README.md#run) accepts this command as a JSON
argument list.
