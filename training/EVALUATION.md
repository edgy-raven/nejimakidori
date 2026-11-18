# Export and evaluate

## Export weights

```sh
python -m model.export \
  --weights /path/to/run/final.weights.h5 \
  --data /path/to/dataset/data --output /path/to/export
```

To export without a corpus, replace `--data` with `--self-play-seed 941`.

| Output | Purpose |
| --- | --- |
| `saved_model/` | TensorFlow prediction signatures |
| `source/` | Matching source for isolated evaluation |
| `export-verification.json` | Weight hash and numerical checks |

Export checks finite outputs, normalized distributions, SavedModel round-trip
parity and policy/full-export parity. Keep `saved_model/` and `source/` together.
Local prediction uses `model.inference.Predictor`.
Response formatting materializes candidate waits and settlement moments once
per batch before building per-action JSON.

## Compare two exports

Frozen-policy adapters retain the complete-history wire contract but decode
only newly appended engine events. Reset policies before starting a new game
cohort. Paired bootstrap summaries process resamples in bounded blocks while
preserving the seeded sampling sequence and reported confidence bounds.

```sh
python -m training.evaluate \
  --new /path/to/export/saved_model \
  --old /path/to/opponent/saved_model \
  --output /path/to/match \
  --games 10000 --workers 1 --devices GPU-UUID
```

| Rule | Reason |
| --- | --- |
| Each player loads its own frozen source | Reproduces its feature and model contract |
| Rotate each seed through four seats | Controls seat allocation |
| Use a game count divisible by four | Keeps complete seed groups |
| Repeat the unchanged command to resume | Reuses completed games |
| Use disjoint seeds for each checkpoint | Later checkpoint matches do not reuse earlier training outcomes |

`summary.json` reports placements, scores, ST3 points and paired uncertainty
bounds. Lower training loss alone does not establish stronger play.

TensorBoard events are written to `OUTPUT/tensorboard`, or to the directory
selected with `--tensorboard`. They report completed games, percent complete,
each side's metrics, and paired differences with confidence bounds. `new`
identifies the `--new` model; `old` identifies its opponents.

## Distill policy opponent

The same harness accepts a distill policy checkpoint (`.pth` or a ZIP containing one
checkpoint) as the old opponent:

```sh
python -m training.evaluate \
  --new /path/to/export/saved_model \
  --old /path/to/model_v4_20240308_best_min.zip \
  --distill-source /path/to/upstream-policy \
  --output /path/to/distill-match \
  --games 4 --seed 941 --workers 1 --devices -1
```

Install PyTorch in the evaluator's Python environment. Use upstream distill policy
commit `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2`, including its matching
`libriichi` extension. Build the extension with the same Python interpreter:

```sh
git clone https://github.com/Equim-chan/Mortal.git /path/to/upstream-policy
git -C /path/to/upstream-policy checkout 0cff2b52982be5b1163aa9a62fb01f03ce91e0d2
cd /path/to/upstream-policy
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

The Linux adapter runs in a separate process, using only each seat's private
MJAI event stream. It follows the upstream agent flow: call `react()` for every
event from `obs.new_events()`, then map the response with
`obs.select_action_from_mjai()`. A trailing delayed-dora announcement updates
the reference policy's state without replacing the preceding decision response; successive
added kans can otherwise erase a valid rob-kan win. No action suppression or
automatic pass substitution is applied. It loads both state dictionaries
strictly, uses evaluation
mode with greedy legal-action selection, and enables upstream's win guard.
There is no training or exploration. RiichiEnv remains responsible for match
rules and rejects any action that cannot be mapped to a legal action.

The distill policy matches use standard Tenhou south-game rules, with four-seat rotations
per seed. Our custom duplicate mode preserves per-seat draw queues, borrowing from
the backs of other queues counterclockwise when a queue empties. All seats
share the common rinshan replacements; exhaustion leaves 14 tiles globally.
It also changes match progression; upstream distill policy matches use standard rules. These matches are therefore recorded as
`riichienv_standard_south`, separately from `riichienv_duplicate_south`.

The match directory freezes the checkpoint, upstream model/engine source,
native extension, and adapter source. Their SHA-256 hashes are recorded in
`evaluation.json`; changing them requires a fresh output directory. The
extension is specific to its Python ABI and platform. The distill policy's source and
weights remain external dependencies with their own licensing/provenance.

Start with a four-game smoke test. CPU inference is supported (`--devices -1`);
larger matches can use a training GPU UUID. Each worker shares one loaded
The distill policy network across its opponent seats. Upstream bots process their event
streams independently; pending neural requests are combined into one engine
call and distributed back to their callers in order. `--batch-games` controls
the available parallel games. This preserves the upstream event flow and
float32 inference, though GPU batching can produce small rounding differences.
Measure full-game throughput and memory before increasing worker counts.

To exercise the integration test with a real checkpoint:

```sh
DISTILL_SOURCE=/path/to/upstream-policy DISTILL_CHECKPOINT=/path/to/model.zip \
  CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. \
  python -m pytest integration/big/test_evaluation.py
```

## Evaluate a training pass

`python -m training.candidate.evaluation` exports a checkpoint and runs its
comparison. An existing export must match the checkpoint hash. The candidate
runner accepts this command as a JSON argument list; see
[candidate setup](candidate/README.md#1-configure-evaluation).
