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
Local prediction uses `model.inference.Predictor`, which validates the exact
serving outputs. The `policy` signature returns five action heads; the full
signature returns the [17 serving tensors](../model/ARCHITECTURE.md#tensor-signatures).
Internal training diagnostics are not exported. Response formatting computes
candidate waits and settlement moments once per batch before building
per-action JSON.

## Compare two exports

The evaluator owns transport and action randomness; each model still loads its
frozen feature and action source. Adapters send complete history and decode
only newly appended engine events. Reset policies with the wall seed for each
game before starting a new cohort.

Drawn/held aliases, physical tile copies and exact policy ties use randomness
keyed by wall seed, acting seat, event count and decision phase. Candidate seat,
model identity, worker and batch position are excluded. This preserves fair
alias choices and common randomness across seat rotations. Evaluation manifests
record this contract; older unseeded matches require a fresh output directory.
Numerical model outputs can still change with inference batch shape, so full
replay equality across arbitrary batch sizes is not guaranteed.

Paired bootstrap summaries process resamples in bounded blocks while
preserving the seeded sampling sequence and reported confidence bounds.

```sh
python -m training.evaluate \
  --new /path/to/export/saved_model \
  --old /path/to/opponent/saved_model \
  --output /path/to/match \
  --games 10000 --seed 941 --workers 1 --devices GPU-UUID
```

| Rule | Reason |
| --- | --- |
| Each player loads its own frozen source | Reproduces its feature and model contract |
| Rotate each seed through four seats | Controls seat allocation |
| Use a game count divisible by four | Keeps complete seed groups |
| Repeat the unchanged command to resume | Reuses completed games |
| Keep both SavedModel contents unchanged | Resume validates every model file before reading results |
| Use disjoint seeds for each checkpoint | Later checkpoint matches do not reuse earlier training outcomes |

`evaluation.json` records SHA-256 hashes for every requested SavedModel file.
Changed models and manifests without these identities require a fresh output
directory, even when all games have completed.

`summary.json` reports placements, scores, ST3 points and paired uncertainty
bounds. Lower training loss alone does not establish stronger play.

TensorBoard events are written to `OUTPUT/tensorboard`, or to the directory
selected with `--tensorboard`. They report completed games, percent complete,
each side's metrics, and paired differences with confidence bounds. `new`
identifies the `--new` model; `old` identifies its opponents.

## Mortal opponent

The same harness accepts a Mortal checkpoint (`.pth` or a ZIP containing one
checkpoint) as the old opponent:

```sh
python -m training.evaluate \
  --new /path/to/export/saved_model \
  --old /path/to/model_v4_20240308_best_min.zip \
  --mortal-source /path/to/upstream-policy \
  --output /path/to/mortal-match \
  --games 4 --seed 941 --workers 1 --devices -1
```

Install PyTorch in the evaluator's Python environment. Use upstream Mortal
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
MJAI event stream. It follows
[Mortal's agent flow](https://riichi.dev/docs/mortal): call `react()` for every
event from `obs.new_events()`, then map the response with
`obs.select_action_from_mjai()`. A trailing delayed-dora announcement updates
Mortal's state without replacing the preceding decision response; successive
added kans can otherwise erase a valid rob-kan win. No action suppression or
automatic pass substitution is applied. It loads both state dictionaries
strictly, uses evaluation mode with greedy legal-action selection, and enables
upstream's win guard.
There is no training or exploration. RiichiEnv remains responsible for match
rules and rejects any action that cannot be mapped to a legal action.

Mortal matches use standard Tenhou south-game rules, with four-seat rotations
per seed. Our custom duplicate mode preserves per-seat draw queues, borrowing from
the backs of other queues counterclockwise when a queue empties. All seats
share the common rinshan replacements; exhaustion leaves 14 tiles globally.
Duplicate mode also changes match progression. Mortal matches use standard
rules and are recorded as `riichienv_standard_south`, separately from
`riichienv_duplicate_south`.

The match directory freezes the checkpoint, upstream model/engine source,
native extension, and adapter source. Their SHA-256 hashes are recorded in
`evaluation.json`; changing them requires a fresh output directory. The
extension is specific to its Python ABI and platform. Mortal's source and
weights remain external dependencies with their own licensing/provenance.

Start with a four-game smoke test. CPU inference is supported (`--devices -1`);
larger matches can use a training GPU UUID. Each worker shares one loaded
Mortal network across its opponent seats. Upstream bots process their event
streams independently; pending neural requests are combined into one engine
call and distributed back to their callers in order. `--batch-games` controls
the available parallel games. This preserves the upstream event flow and
float32 inference, though GPU batching can produce small rounding differences.
Measure full-game throughput and memory before increasing worker counts.

To exercise the integration test with a real checkpoint:

```sh
MORTAL_SOURCE=/path/to/upstream-policy MORTAL_CHECKPOINT=/path/to/model.zip \
  CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. \
  python -m pytest integration/big/test_evaluation.py
```

## Evaluate a training pass

`python -m training.checkpoint_match` exports a checkpoint and runs its
comparison under `OUTPUT/policy/` and `OUTPUT/matches/`, respectively. An
existing export must match the checkpoint hash. The MMD
runner accepts this command as a JSON argument list; see
[MMD setup](corrections/README.md).
