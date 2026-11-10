# Export and evaluate

## Export weights

```sh
python -m model.export \
  --weights /path/to/run/mixed-awr-5.weights.h5 \
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

## Compare two exports

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
| Use fresh seeds for confirmation | Separates model selection from confirmation |

`summary.json` reports placements, scores, ST3 points and paired uncertainty
bounds. Lower training loss alone does not establish stronger play.

## Evaluate a training pass

`python -m training.candidate.evaluation` exports a checkpoint and runs its
comparison. An existing export must match the checkpoint hash. The candidate
runner accepts this command as a JSON argument list; see
[candidate setup](candidate/README.md#1-configure-evaluation).
