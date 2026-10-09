# Nejimakidori v1 weights

Release: [`nejimakidori-v1`](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-v1).
Binary weights are release assets, separate from model source and Git history.
Use a dedicated weights directory outside the source checkout:

```sh
export WEIGHTS_PATH=/path/to/weights
mkdir -p "$WEIGHTS_PATH"
gh release download nejimakidori-v1 \
  --repo edgy-raven/nejimakidori --dir "$WEIGHTS_PATH" \
  --pattern 'nejimakidori-v1.tar.gz' --pattern SHA256SUMS
(cd "$WEIGHTS_PATH" && sha256sum -c SHA256SUMS)
tar -xzf "$WEIGHTS_PATH/nejimakidori-v1.tar.gz" -C "$WEIGHTS_PATH"
export RELEASE_PATH="$WEIGHTS_PATH/nejimakidori-v1"
```

The bundle contains:

- `saved_model/`: the evaluated inference export, with full and policy signatures.
- `final.weights.h5`: the matching trainable joint-model checkpoint.
- `source/`: matching inference source, including third-party notices.
- `evaluation/`: both completed 10,000-game comparisons.
- `runtime-dependencies.json` and a per-file SHA-256 `manifest.json`.

Build the runtime from [the repository instructions](../README.md) and mount
`saved_model/` read-only. `main` is the v1 source; experimental phasic training
is on [`v2`](https://github.com/edgy-raven/nejimakidori/tree/v2), initialized from
`final.weights.h5`. Joint weights are not a standalone belief initializer.

V1 is the paired-policy candidate evaluated against production 83570 and old
production 133712. Its ST3/game advantages were +0.02 (95% interval -2.97 to
+2.86) and +2.27 (-0.65 to +5.21), respectively. Promotion preserves the tested
weights; neither comparison established a positive ST3 advantage.
The previous production release remains available under its original tag.
