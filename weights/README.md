# Model files

Download [Nejimakidori 2026-11-10](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-20261110)
and follow the [installation guide](../README.md#run-the-model).

The release contains two downloads:

| Download | Contents |
| --- | --- |
| `nejimakidori-20261110.tar.gz` | Model, training weights, source and evaluation results. |
| `SHA256SUMS` | SHA-256 checksum for the archive. |

Verify the archive before extracting it:

```sh
sha256sum -c SHA256SUMS
tar -xzf nejimakidori-20261110.tar.gz
```

The extracted directory contains:

| Path | Purpose |
| --- | --- |
| `saved_model/` | TensorFlow model used by the inference service. |
| `final.weights.h5` | Weights for initializing a new training run with `--initial`. |
| `source/` | Source snapshot matching the release. |
| `manifest.json` | Model identity and SHA-256 hashes for the included files. |
| `export-verification.json` | Export validation results. |
| `evaluation.json` | Results and paired uncertainty bounds from a 10,000-game comparison. |

Use the source and weights from the same release. Training weights contain no
optimizer state. To continue training, see the [training guide](../training/README.md).
