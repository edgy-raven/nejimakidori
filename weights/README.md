# Model files

Download [Nejimakidori 2026-11-10](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-20261110)
and complete [source setup](../SETUP.md).

The release contains two downloads:

| Download | Contents |
| --- | --- |
| `nejimakidori-20261110.tar.gz` | Model, training weights, source and evaluation results. |
| `SHA256SUMS` | SHA-256 checksum for the archive. |

## Download and verify

Download the archive and checksum from the
[release](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-20261110):

```sh
mkdir -p artifacts
cd artifacts
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-20261110/nejimakidori-20261110.tar.gz
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-20261110/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf nejimakidori-20261110.tar.gz
cd ..
```

The archive includes `final.weights.h5` for training, a TensorFlow SavedModel,
matching source, and evaluation results. The contents are listed below.

## Archive contents

| Path | Purpose |
| --- | --- |
| `saved_model/` | TensorFlow SavedModel for prediction and evaluation. |
| `final.weights.h5` | Weights for initializing a new training run with `--initial`. |
| `source/` | Source snapshot matching the release. |
| `manifest.json` | Model identity and SHA-256 hashes for the included files. |
| `export-verification.json` | Export validation results. |
| `evaluation.json` | Results and paired uncertainty bounds from a 10,000-game comparison. |

Frozen evaluation uses the matching source included in the release. Training
weights contain no optimizer state and require a compatible model architecture. To continue training, see the [training guide](../training/README.md).
