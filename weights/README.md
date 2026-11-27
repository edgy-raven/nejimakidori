# Pretrained weights

Download [Nejimakidori v2](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-v2-expert-r1-20261119)
and use the matching source bundled with the release. Its architecture and
input contract differ from the current development source.

## Download and verify

Run these commands in a directory outside the source checkout:

```sh
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-v2-expert-r1-20261119/nejimakidori-v2-expert-r1-20261119.tar.gz
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-v2-expert-r1-20261119/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf nejimakidori-v2-expert-r1-20261119.tar.gz
```

## Contents

| Path | Purpose |
| --- | --- |
| `saved_model/` | TensorFlow SavedModel for prediction and evaluation |
| `final.weights.h5` | Training weights; no optimizer state |
| `source/` | Source matching the model and native input contract |
| `manifest.json` | Model identity and included-file checksums |
| `export-verification.json` | Export validation results |
| `evaluation.json` | Results and paired uncertainty bounds from 10,000 games |

Follow [source setup](../SETUP.md) in the bundled source to use the release.
The committed [manifest](manifest.json) and [checksum](SHA256SUMS) describe
that archive. For further training, use architecture-compatible weights with
`--initial`; see the [training guide](../training/README.md).
