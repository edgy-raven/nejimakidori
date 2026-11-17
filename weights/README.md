# Model files

Download [Nejimakidori v2.0](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-v2.0)
and complete [source setup](../SETUP.md).

The release contains two downloads:

| Download | Contents |
| --- | --- |
| `nejimakidori-v2.0.tar.gz` | Model, training weights, source and evaluation results. |
| `SHA256SUMS` | SHA-256 checksum for the archive. |

## Download and verify

Download the archive and checksum from the
[release](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-v2.0):

```sh
mkdir -p artifacts
cd artifacts
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-v2.0/nejimakidori-v2.0.tar.gz
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-v2.0/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf nejimakidori-v2.0.tar.gz
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

## Evaluation

The released checkpoint completed 10,000 games against the previous v2 release,
with a mean difference of +1.4422 ST3 points/game (paired 95% bounds
-1.1501 to +4.1367). This comparison does not establish a strength advantage.
The release contains the evaluated policy weights, before subsequent match-log
critic fitting.
