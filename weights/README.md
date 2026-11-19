# Model files

Download [Nejimakidori v2](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-v2-expert-r1-20261119)
and use the matching `source/` bundled in the archive for inference or
evaluation. [Source setup](../SETUP.md) documents the current development tree.

This archive contains the expert-corpus R1 checkpoint. Its frozen source uses the
previous paired-correction recipe; the working training code now uses
[match-then-root MMD](../training/corrections/README.md).

The release contains two downloads:

| Download | Contents |
| --- | --- |
| `nejimakidori-v2-expert-r1-20261119.tar.gz` | Model, training weights, source and evaluation results. |
| `SHA256SUMS` | SHA-256 checksum for the archive. |

## Download and verify

Download the archive and checksum from the
[release](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-v2-expert-r1-20261119):

```sh
mkdir -p /path/to/artifacts/nejimakidori-v2
cd /path/to/artifacts/nejimakidori-v2
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-v2-expert-r1-20261119/nejimakidori-v2-expert-r1-20261119.tar.gz
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-v2-expert-r1-20261119/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf nejimakidori-v2-expert-r1-20261119.tar.gz
```

The archive includes `final.weights.h5` for training, a TensorFlow SavedModel,
matching source, and evaluation results. The contents are listed below.

## Archive contents

| Path | Purpose |
| --- | --- |
| `saved_model/` | TensorFlow SavedModel for prediction and evaluation. |
| `final.weights.h5` | Weights for an architecture-compatible training model. |
| `source/` | Source snapshot matching the release. |
| `manifest.json` | Model identity and SHA-256 hashes for the included files. |
| `export-verification.json` | Export validation results. |
| `evaluation.json` | Results and paired uncertainty bounds from a 10,000-game comparison. |

Frozen evaluation uses the matching source included in the release. The current
source has a different native input contract and serving signature; its strict
loader is not an adapter for this archive. In particular, the current policy
omits the learned `sakigiri_order` scalar and its scorer input channel. Weights
containing that branch are not compatible with the current architecture, and
no live conversion or runtime fallback is provided. Follow the
[native compatibility contract](../log_dataset/native_collection/README.md#input-tensors)
and validate prediction parity for any exact conversion before training or
re-exporting.

Training weights contain no optimizer state. Use `--initial` only with a
compatible model, then follow the [training guide](../training/README.md).
Keep downloaded archives, extracted weights and new training outputs outside
the source checkout. The committed manifest describes the frozen release,
including source files that no longer exist in the current tree.

## Evaluation

The release evaluation compares the selected checkpoint against the previous
v2 release using 10,000 games. See `evaluation.json` in the archive for the final results and
paired uncertainty bounds. The archive contains the exact evaluated
round-one correction weights.
