# V1 weights

Download [nejimakidori-v1](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-v1).
The [quick start](../README.md#run-v1) downloads the archive, verifies
`SHA256SUMS`, extracts it outside the checkout and starts inference.

## Bundle contents

```text
nejimakidori-v1/
├── saved_model/               TensorFlow inference export
├── final.weights.h5           Trainable joint-model weights
├── source/                   Matching model, dataset and service source
├── evaluation/               Two 10,000-game match summaries
├── runtime-dependencies.json Recorded dependency versions
└── manifest.json             Per-file SHA-256 hashes and release metadata
```

| Artifact | Use |
| --- | --- |
| `saved_model/` | Serving and frozen-policy evaluation. Includes full diagnostic and policy-only signatures. |
| `final.weights.h5` | Initialize the matching joint architecture for further training. Contains no optimizer recovery state. |
| `source/` | Build the runtime that matches these weights. |
| `manifest.json` | Verify which source, weights and evaluation belong to this release. |

The [runtime Dockerfile](../docker/Dockerfile.model) copies the bundled source
into the image and reads `saved_model/` from a read-only mount. Keep binary
artifacts outside Git. Do not substitute current development source for the
source required by a saved model's feature and output schemas.

Joint weights are not a standalone belief checkpoint. The current training
CLI requires `--beliefs` even when `--initial` supplies joint weights; in that
case the joint checkpoint takes precedence and the belief file is not loaded.
See [initialization](../model/README.md#initialization).

[Evaluation results](../README.md#evaluation) describe the released model.
