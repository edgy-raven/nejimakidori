# Released model

The current model is
[Nejimakidori distill](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-distill-20261110).
It is the checkpoint used by the production API and RiichiLab atk.

After completing [setup](../README.md#setup), download and verify the bundle:

```sh
mkdir -p artifacts
cd artifacts
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-distill-20261110/nejimakidori-distill-20261110.tar.gz
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-distill-20261110/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf nejimakidori-distill-20261110.tar.gz
cd ..
MODEL_PATH="$PWD/artifacts/nejimakidori-distill-20261110/saved_model" \
  python -m service.serve
```

Use the source at the release tag with these weights. The bundle also retains
its matching source snapshot. It contains:

| Artifact | Use |
| --- | --- |
| `saved_model/` | Serve predictions or evaluate the released policy. |
| `final.weights.h5` | Initialize AWR with `--initial`; no optimizer state. |
| `source/` | Matching model, feature, serving and training source. |
| `manifest.json` | File hashes and checkpoint provenance. |
| `evaluation.json` | The 10,000-game comparison with the previous model. |

The comparison measured +0.785 ST3 points/game, with paired bounds of
−2.33 to +3.83. This does not establish a strength improvement.

See [serving](../service/README.md) for requests and configuration, or
[training](../training/README.md) to continue from `final.weights.h5`.
The Docker runtime builds source from its build context; a SavedModel mount
does not replace that source.
