# Production weights

Production release: [`production-83570-20261008`](https://github.com/edgy-raven/nejimakidori/releases/tag/production-83570-20261008).
The binary bundle is a release asset, separate from model source and Git history.
Use a dedicated weights directory outside the source checkout:

```sh
export WEIGHTS_PATH=/path/to/weights
mkdir -p "$WEIGHTS_PATH"
gh release download production-83570-20261008 \
  --repo edgy-raven/nejimakidori --dir "$WEIGHTS_PATH" \
  --pattern 'production-83570-20261008.tar.gz' --pattern SHA256SUMS
(cd "$WEIGHTS_PATH" && sha256sum -c SHA256SUMS)
tar -xzf "$WEIGHTS_PATH/production-83570-20261008.tar.gz" -C "$WEIGHTS_PATH"
export RELEASE_PATH="$WEIGHTS_PATH/production-83570-20261008"
```

The unpacked bundle contains `saved_model/`, matching inference `source/`,
`runtime-dependencies.json` and a per-file SHA-256 `manifest.json`. Its source
includes the Nyanten license and provenance. The model is unchanged from the
previous production bundle; the publication adds the missing vendor notices.

Build the runtime from [the repository instructions](../README.md).
Mount `saved_model/` read-only; never pair these weights with current candidate
training source. A training initializer is a separate input, not this export.
The checked-in manifest and SHA256SUMS pin this release. New candidate exports
remain local until evaluated and explicitly promoted.
