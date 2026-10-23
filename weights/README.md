# Weights and exports

Model binaries live outside this checkout. The
[v1 release](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-v1)
has its own [download and run instructions](https://github.com/edgy-raven/nejimakidori/tree/main#run-v1).
V1 weights require their matching source and cannot initialize the current v2
architecture.

For v2, follow [training and export](../model/README.md) to create compatible
artifacts:

| Artifact | Use |
| --- | --- |
| `latest.weights.h5` | Start fresh from the latest checkpoint with `--initial`. |
| `final.weights.h5` | Initialize a new joint-model training run with `--initial`. Contains model weights only. |
| `recovery/ckpt-*` | Resume a matching phasic run, including optimizer state and counters. |
| `policy/saved_model/` | Serve predictions or evaluate against another model. |
| `policy/source/` | Source preserved by export for the frozen player. |

`model.training` initializes joint weights with `--initial` and writes
`latest.weights.h5` alongside recovery checkpoints for fresh starts.

Keep the export and its source together. The current
[Dockerfile](../docker/Dockerfile.model) builds source from its build context;
mounting an older SavedModel does not replace that code with its release source.
