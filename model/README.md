# Model and inference

Network components, feature representations, scoring and inference live here.
See [architecture](ARCHITECTURE.md) for model behavior and tensor ownership.

Training runners, optimizers, collection, evaluation and experiment tools live
in [training/](../training/README.md). Model imports do not load those workflows;
the export CLI imports snapshot and collection helpers only when invoked.

Export a checkpoint with `python -m model.export --help`. Run the HTTP API
using the [service guide](../service/README.md).
