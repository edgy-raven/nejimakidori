# Nejimakidori v2

Nejimakidori (ねじまき鳥) is a four-player riichi mahjong model.
This repository contains its network, dataset tools, training and release weights.

The model combines tile-belief attention with a shared player critic for
current-hand settlement. This checkpoint adds targeted paired decision
corrections with a fixed-policy anchor.

## Guides

| Task | Guide |
| --- | --- |
| Install dependencies | [Source setup](SETUP.md) |
| Download and verify weights | [Weights](weights/README.md) |
| Prepare recorded games | [Datasets](log_dataset/README.md) |
| Train paired corrections | [Training](training/README.md) |
| Export and compare models | [Evaluation](training/EVALUATION.md) |
| Understand predictions | [Architecture](model/ARCHITECTURE.md) |
| Build the training container | [Docker](docker/README.md) |

## Repository map

| Directory | Contents |
| --- | --- |
| `model/` | Network, losses, inference and export |
| `log_dataset/` | Features, labels and simulator integration |
| `training/` | Collection, optimization and evaluation |
| `weights/` | Release metadata and checksums |
| `integration/` | Model and training verification |
| `docker/` | Training environment |

Matching source is included with the release. Training weights contain no
optimizer state. Each folder keeps its own usage guide and source map.
