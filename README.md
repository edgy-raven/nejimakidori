# Nejimakidori v2 hotfix

Nejimakidori (ねじまき鳥) is a four-player riichi mahjong model.
This repository contains its network, dataset tools, training and release weights.

The v2 hotfix corrects action-conditioned hand-value features: closed dama
hands no longer inherit riichi scoring, and no-yaku dama has no legal ron.
It retains shared river encoding, candidate attention and current-hand
settlement learning. The released checkpoint completed a recorded-policy
training pass with Q-weighted imitation and a fixed production anchor.
See the [architecture](model/ARCHITECTURE.md) and
[training objective](training/OBJECTIVE.md) for the complete method.

## Start here

| Task | Guide |
| --- | --- |
| Install dependencies | [Source setup](SETUP.md) |
| Build the training container | [Docker](docker/README.md) |
| Download and verify weights | [Weights](weights/README.md) |
| Prepare recorded games | [Datasets](log_dataset/README.md) |
| Train the current candidate | [Training](training/README.md) |
| Export and compare models | [Evaluation](training/EVALUATION.md) |
| Understand predictions and constraints | [Model](model/README.md) |

## Repository map

| Directory | Owns |
| --- | --- |
| `model/` | Network, losses, inference and export |
| `log_dataset/` | Source games, features, labels and native simulator integration |
| `training/` | Current candidate, collection, updates and evaluation |
| `weights/` | Release metadata and checksums |
| `integration/` | Model and training verification |
| `docker/` | Reproducible training environment |

## Setup

Follow [source setup](SETUP.md), then build a dataset or download weights.
Shared Python dependencies are listed in [requirements.txt](requirements.txt).

## Development

`main` tracks the selected candidate. Alternative experiments belong on separate
branches. Each folder keeps its own Markdown guides and source map.

```sh
python -m pip install pytest
python smoke.py small
python smoke.py big
```

The full suite includes CPU performance checks.
