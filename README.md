# Nejimakidori

Nejimakidori (ねじまき鳥) is a four-player riichi mahjong model.
It predicts legal actions, hidden-hand properties and hand settlements from
the information available to a player. This repository contains the model,
dataset tools, training and evaluation workflows, and release metadata.

## Start here

| Task | Guide |
| --- | --- |
| Install dependencies | [Source setup](SETUP.md) |
| Build the training container | [Docker](docker/README.md) |
| Download and verify weights | [Weights](weights/README.md) |
| Prepare recorded games | [Datasets](log_dataset/README.md) |
| Train a model | [Training](training/README.md) |
| Export and compare models | [Evaluation](training/EVALUATION.md) |
| Understand predictions and constraints | [Model](model/README.md) |
| Explore inputs, outputs and gradient flow | [Model atlas](https://ogame.squareqrow.com/atlas/) |

## Repository map

| Directory | Owns |
| --- | --- |
| `model/` | Network, losses, inference and export |
| `log_dataset/` | Source games, features, labels and native simulator integration |
| `training/` | Imitation, self-play, policy improvement and evaluation |
| `weights/` | Release metadata and checksums |
| `integration/` | Model and training verification |
| `docker/` | Reproducible training environment |

## Development

After [setup](SETUP.md), run the correctness checks:

```sh
python -m pip install pytest
python smoke.py small
python smoke.py big
```

See [verification](integration/README.md) for test coverage and optional
performance comparisons. Release weights include their matching source;
follow the [weights guide](weights/README.md) to use them.
