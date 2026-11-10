# Model

## Guides

| Topic | Read |
| --- | --- |
| Inputs, predictions and settlement | [Architecture](ARCHITECTURE.md) |
| Legality, gradient constraints and compatibility | [Constraints](CONSTRAINTS.md) |
| Current training objective | [Objective](../training/OBJECTIVE.md) |
| Export and match evaluation | [Evaluation](../training/EVALUATION.md) |

## Source map

| Area | Modules |
| --- | --- |
| Network and shared representations | [network.py](network.py), [encoder.py](encoder.py) |
| Policy and legal choices | [actor.py](actor.py), [actions.py](actions.py) |
| Opponent beliefs | [beliefs.py](beliefs.py) |
| Action outcomes and settlement | [critic.py](critic.py), [joint_settlement.py](joint_settlement.py) |
| Score vocabulary and legality | [outcomes.py](outcomes.py), [payment_scoring.py](payment_scoring.py) |
| Shared loss reductions | [losses.py](losses.py) |
| SavedModel creation and loading | [export.py](export.py), [inference.py](inference.py) |

Training orchestration lives in [training/](../training/README.md). Model imports
do not load runners; export loads snapshot and collection helpers when invoked.
