# Model

## Guides

| Topic | Read |
| --- | --- |
| Inputs, predictions and settlement | [Architecture](ARCHITECTURE.md) |
| Legality, gradient constraints and compatibility | [Constraints](CONSTRAINTS.md) |
| Training objective | [Objective](../training/OBJECTIVE.md) |
| Export and match evaluation | [Evaluation](../training/EVALUATION.md) |
| Explore inputs, outputs and architecture | [Interactive catalog](https://ogame.squareqrow.com/atlas/) |

## Source map

| Area | Modules |
| --- | --- |
| Network and shared representations | [network.py](network.py), [encoder.py](encoder.py) |
| Candidate and river attention | [candidate_fusion.py](candidate_fusion.py), [river_attention.py](river_attention.py) |
| Neural input encoding and categorical masks | [input_features.py](input_features.py) |
| Candidate route and ron heads | [candidate_predictions.py](candidate_predictions.py) |
| Policy and legal choices | [actor.py](actor.py), [actions.py](actions.py) |
| Shared opponent and completion beliefs | [beliefs.py](beliefs.py) |
| Action outcomes and settlement | [critic.py](critic.py), [joint_settlement.py](joint_settlement.py) |
| Score vocabulary and legality | [outcomes.py](outcomes.py), [payment_scoring.py](payment_scoring.py) |
| Shared loss reductions | [losses.py](losses.py) |
| SavedModel creation and loading | [export.py](export.py), [inference.py](inference.py) |

Training and evaluation commands are in [training/](../training/README.md).

## Prediction inputs

`Predictor.infer()` accepts a list of feature objects produced by
`log_dataset.features.pack()`. Each object includes its legal chi/pon states.
Use `log_dataset.call_states.batch()` to prepare direct SavedModel inputs.

| Fields | Batched shape | Meaning |
| --- | --- | --- |
| Base features | `[B, ...]` | Current decision states |
| `post_call_<feature>` | `[C, ...]` | Accepted chi/pon states before discarding |
| `post_call_owner` | `[C]` | Index into the base decision batch |
| `post_call_option` | `[C]` | Chi patterns 0–2 or pon 3 |

`C` counts available call branches across the batch and may be zero. Pass and
open kan have no immediate discard state. For a single feature object, owners
are zero; batching offsets them. Empty JSON call arrays are valid. Base and
post-call tensors must not be stacked or sliced as if they shared a row axis.
