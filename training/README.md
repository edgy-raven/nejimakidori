# Training

The released checkpoint uses targeted paired decision corrections.
Only selected roots train the policy: each compares two actions using matched
current-hand continuations and a fixed-policy KL anchor.

## Guides

| Task | Guide |
| --- | --- |
| Collect paired roots and train | [Corrections](corrections/README.md) |
| Understand the objective | [Objective](OBJECTIVE.md) |
| Export and compare checkpoints | [Evaluation](EVALUATION.md) |
| Prepare recorded games | [Dataset guide](../log_dataset/README.md) |

## Source map

| Module | Responsibility |
| --- | --- |
| [corrections/](corrections/) | Root selection, paired outcomes and training |
| [self_play.py](self_play.py) | Simulator workers and continuations |
| [evaluate.py](evaluate.py) | Seeded, seat-rotated checkpoint matches |
| [frozen_policy.py](frozen_policy.py) | Isolated frozen-policy inference |
| [candidate/](candidate/) | Earlier candidate workflow and shared utilities |

Earlier candidate workflows remain separate from the paired correction runner.
