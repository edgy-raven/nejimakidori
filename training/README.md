# Training

`main` supports the current candidate: recorded and generated replay trained
with AWR, a fixed reference policy, and optional risk-adjusted action weights.

## Guides

| Task | Read |
| --- | --- |
| Start a candidate and monitor progress | [Candidate](candidate/README.md) |
| Understand losses, units and gradient boundaries | [Objective](OBJECTIVE.md) |
| Understand which positions and action pairs are sampled | [Sampling](candidate/SAMPLING.md) |
| Export weights and compare playing strength | [Evaluation](EVALUATION.md) |

## Source map

| Module | Responsibility |
| --- | --- |
| [candidate/](candidate/) | Run scheduling, replay pools, collection and progress |
| [trainer.py](trainer.py) | AWR update and distribution across training devices |
| [optimization.py](optimization.py) | Gradient clipping, synchronization and optimizer step |
| [self_play.py](self_play.py) | Simulator workers, action selection and paired continuations |
| [evaluate.py](evaluate.py) | Seeded, seat-rotated matches between exports |
| [frozen_policy.py](frozen_policy.py) | Frozen source snapshots and isolated opponent processes |

## Development policy

Keep `main` aligned with the selected candidate. Develop alternative objectives,
architectures and experiment runners on separate branches. Merge only the
selected implementation, its necessary tests and its documentation.
