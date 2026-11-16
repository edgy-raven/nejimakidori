# Training

`main` supports the current candidate: recorded and generated replay trained
with Q-weighted expert imitation, magnetic self-play improvement and
positive expert priors.

Recorded imitation uses equal expert/distill policy shares with separate Q weights.
The [hybrid workflow](HYBRID.md) describes reusable greedy labels and their
Q-weighted objective.

## Guides

| Task | Read |
| --- | --- |
| Start a candidate and monitor progress | [Candidate](candidate/README.md) |
| Understand losses, units and gradient boundaries | [Objective](OBJECTIVE.md) |
| Understand which positions and action pairs are sampled | [Sampling](candidate/SAMPLING.md) |
| Export weights and compare playing strength | [Evaluation](EVALUATION.md) |
| Fit Q on completed match logs | [Match records](MATCH_RECORDS.md) |

## Source map

| Module | Responsibility |
| --- | --- |
| [candidate/](candidate/) | Run scheduling, replay pools, collection and progress |
| [trainer.py](trainer.py) | Source-specific actor gradients, optimizer steps and distribution across training devices |
| [hybrid.py](hybrid.py) | Aligned expert records and distill-policy label reader |
| [distill_policy.py](distill_policy.py) | Frozen upstream policy adapter |
| [match_records.py](match_records.py) | Current-hand critic data from completed matches |
| [temperature.py](temperature.py) | Rolling ESS calibration for expert and distill policy temperatures |
| [self_play.py](self_play.py) | Simulator workers, action selection and paired continuations |
| [evaluate.py](evaluate.py) | Seeded, seat-rotated matches between exports |
| [frozen_policy.py](frozen_policy.py) | Frozen source snapshots and isolated opponent processes |

## Development policy

Keep `main` aligned with the selected candidate. Develop alternative objectives,
architectures and experiment runners on separate branches. Merge only the
selected implementation, its necessary tests and its documentation.
