# Training

The active loop is [match-then-root magnetic mirror descent](corrections/README.md).
Play 10,000 matches against Mortal, train on all four seats, select a balanced
mixture of eligible and normal roots, and train again on self-play reruns.
Both phases update the actor at every retained decision, including descendants.

Expert imitation is reserved for restoring learned structures. The
[restoration labels](restoration.py), the finite
[dataset reader](../log_dataset/training_data.py), dataset preparation and model supervised
losses remain available; they are not improvement stages in the MMD loop.

| Module or guide | Responsibility |
| --- | --- |
| [corrections/](corrections/README.md#source-layout) | Match/root runner, collection, rollout storage and MMD |
| [OBJECTIVE.md](OBJECTIVE.md) | Losses, units and gradient boundaries |
| [trainer.py](trainer.py) | Shared optimizer, detached Q and critic fitting |
| [restoration.py](restoration.py) | Observed policy labels for structural restoration |
| [match_records.py](match_records.py) | Current-hand outcomes from match logs |
| [MATCH_RECORDS.md](MATCH_RECORDS.md) | Match-record transport contract |
| [self_play.py](self_play.py) | Native trajectory collection and batched inference |
| [checkpoint_match.py](checkpoint_match.py) | Verified export and match command |
| [EVALUATION.md](EVALUATION.md) | Export and independent playing-strength checks |
| [evaluate.py](evaluate.py) | Seeded, seat-rotated export comparisons |
| [mortal_policy.py](mortal_policy.py) | Frozen Mortal opponent adapter |
| [frozen_policy.py](frozen_policy.py) | Frozen source and isolated opponents |
| [monitor.py](monitor.py) | Atomic progress metadata |

Keep the selected implementation on `main`. Alternative architectures and
objectives belong on separate branches.
