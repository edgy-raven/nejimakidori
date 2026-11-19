# Paired correction objective

For a selected position, compare the frozen policy action `a` with alternative
`b`. Both branches use the same unseen-wall seed and continue to the current
hand's settlement. Project each settlement through the fixed continuation map:

`J = ST3 / 100 + P(first) - P(fourth)`

With `delta = J(b) - J(a)` and `p = pi(b) / (pi(a) + pi(b))`, the actor loss is:

`-p * delta + 0.3 * KL(pi || fixed_policy)`

The KL covers every legal action at the root. Positive and negative gaps both
train; ordinary game positions and continuation decisions do not. Q proposes
disagreement roots but does not supply the policy reward. There is no state V,
imitation objective, entropy bonus or separate downside term in this runner.

Both observed branch payments train the current-hand critic with coefficient
0.1. Existing structural constraints retain coefficient 0.01. Policy and critic
updates share the encoder. This can affect decisions outside the sampled roots.

See [the collection and training contract](corrections/README.md) for categories,
labels, limits and commands. [trainer.py](corrections/trainer.py) implements the
objective, and [critic.py](../model/critic.py) defines payment supervision.
