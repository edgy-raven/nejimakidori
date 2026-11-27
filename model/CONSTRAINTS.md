# Constraints and gradients

[Architecture](ARCHITECTURE.md) · [Training objective](../training/OBJECTIVE.md)

## Legal support

The native engine determines legal actions. Forward masks remove unavailable
choices before policy normalization. Uncomputed candidate slots are padding;
they do not receive a penalty for their arbitrary score offset. Positions
without a legal win opportunity do not train the model to decline wins.

Opponent shanten is fixed to tenpai after declared riichi or four fixed melds.
Count marginals respect known meld copies, concealed hand sizes and shared
remaining-copy limits. These marginals do not define a joint distribution
over complete hidden hands.

Completion yaku masks exclude publicly impossible combinations. Our own
completion-score probabilities respect the corresponding yaku lower bounds and yakuman mass.
Possible yaku do not become positive labels merely because they are feasible.
Candidate ron requires a structural wait and learned eligibility, with zero
liability for publicly safe tiles and actions without a discard.
Both wait and eligibility are inferred from the observed state once per
opponent and tile. Changing a hypothetical hand or cut does not change the
opponent posterior. Hypothetical calls inherit the parent posterior; their
future completion and settlement predictions can depend on the action.

Settlement support enforces score legality, dealer shares, deposits, claim
priority and hand-ending rules. Last-discard boundaries still allow ron before
a draw or abort. Four fixed melds imply tenpai but do not imply a riichi
deposit. Call attempts retain the possibility of preemption by other claims.

Opening symmetry pools equivalent discards without changing their total
probability. Final-wall safety rules preserve legal branch mass and retain
possible nagashi mangan outcomes.

## Gradient boundaries

| Objective or connection | Gradient behavior |
| --- | --- |
| Direct belief supervision | Trains the belief readout and its shared representation |
| Observed settlement likelihood | Trains the critic, shared encoder and candidate attention |
| Candidate route supervision | Trains the readout, shared encoder and candidate attention |
| Immediate ron supervision | Trains the board-level eligibility readout and shared encoder; structural waits are detached |
| Belief consumed by another prediction | Detached at the consumption boundary |
| Shared representation consumed by the actor | Detached |
| Q used by policy optimization | Detached |

Board-level beliefs read observed state and our current concealed hand, with
no candidate-bank pooling or candidate-analysis inputs. Candidate attention
allows a selected candidate's predictive loss to reach other legal representations;
it does not create labels for unobserved alternatives. Unknown targets,
padding, already-riichi decisions and forced choices contribute no supervised
training mass. Discretionary dama discards remain eligible.

Forward constraints also provide detached targets for raw predictions.
Categorical projections use forward KL; binary marginals use Bernoulli KL.
Count supervision differentiates the constrained solution while enforcing
hand-size and copy limits. Each constraint belongs to its producing head and
is included once in the relevant objective.

Critic directional penalties encourage greater opponent tenpai to increase
that opponent's winning probability and exhaustive-draw readiness, and greater
wait probability to increase ron on the candidate tile. Other inputs are
held fixed and probe inputs are detached. An own-hand direction penalty
also reduces the explicit candidate shanten by one (down to tenpai), holding
all other inputs fixed, and penalizes a decrease in expected
`J = projected_ST3 / 100 + P(first) - P(fourth)` with squared positive error.
It applies to nonterminal discard candidates above tenpai, including call
follow-up discards; final-wall and abort states are excluded because their
readiness support is fixed. It trains the critic without moving the encoder
or belief inputs. `shanten_direction_loss` reports its contribution to the
unit-weight critic constraint objective; its coefficient does not decay.
These are soft consistency penalties,
not guarantees of global monotonicity.

## Data and checkpoint contracts

Feature layouts, labels and score vocabularies must agree across datasets,
weights and exports. Consumers validate the declared schema and signatures.
Regenerate records when their contract changes; changing metadata does not
convert stored tensors. A released export must be used with its matching
source and feature producer.
