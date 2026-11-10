# How the model works

Nejimakidori predicts a move and the likely consequences of the current hand.
A shared encoder reads the visible board, rivers and candidate hands.
Four residual board blocks feed the action policy, opponent beliefs,
state value and settlement predictions.
The joint network is assembled in [network.py](network.py).

For commands, see [training and evaluation](../training/README.md). For tensor names,
shapes and encoding, see the [native interface](../log_dataset/native_collection/README.md).

## From a game state to an action

1. The native engine reconstructs public state and legal actions from events.
2. Feature extraction builds the current hand and legal post-action hands,
   including call/discard combinations. Opponents' concealed tiles are excluded.
3. The shared encoder produces board, river and candidate representations.
4. Belief and value predictions feed the final policy. Legal masks remove
   unavailable choices before selection.

The policy has discard, riichi/discard, response, kan and win/pass heads. A response
includes its follow-up discard. Physical copies of the same tile share an
ordinary discard action; red fives remain distinct. Optional ron and tsumo
are learned choices; nine-terminals aborts remain automatic. Win/pass learns
from recorded choices and observed hand returns, using the fitted state-value
baseline rather than an action-conditioned win forecast.

[actor.py](actor.py) applies opening symmetry and final-wall
safety rules. During training, a detached policy target also guides early-hand
efficiency and late-hand safe discards. The `--prior-weight` option controls
that training loss. Exact rules live alongside their implementation.

## What the predictions mean

| Prediction | Meaning |
| --- | --- |
| Opponent beliefs | Shanten, structural waits and concealed tile-count marginals. Counts are constrained by public hand sizes and available copies. |
| Completion yaku and score | The highest-scoring legal completion of the next observed tenpai shape, separately for ron and tsumo. Unknown targets are masked. |
| Route affinity | How much of a candidate's visible shortest-route support belongs to tanyao, flush or outside-hand families. |
| Immediate outcomes | Ron claims and payments, exhaustive draws and aborts at the current action boundary. |
| Future settlement | A probability distribution over ways the hand can end and the resulting transfers. |
| State value (training) | Expected observed hand return with fixed continuation, in ST3 / 100 units; conditioned on the pre-action state. |
| Final placement | Settlement outcomes integrated through a fixed continuation model for the rest of the match. |

Structural waits can be blocked by furiten or lack of yaku. Completion potential
and route affinity therefore have different meanings from win probability.
The candidate value-range inputs are also conditional completion bounds: they
search available finishes from zero/one-shanten hands, without predicting how
likely those finishes are.

Call forecasts condition on attempting the call, including preemption by
competing claims. Accepted-call features describe the hypothetical candidate;
hard scoring constraints use guaranteed pre-call facts. The planned discard
is contingent, so its outcomes belong to the joint hand forecast rather than
a guaranteed immediate transition. Training retains preempted attempts.
Response passes also await priority resolution: neither last-wall exhaustion
nor acceptance of a pending riichi declaration is assumed at that boundary.

## Scoring and match value

The critic predicts events and hand values. Rule-based settlement converts
those into payments, handling dealer shares, honba, deposits and pot priority.
Tsumo shares use one common hand value. Multiple ron claims are represented
jointly; triple ron follows the simulator's abort rule.

[payment_scoring.py](payment_scoring.py) defines the score vocabulary and
legality masks. [joint_settlement.py](joint_settlement.py) combines events,
scores and deposits. Unsupported payments, including some responsibility and
nagashi allocations, are excluded from settlement likelihood and counted in
coverage metrics.

Training values actions by projected Saint 3 (ST3) match points. Completed
matches use exact final scores and ranks. Earlier hands use a fixed continuation
model based on scores, remaining hands and scheduled dealerships. Its parameters
are separate from the learned current-hand critic.

## Training

The [candidate objective](../training/OBJECTIVE.md) defines actor weighting,
V targets, fixed-policy KL and gradient boundaries. The critic and V predict
current-hand outcomes projected through the same fixed continuation model.

## Where to change things

| Area | Source |
| --- | --- |
| Public features and packed layout | [native collection](../log_dataset/native_collection/README.md) |
| Joint model assembly | [network.py](network.py) |
| Board, hand, river and action representations | [encoder.py](encoder.py) |
| Opponent beliefs, count constraints and supervision | [beliefs.py](beliefs.py) |
| Policy and candidate auxiliary heads, priors and losses | [actor.py](actor.py) |
| Legal actions, model indexing and native selection | [actions.py](actions.py) |
| Payment critic, immediate outcomes and action forecasts | [critic.py](critic.py) |
| State-value head and regression | [network.py](network.py), [trainer.py](../training/trainer.py) |
| Score support and settlement | [payment_scoring.py](payment_scoring.py), [joint_settlement.py](joint_settlement.py) |
| Component losses and gradient boundaries | [actor.py](actor.py), [beliefs.py](beliefs.py), [critic.py](critic.py) |
| Shared loss reductions and optimizer updates | [losses.py](losses.py), [optimization.py](../training/optimization.py) |
| Self-play and candidate updates | [self_play.py](../training/self_play.py), [trainer.py](../training/trainer.py) |
| Frozen sources and isolated opponents | [frozen_policy.py](../training/frozen_policy.py) |
| Recorded labels and sampling | [dataset guide](../log_dataset/README.md) |

## Constraints and compatibility

See [constraints](CONSTRAINTS.md) for masks, raw-head supervision, call priority
and schema compatibility.
