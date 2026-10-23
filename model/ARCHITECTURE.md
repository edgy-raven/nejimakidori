# How the model works

Nejimakidori predicts a move and the likely consequences of the current hand.
A shared encoder reads the visible board, rivers and candidate hands.
Four residual board blocks feed the action policy, opponent beliefs and
settlement predictions.
The joint network is assembled in [network.py](network.py).

For commands, see [training and evaluation](README.md). For tensor names,
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
from recorded choices and PPO hand returns, using the current-hand critic
baseline rather than an action-conditioned win forecast.

The win head adds trainable weights. Earlier checkpoints lack those weights,
and corpora built with automatic wins omit these decisions. Regenerate training
records to include win/pass examples; record field shapes are unchanged.

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
| Final placement | Settlement outcomes integrated through a fixed continuation model for the rest of the match. |

Structural waits can be blocked by furiten or lack of yaku. Completion potential
and route affinity therefore have different meanings from win probability.
The candidate value-range inputs are also conditional completion bounds: they
search available finishes from zero/one-shanten hands, without predicting how
likely those finishes are.

Call forecasts are conditional on the call taking effect. A competing ron or
another accepted call can preempt it. The API preserves that distinction when
interpreting recorded response decisions; see [the service guide](../service/README.md#decision-semantics).

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

## What training optimizes

Offline training combines expert-action imitation with settlement, belief and
auxiliary supervision. Advantage-weighted regression (AWR) gives expert actions
more weight when their projected ST3 exceeds the policy baseline. The weights
are detached from the critic. Conditional call forecasts cannot compare
submitted responses fairly, so response decisions receive neutral AWR weights.

Self-play uses proximal policy optimization (PPO) on sampled actions and their
observed returns. It also trains beliefs and the settlement critic on eligible
fresh positions. Expert AWR records and PPO trajectories remain separate.

The final optional AWR pass penalizes divergence from the actor saved at the end
of PPO. [The training guide](README.md#phasic-training) explains phase budgets,
optimizers and checkpoint recovery.

## Where to change things

| Area | Source |
| --- | --- |
| Public features and packed layout | [native collection](../log_dataset/native_collection/README.md) |
| Joint model assembly | [network.py](network.py) |
| Board, hand, river and action representations | [encoder.py](encoder.py) |
| Opponent beliefs, count constraints and supervision | [beliefs.py](beliefs.py) |
| Policy and candidate auxiliary heads, priors and losses | [actor.py](actor.py) |
| Legal actions, model indexing and native selection | [actions.py](actions.py) |
| Payment critic, immediate outcomes and AWR values | [critic.py](critic.py) |
| Score support and settlement | [payment_scoring.py](payment_scoring.py), [joint_settlement.py](joint_settlement.py) |
| Component losses and gradient boundaries | [actor.py](actor.py), [beliefs.py](beliefs.py), [critic.py](critic.py) |
| Shared loss reductions and optimizer updates | [losses.py](losses.py), [optimization.py](optimization.py) |
| Self-play and phase scheduling | [self_play.py](self_play.py), [training.py](training.py) |
| Frozen sources and isolated opponents | [frozen_policy.py](frozen_policy.py) |
| Recorded labels and sampling | [dataset guide](../log_dataset/README.md) |

Feature layouts, label IDs and score vocabularies must agree across datasets,
weights and exports. Dataset summaries are checked against the current source.
When these definitions change, regenerate records and supply compatible weights.
Retained exports include their matching source for evaluation.

Each component defines its differentiable objectives alongside its forward
model. `training.py` combines them under one gradient tape per update, so the
shared encoder receives their combined gradient. Actor AWR weights and anchor
targets remain detached; the critic owns its chunked custom backward pass.
Validation and phase scheduling use the same training coordinator.
