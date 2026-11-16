# How the model works

Nejimakidori predicts a move and the likely consequences of the current hand.
A shared encoder reads the visible board, rivers and candidate hands.
Four residual board blocks feed the action policy, opponent beliefs
and settlement predictions.
The joint network is assembled in [network.py](network.py).

For commands, see [training and evaluation](../training/README.md). For tensor names,
shapes and encoding, see the [native interface](../log_dataset/native_collection/README.md).

## From a game state to an action

1. The native engine reconstructs public state and legal actions from events.
2. Feature extraction builds the current hand and legal post-action hands,
   including call/discard combinations. Opponents' concealed tiles are excluded.
3. The shared encoder produces board, river and candidate representations.
4. Belief and completion predictions feed the final policy. Legal masks remove
   unavailable choices before selection.

The policy has discard, riichi/discard, response, kan and win/pass heads. A response
includes its follow-up discard. Physical copies of the same tile share an
ordinary discard action; red fives remain distinct. Optional ron and tsumo
are learned choices; nine-terminals aborts remain automatic. Win/pass learns
from recorded choices and action-conditioned current-hand forecasts.

[actor.py](actor.py) applies opening symmetry and final-wall safety rules.
Its other heuristic prior outputs are training diagnostics.
Positive-only first-riichi and call priors add an expert cross-entropy term
under the [candidate objective](../training/OBJECTIVE.md#fixed-policy-pull).
Their threat and post-action scoring facts bypass learned encoder projections;
they neither change inference logits nor introduce new trained parameters.

## What the predictions mean

| Prediction | Meaning |
| --- | --- |
| Opponent beliefs | Shanten, structural waits and held tile-count marginals, including known melds. Concealed counts respect hand sizes and available copies; riichi and four fixed melds fix shanten to tenpai. |
| Completion yaku and score | The highest-scoring legal completion of the next observed tenpai shape, separately for ron and tsumo. Unknown targets are masked. |
| Route affinity | How much of a candidate's visible shortest-route support belongs to tanyao, flush or outside-hand families. |
| Hand settlement | A probability distribution over ways the hand can end and the resulting transfers. |
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

Root scoring computes settlement payoffs once per public position and reuses
them across candidate actions. Only the settlement probabilities depend on the
action. Payoff construction and action forecasts use bounded GPU chunks; the
shared payoff tensor exists only for the current root batch.

## Tile interactions in opponent beliefs

Initial count and structural-wait predictions feed 34 tile tokens per opponent.
Each token includes tile identity, ordinary/red count probabilities, conditional
ukeire probabilities, public availability and meld counts, shanten probabilities
and board context. Two attention blocks share weights across opponents. Each
block uses four attention heads at width 64, pre-normalization and a residual
128-wide GELU feed-forward layer. Every tile can attend to every other tile,
including across suits; there are no fixed suji connections or recurrent state.

A zero-initialized residual output refines ordinary/red count logits and all
four shanten-conditioned ukeire slices. The normal runtime compiles tile
attention as one XLA function, alongside the existing hand and river blocks.
Shanten logits retain their existing head. Existing count masks, projection and constraint supervision apply after
refinement. Losses, labels and exported tensor shapes are unchanged. This
learns interactions in marginal predictions; it does not define a joint
probability distribution over hands or wait sets.

The new architecture adds weights. Old checkpoints require an explicit weight
transfer before use; normal loading remains strict. Frozen training runs and
released serving models do not acquire these layers automatically.

## Training

The [candidate objective](../training/OBJECTIVE.md) defines actor weighting,
policy anchors and gradient boundaries. The critic predicts current-hand
outcomes projected through the fixed continuation model. There is no separate
state-value head.

Recorded and generated decisions directly supervise all three belief heads.
Hidden-state labels stay outside inference inputs. The same constrained
beliefs feed the policy and Q. Directional training penalties encourage higher
opponent tenpai to increase that opponent's win probability and conditional
exhaustive-draw readiness, and higher wait probability on the candidate tile
to increase that opponent's ron against the actor, with other inputs fixed.
These are soft penalties, not global monotonicity guarantees.

One settlement head handles immediate and later endings. Unplayed terminal
targets assume every legal opponent ron is claimed; see
the [target assumptions and phase coverage](../training/OBJECTIVE.md). Under
that approximation, actions receive settlement supervision without rollouts:
ron takes priority over aborts, including triple-ron abort; exhaustive draws
use known hidden readiness when visible discards rule out nagashi. Unknown
winning amounts are marginalized.
Automatic aborts without a policy action have no Q slot.

## Where to change things

| Area | Source |
| --- | --- |
| Public features and packed layout | [native collection](../log_dataset/native_collection/README.md) |
| Joint model assembly | [network.py](network.py) |
| Board, hand, river and action representations | [encoder.py](encoder.py) |
| Opponent beliefs, count constraints and supervision | [beliefs.py](beliefs.py) |
| Policy and candidate auxiliary heads, priors and losses | [actor.py](actor.py) |
| Legal actions, model indexing and native selection | [actions.py](actions.py) |
| Payment critic and action forecasts | [critic.py](critic.py) |
| Score support and settlement | [payment_scoring.py](payment_scoring.py), [joint_settlement.py](joint_settlement.py) |
| Component losses and gradient boundaries | [actor.py](actor.py), [beliefs.py](beliefs.py), [critic.py](critic.py) |
| Shared loss reductions and optimizer updates | [losses.py](losses.py), [trainer.py](../training/trainer.py) |
| Self-play and candidate updates | [self_play.py](../training/self_play.py), [trainer.py](../training/trainer.py) |
| Frozen sources and isolated opponents | [frozen_policy.py](../training/frozen_policy.py) |
| Recorded labels and sampling | [dataset guide](../log_dataset/README.md) |

## Constraints and compatibility

See [constraints](CONSTRAINTS.md) for masks, raw-head supervision, call priority
and schema compatibility.
