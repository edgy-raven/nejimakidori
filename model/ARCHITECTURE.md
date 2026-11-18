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
likely those finishes are. Dama uses undeclared scoring; declaration and
already-accepted riichi use riichi scoring. The action-specific ron-valid
flag is false for no-yaku dama. See the [native input contract](../log_dataset/native_collection/README.md#action-conditioned-win-values)
for dataset regeneration requirements. Fixed-magnet scoring tensors are kept
outside learned projections; training uses them only to preserve the released
reference policy's decisions.

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

The critic applies shared player and claim projections rather than separate
weights for each opponent/payment slot. Player descriptors combine public
evidence with mean/max summaries of the shared river convolutions, before
stream-specific attention. Wind descriptors expose dealer/double-wind status,
known unavailable wind copies and dora multiplicity rather than wind identity.
Scores and placement priority remain explicit public inputs. Our player token
is distinguished because its hand and candidate action are known.
Public meld tile counts and red counts retain observed meld identities. These
are pre-action facts; the common candidate context supplies our proposed hand.
Per-action descriptors add projected ordinary draws, eligibility by turn to
chi the active discard, and the discard source. Draw projections share the
actor's pass/call/kan continuation calculation and exclude rinshan. Chi turn
eligibility is not a claim that the hidden hand can make the call.

Every player receives the same common candidate context. Existing belief and
completion predictions stay in that common context; separate learned opponent
predictions are not added to the player-specific descriptors. Equal descriptors
and outcome roles therefore produce equal scores at fixed common context.
The unchanged common encoder still depends on seat ordering: this is not a
claim of end-to-end invariance under arbitrary raw-input player permutations.

Shared role pooling scores whole win and draw masks, retaining all49 event
choices and16 conditional deposit masks. Nonlinear event and deposit scoring
retain interactions rather than multiplying independent player probabilities.
The existing factorization of simultaneous winners' hand-value likelihoods is
unchanged. Sharing parameters does not remove that approximation.

Changing these heads requires an explicit fitted checkpoint migration. The
actor, belief heads, common critic projection and shared hand-value decoder
can retain their weights; slot-specific event, deposit and claim projections
cannot be copied into the new shared layers with exact prediction parity.

[payment_scoring.py](payment_scoring.py) defines the score vocabulary and
legality masks. [joint_settlement.py](joint_settlement.py) combines events,
scores and deposits. Unsupported payments, including some responsibility and
nagashi allocations, are excluded from settlement likelihood and counted in
coverage metrics.

Training values actions by projected Saint 3 (ST3) match points. Completed
matches use exact final scores and ranks. Earlier hands use a fixed continuation
model based on scores, remaining hands and scheduled dealerships. Its parameters
are separate from the learned current-hand critic.

The frozen four-parameter single-Gumbel continuation was recalibrated on
November 17 using 100,000 completed games: seven duplicate training matches
and three standard South Mortal matches. Its variance is a positive
`a + c*T + d*T²` curve in scheduled hands remaining, with positive dealership
drift. This is a pooled-match calibration, with no opponent-strength or
rules-format input. The additional data do not establish a held-out improvement
over the preceding 80,000-game calibration. Analytic mixtures remain experiments.

Root scoring computes settlement payoffs once per public position and reuses
them across candidate actions. Only the settlement probabilities depend on the
action. Payoff construction and action forecasts use bounded GPU chunks; the
shared payoff tensor exists only for the current root batch.

## Shared river self-attention

After the per-stream convolutions, one residual self-attention block connects
all 192 observed river events across opponents and pre/post-riichi streams.
It uses width 64, four heads, pre-normalization, and learned stream and position
embeddings. Invalid keys are masked, invalid query outputs remain zero, and
a zero null key handles empty rivers. Its bias-free output projection starts
at zero. The refined tokens feed both the pooled board representation (and
therefore opponent beliefs) and candidate-to-river attention, reaching Q
through both paths. No future events or hidden hands are attention inputs.

This block is implemented but has not been strength- or Q-quality-tested.

## Candidate attention to rivers

Each legal candidate queries the river encoder's 192 event tokens: three
opponents, two pre/post-riichi streams and 32 positions per stream. Tokens
retain the existing convolutional event features, with learned stream and
position embeddings. Four attention heads of width 16 produce a residual
correction to the 384-wide candidate representation shared by policy and
critic. The pooled river representation still feeds the board encoder.

Queries are packed by board using only legal candidates; river keys are shared
across those candidates. Invalid events are masked. A zero-valued null token
allows empty rivers to produce exactly zero correction, including after
training. The output projection starts at zero, preserving existing predictions
when weights are explicitly transferred. Training must learn the new branch
before it can change decisions. Input features and exported outputs are unchanged.

To add missing river attention branches to otherwise matching weights, run:

```sh
CUDA_VISIBLE_DEVICES=-1 python -m model.initialize \
    --weights previous.weights.h5 --output river.weights.h5
```

The converter rejects unrelated architecture differences and existing output
files. Normal checkpoint loading remains strict. This command does not start
training or alter frozen runs or serving models.

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
