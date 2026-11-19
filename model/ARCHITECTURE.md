# Model architecture

The current source implements a shared public-state encoder, a factorized
legal-action policy, opponent belief marginals and an action-conditioned
current-hand payment distribution. Four residual board blocks connect the
encoder to the policy, beliefs and critic. [network.py](network.py) assembles
the graph; the [interactive catalog](https://ogame.squareqrow.com/nejimakidori/catalog/) indexes its blocks,
operators, equations and phase-specific gradient paths.

This document describes the source contract. Released checkpoints retain the
architecture and input/output signatures of their bundled source.

For commands, see [training and evaluation](../training/README.md). The
[native interface](../log_dataset/native_collection/README.md) defines tensor
encodings and packed transport.

The network input contract contains 124 native tensors. Candidate-hand indices
serve gathers only; they do not enter learned board projections. Dora indicator
IDs and their mask are absent; `dora_multiplicity` supplies tile-value evidence.
The board projection kernel is `[4630, 512]`. The four action-scoring tensors
remain candidate-local and bypass global board and hand-bank conditioning.
The constant `river_post_riichi` field is omitted from transport; the shared
river encoder reconstructs its zero channel. Hand-bank board conditioning has
shape `[10377, 128]`.

Four board residual blocks use `512 → 1024 → 1024 → 512`, with GELU after each
hidden transform. Separate belief `512 → 512 → 512` and critic
`128 → 512 → 128` residual refinements precede their readouts. Both refinements
use pre-normalization and a zero-initialized final projection, so each starts
as an identity map. The board-block redesign is not identity-preserving.
Relative to the preceding architecture, pruning the global scoring copies and
constant river input while adding this capacity reduces the parameter count
by 86,784. This budget accounting does not establish a quality improvement. Earlier checkpoints require
explicit weight transfer and compatible exports. Removing learned input rows
is not an output-preserving conversion. No runtime schema adapter is provided.

## Tensor signatures

Both SavedModel signatures accept the same 124 named input tensors. The
`policy` signature returns the five policy heads; `serving_default` returns
17 tensors. Shapes below omit the leading row axis. Most rows are public
positions; the last two outputs instead have one row per active candidate.
`critic_action_indices` maps each candidate row to `(position, action)`.

| Output | Shape |
| --- | --- |
| `discard_policy_logits` | `[37]` |
| `riichi_policy_logits` | `[74]` |
| `response_policy_logits` | `[150]` |
| `kan_action_logits` | `[35]` |
| `win_decision_logits` | `[2]` |
| `opponent_shanten_logits` | `[3, 4]` |
| `riichi_action_logits`, `response_action_logits`, `chi_pattern_logits` | `[2]`, `[4]`, `[3]` |
| `terminal_payment_probabilities` | `[4, 4, 4, 60]` |
| `payment_values` | `[60]` |
| `final_placement_probabilities` | `[4]` |
| `review_outcome_probabilities` | `[7]` |
| `settlement_mean`, `settlement_variance` | `[5]` each |
| `critic_expected_points` | `[4]` per active candidate |
| `critic_action_indices` | `[2]` per active candidate |

Count, ukeire and completion predictions remain internal network features and
restoration targets. [actions.py](actions.py), [export.py](export.py) and
[inference.py](inference.py) define the exact output and loading contracts.
Frozen exports keep their original signatures; regenerate an export before
loading it with the current strict inference API. Output pruning changes
neither learned weights nor continuation values.

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
are learned choices; nine-terminals aborts remain automatic. Restoration
supervises win/pass from recorded choices. MMD currently shares one forecast
between accepting and declining, so Q supplies no accept/decline gradient.

[actor.py](actor.py) applies opening symmetry and final-wall safety rules.
Its other heuristic prior outputs support supervised restoration.
The [MMD objective](../training/OBJECTIVE.md#magnetic-mirror-descent) uses
detached action values and two policy anchors without positive-action priors.
Supervised restoration helpers remain separate from the improvement loop.

The final candidate scorer combines candidate representations, route affinity,
ron predictions and completion context. It has no learned `sakigiri_order`
scalar. Removing that branch also changes the scorer's input width: checkpoints
with the former branch are incompatible with strict current-model loading.
No live checkpoint conversion or runtime fallback is provided. Frozen exports
retain their original graph. Sakigiri witnesses remain a root-mining criterion,
independent of the network architecture.

## What the predictions mean

| Prediction | Meaning |
| --- | --- |
| Opponent beliefs | Shanten, structural waits and held tile-count marginals, including known melds. Concealed counts respect hand sizes and available copies; riichi and four fixed melds fix shanten to tenpai. |
| Completion yaku and score | The highest-scoring legal completion of the next observed tenpai shape, separately for ron and tsumo. Unknown targets are masked. |
| Route affinity | How much of a candidate's visible shortest-route support belongs to tanyao, flush or outside-hand families. |
| Hand settlement | A probability distribution over ways the hand can end and the resulting transfers. |
| Final placement | Settlement outcomes integrated through a fixed continuation model for the rest of the match. |

Structural waits can be blocked by furiten or lack of yaku. Completion potential
and route affinity are distinct from win probability.
The candidate value-range inputs are also conditional completion bounds: they
search available finishes from zero/one-shanten hands, without predicting how
likely those finishes are. Dama uses undeclared scoring; declaration and
already-accepted riichi use riichi scoring. The action-specific ron-valid
flag is false for no-yaku dama. See the [native input contract](../log_dataset/native_collection/README.md#action-conditioned-win-values)
for dataset regeneration requirements. Fixed-reference scoring tensors are
absent from the current input contract.

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
evidence with mean/max summaries of the shared river convolutions. Wind
descriptors expose dealer/double-wind status,
known unavailable wind copies and dora multiplicity rather than wind identity.
Scores and placement priority remain explicit public inputs. The actor token
contains a self indicator because its hand and candidate action are known.
All four seats use the same player projection weights. Dealer status overlaps
with self/opponent identity: any of the four seats can be dealer; there is no
separate dealer or self-only projection.
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
The common encoder depends on seat ordering, so the full network is not
invariant under arbitrary raw-input player permutations.

Shared role pooling scores whole win and draw masks, retaining all 49 event
choices and 16 conditional deposit masks. Nonlinear event and deposit scoring
retain interactions rather than multiplying independent player probabilities.
Simultaneous winners' hand-value likelihoods remain factorized conditional on
the shared context. Parameter sharing does not remove that approximation.

Changing these heads requires an explicit fitted checkpoint migration. The
actor, belief heads, common critic projection and shared hand-value decoder
can retain their weights; slot-specific event, deposit and claim projections
cannot be copied into the new shared layers with exact prediction parity.

[payment_scoring.py](payment_scoring.py) defines the score vocabulary and
legality masks. [joint_settlement.py](joint_settlement.py) combines events,
scores and deposits. Unsupported payments, including some responsibility and
nagashi allocations, are excluded from settlement likelihood and counted in
coverage metrics.

Training values actions by `ST3/100 + P(first) - P(fourth)`. Raw Saint 3 (ST3)
match points remain a separate evaluation and diagnostic output. Completed
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

Actor and root scoring compute settlement payoffs once per public position
and reuse them across candidate actions. Only the settlement probabilities depend on the
action. Payoff construction and action forecasts use bounded GPU chunks; the
shared payoff tensor exists only for the current batch.

## Shared river encoding

Opponent rivers use shared tile embeddings and three masked convolutional
blocks, separately for pre- and post-riichi streams. Each stream contributes
its ordered token features and maximum to the board projection. Shared mean
and maximum summaries also feed the critic's per-player context. Padding is
masked after every convolution; empty streams remain zero.

Candidate representations combine the hand and board encodings directly.
There is no river self-attention or candidate-to-river attention. Tile
attention in opponent beliefs and the shared player/event critic remain.

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
Shanten logits use a separate head. Count masks, projection and constraint
supervision apply after refinement. Attention couples marginal predictions;
it does not define a joint distribution over concealed hands or wait sets.

Checkpoints without tile-attention weights require explicit transfer and
fitting before use. Loading remains strict; frozen exports retain their own
architecture.

## Training

The [MMD objective](../training/OBJECTIVE.md) defines action values,
policy anchors and gradient boundaries. The critic predicts current-hand
outcomes projected through the fixed continuation model. There is no separate
state-value head.

Restoration can directly supervise all three belief heads. MMD uses indirect
belief gradients and structural constraints. Hidden-state labels stay outside
inference inputs. The same constrained beliefs feed the policy and Q. Directional training penalties encourage higher
opponent tenpai to increase that opponent's win probability and conditional
exhaustive-draw readiness, and higher wait probability on the candidate tile
to increase that opponent's ron against the actor, with other inputs fixed.
These are soft penalties, not global monotonicity guarantees.

One settlement head handles immediate and later endings. The critic learns
from observed current-hand payments for recorded or rerun actions. Unplayed
actions receive no settlement target; MMD evaluates them with detached critic
forecasts. See the [phase coverage](../training/OBJECTIVE.md).
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
| Self-play and MMD updates | [self_play.py](../training/self_play.py), [corrections/](../training/corrections/README.md) |
| Frozen sources and isolated opponents | [frozen_policy.py](../training/frozen_policy.py) |
| Recorded labels and sampling | [dataset guide](../log_dataset/README.md) |

## Constraints and compatibility

See [constraints](CONSTRAINTS.md) for masks, raw-head supervision, call priority
and schema compatibility.
