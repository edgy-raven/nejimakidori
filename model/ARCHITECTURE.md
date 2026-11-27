# Model architecture

Nejimakidori predicts legal actions, opponent beliefs and current-hand
settlements for four-player South games. A shared encoder combines the known
hand with public table state. The actor chooses an action; the critic predicts
its consequences. Hidden opponent tiles are training labels, never inputs.

This document describes the source architecture. Use the source bundled with
a released checkpoint for its exact input and output contract.

[Model guide](README.md) · [Constraints](CONSTRAINTS.md) ·
[Training objective](../training/OBJECTIVE.md) ·
[Interactive atlas](https://ogame.squareqrow.com/atlas/)

## Inputs and representations

The [native interface](../log_dataset/native_collection/README.md) defines
101 feature tensors for each decision state. Their main groups are:

| Group | Contents |
| --- | --- |
| Table | Scores, ranks, round wind, hands remaining, wall, dealer and deposits |
| Hand | Concealed tile counts, red fives, fixed melds and resulting-hand states |
| Opponents | Public melds, discards, riichi declarations and visible tile evidence |
| Candidates | Legal actions, resulting-hand references and deterministic scoring support |

Scores and ranks use East, South, West, North order. Opponent-specific tensors
use seats relative to the acting player. `seat_wind` connects these axes.
`hands_remaining` is 7 at East 1, 0 at South 4 and negative in extensions.

Tiles share one base-34 identity projection across hand, river, table and
candidate encoders. Red fives use the corresponding ordinary five plus a red
flag. Padding contributes no tile identity. Other categorical values use
one-hot coordinates or shared category embeddings; counts and magnitudes are
scaled separately from masks and probabilities.

Each state has a bank of up to 16 distinct hand states, including the current
hand and legal discard/kan results. Every hand state carries its own melds and
19 yaku-feasibility flags. These flags describe possible routes; they are not
predicted yaku probabilities. `candidate_fu` records the fixed-meld fu
contribution, not a completed hand's total fu.

## Shared encoder

Public river events pass through shared convolutions and attention across
opponents and riichi segments. Their summaries join table features to form a
shared context. That context conditions the hand encoder before suit-local
convolutions; honors have their own projection. The board encoder combines
only the current hand (bank slot zero) with table context through six residual
blocks. Candidate hands, action legality and candidate shanten/ukeire summaries
do not enter that table or board representation. Our current concealed tiles
remain evidence about the copies available to opponents.

Every legal candidate receives the representation of its resulting hand,
shared board context and action-specific evidence. Candidate attention joins
all legal alternatives for the same decision, including call-plus-discard
choices. Each candidate retains its own output; candidates from different
decisions cannot interact.
Exact candidate yaku distances, discard analysis, retained safe tiles, kan
efficiency and abort flags enter candidate scoring rather than board inference.

Candidate route readouts and the settlement critic consume these fused
representations. Immediate ron eligibility is predicted once per opponent and
base tile from the board, then indexed by the proposed cut. The policy reads
detached copies. Supervision on an observed candidate can therefore reach
alternative representations through attention,
without assigning unknown outcomes to those alternatives.

The complete model has **18,957,595 trainable parameters**, including
**3,489,776** in the settlement critic. The board encoder has six residual
blocks with 512-wide state and two 768-wide hidden layers per block. The
critic has six residual blocks with 256-wide state and 896-wide hidden
layers. This favors depth over widening or adding convolution stages.
Shared hand, river, board and initial candidate projections use BF16 computation with FP32
variables. The opponent tile-attention projections, attention and feedforward
blocks also compute in BF16, retaining FP32 normalization and belief readouts.
Predictions, probabilities and losses use FP32; constrained count
gradients use higher precision for numerical stability.

## Calls and action selection

A chi or pon and its follow-up discard form one candidate. Feature extraction
constructs the hypothetical accepted-call state, including changed melds and
kuikae restrictions. That state uses the same encoder and discard scorer as an
ordinary decision.

Hypothetical calls reuse the original decision's opponent shanten, improving
tiles, held-count distributions, structural waits and ron eligibility. A call
proposal adds no hidden-hand evidence. Its changed board and hand still
determine candidate representations and completion potential. Once a new
state is actually observed, ordinary inference runs on that observation.

The response policy first assigns probability to six branches: pass, three
chi patterns, pon and open kan. Each chi/pon branch is multiplied by its
conditional discard distribution. Pass and open kan have no immediate discard
continuation. Greedy selection chooses the branch by marginal probability
before choosing its discard.

The other policy heads cover ordinary discard, dama versus riichi with a
discard, concealed/added kan, and win versus decline. Red fives remain distinct
actions. Equivalent physical copies share a tile action. Automatic aborts
and forced choices are handled by the game rules.

Call settlement forecasts include competing claims that may preempt the
attempt. Accepted-call features describe the proposed result; guaranteed
public constraints remain those of the original call window.

## Supervised predictions

The model trains seven named prediction groups alongside the policy and
settlement objectives:

| Prediction | Meaning |
| --- | --- |
| Opponent shanten | Four classes: tenpai, one away, two away and three or more away |
| Opponent improving tiles | Structural improvement probability for each tile, conditional on shanten |
| Opponent tile counts | Marginal held-count distributions, constrained by known melds, hand sizes and remaining copies |
| Completion yaku | Yaku of the highest-scoring ordinary completion witness for the next observed tenpai shape, separately for ron and tsumo |
| Completion score | Our own total han/fu for that same completion witness; ron and tsumo |
| Candidate route affinity | Visible shortest-route support for tanyao, flush and outside-hand families |
| Candidate ron | Each opponent's legal ron probability on the proposed discard |

Opponent and completion predictions share a board representation. Tile
attention couples count and improving-tile predictions across suits. Opponent
yaku evidence includes our hand, public availability and detached
opponent count predictions. Completion-score logits have shape `[1, 2, 38]`:
only our own seat, for ron and tsumo. No opponent completion-score readout,
residual, supervised loss or actor/critic score input remains.

Structural wait probability is tenpai probability multiplied by conditional
wait probability. Candidate ron combines that structural wait with learned
board-level, tile-specific eligibility. Dama, riichi and alternative calls
offering the same tile share immediate danger under the same observed state;
red and ordinary fives share their base-tile danger. Known-safe tiles and
actions without a discard have zero ron
liability. Structural waits alone do not imply legal ron: furiten and yaku
requirements still matter.

Completion yaku and score targets refer to one shared witness. Scores include
known riichi, visible dora and menzen tsumo where applicable, but exclude
hypothetical ura and situational bonuses. Completion potential and route
support do not represent the probability of eventually winning.

This separation changes table, board-fusion, candidate and ron-readout shapes.
Earlier checkpoints require an explicit architecture transfer before training
with this source; their frozen source remains the matching evaluation path.

## Settlement and action value

The critic predicts how the current hand ends: ron claims, tsumo, exhaustive
draw or abort, with deposit states and conditional winning-hand values.
Rule-based settlement converts these predictions to transfers, including
dealer shares, honba, riichi deposits and pot priority. Multiple ron claims
are represented jointly; triple ron follows the simulator's abort rule.

Each claim predicts one distribution over total han/fu, including future
situational yaku and hidden ura. Fixed-meld fu, visible bonus han, open/closed
status and required riichi enter the score network directly; player and
candidate context supplies yaku evidence. Public minima and legal score
support constrain the distribution. Fixed-meld fu is not completed-hand fu,
and possible yaku are not guaranteed han. There is no separate added-han
prediction or bonus-conditioned score axis. This differs from the ordinary
completion prediction above. Claim amounts are conditionally factorized
given their shared context; all tsumo payer shares use the same score law.
Observed settlement likelihood trains the total score prediction without
separate han/fu supervision.

Observed payments supervise the action actually played. Unknown amounts can
be marginalized; unsupported responsibility or nagashi allocations are
excluded from likelihood and counted in coverage metrics. Unplayed actions
receive no observed settlement target.

Action value is derived from settlements through a fixed continuation model:

`J = expected ST3 / 100 + P(first) - P(fourth)`

Completed matches use exact scores and ranks. Earlier hands use scores,
remaining hands and scheduled dealerships to forecast final placement. The
continuation coefficients are fixed during training. There is no separate
state-value head or final-match-return regression.

## Gradient ownership

Every belief is detached wherever another prediction, the actor or critic
consumes it. Its own direct supervision still trains its readout and upstream
shared representations. Settlement supervision also trains the shared encoder
and candidate attention. Policy losses train policy-specific layers through
detached shared representations; Q is detached in policy objectives.

All seven prediction groups train on eligible decisions with known labels.
Already-riichi and forced decisions are excluded; discretionary dama choices
remain eligible. See [constraints](CONSTRAINTS.md) and the
[training objective](../training/OBJECTIVE.md) for loss and phase coverage.

## SavedModel interface

Both signatures accept 204 named inputs: 101 base tensors, 101 sparse
accepted-call tensors, and their owner and option vectors. Base tensors have
`B` decision rows; accepted-call tensors have `C` rows, which may be zero.
Use [call_states.py](../log_dataset/call_states.py) to batch and select them.

The `policy` signature returns the five policy heads. `serving_default`
returns the following tensors. Shapes omit the leading decision axis unless
marked as candidate outputs.

| Output | Shape |
| --- | --- |
| `discard_policy_logits` | `[37]` |
| `riichi_policy_logits` | `[74]` |
| `response_policy_logits` | `[150]` |
| `kan_action_logits` | `[35]` |
| `win_decision_logits` | `[2]` |
| `opponent_shanten_logits` | `[3, 4]` |
| `riichi_action_logits` | `[2]` |
| `response_action_logits` | `[4]` |
| `chi_pattern_logits` | `[3]` |
| `terminal_payment_probabilities` | `[4, 4, 4, 60]` |
| `payment_values` | `[60]` |
| `final_placement_probabilities` | `[4]` |
| `review_outcome_probabilities` | `[7]` |
| `settlement_mean`, `settlement_variance` | `[5]` each |
| `critic_expected_points` | `[4]` per active candidate |
| `critic_action_indices` | `[2]` per active candidate |

`critic_action_indices` identifies each candidate as `(decision, action)`.
Decision-level forecasts average over the active policy. Typed payments and
settlement moments exclude honba; current-hand action values use the complete
settlement rules. Other belief predictions are internal training and policy
inputs rather than serving outputs.
