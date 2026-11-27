# Record contract

[Build a dataset](README.md)

## Training positions and sampling

Reserved evaluation games are excluded before deterministic train, validation
and held-out partitioning. Historical ranks select all 9+ dan seats, 75% of
8d seats and 50% of 7d seats. Other players still supply game context.

Forced moves supply replay context only. A riichi decision selects both the
declaration and its discard. Staying dama also supplies an ordinary discard
row when multiple distinct cuts are legal. Declaration discards and subsequent
locked discards supply context only; optional wins and kans remain decisions.
Replay reconstructs missed-win furiten before exposing ron choices. A passed
ron after riichi cannot create later ron-pass labels in that hand. A legal
ron declined on the last discard remains a win-pass row before an exhaustive
draw. Accepted calls censor other seats' losing claims, which are not labeled
as passes.

Expert imitation draws a corpus-sized budget per pass using 50% natural and
50% focused examples. The raw reader remains natural; only the training loader
applies emphasis. Integer `meta/focused_priority` supplies the focused stream
and never enters model inputs. Stored placement-gap metadata from existing
corpora is inert: no loader selects, weights or requires it. There is no
placement-based pass or quota. Corpus counts retain their recorded metadata;
raw inputs need not be rewritten merely to stop consuming a field.

Initializer Monte Carlo pairs focused roots with natural-source roots, and
main-loop collection pairs targeted roots with unfiltered normal roots, both
50/50 by root starts. All generated eligible descendants train normally.

Validation uses a deterministic 1/16 decision sample from the natural
validation partition, stored separately to avoid scanning training shards.
[afk.py](afk.py) excludes hands with suspected inactivity: consecutive long
all-tsumogiri hands, or long unforced tsumogiri runs ignoring improving draws
outside tenpai and opponent riichi. These are behavioral heuristics; archives
have no timeout signal. A flagged hand is removed for every seat while other
hands retain their context.

## Features and labels

The `root-states/` sidecar stores complete native simulator states for retained
training decisions, keyed by game, hand, decision index, actor and phase. These
states contain hidden information for simulation and never enter model inputs.
They preserve actual recorded opponent hands; wall randomization happens when
forking. Saved sakigiri witnesses select roots only. Runtime initialization
requires the completed sidecar to match the dataset summary and input schema.

The [native engine](native_collection/README.md) extracts public features.
[replay.py](replay.py) reconstructs decisions; [training_targets.py](training_targets.py)
and [scoring.py](scoring.py) produce labels. Hidden hands and later outcomes
are available to target generation, never to policy inputs.
[records.py](records.py) defines label names, shapes and serialization.

The native schema contains 101 tensors and 8,958 bytes per row. Scores and
ranks use current seat-wind order; other seat axes are actor-relative. Each
state has a 16-slot bank for its current hand, ordinary discards and kans.
The native contract specifies tile sentinels and the 110-row scoring map.

Every record requires `feature/post_call`, a variable-length byte string.
Each legal chi/pon entry contains one option byte (chi 0–2, pon 3) followed by
a complete row after the call, totaling 8,959 bytes. Other decisions store an
empty string. Calls accompanying a ron decision remain available after a
declined win. Open kan has no immediate discard state. Decode after local
row selection, concatenate the entries and construct their owner indices.

`reference/dora_tile_id` and `reference/dora_valid` preserve five ordered
indicator slots for a frozen reference policy, outside the learned feature
schema. IDs are base-34 marker identities with zero-filled unused slots.
The decoder exposes them as `labels.reference_inputs`. Indicator order cannot
be recovered from dora multiplicities; `reference_schema` records this
sideband separately.

Belief labels cover counterfactual ron, local yaku-route affinity, opponent
concealed counts, shanten, ukeire, completion yaku and completion han/fu.
`candidate_yaku_affinity[16,3]` labels the local hand bank. Joint chi/pon
candidates are masked from parent-bank route supervision because their hands
belong to accepted-call states.

`round_completion_yaku` and `round_completion_score` use a highest-point
ordinary ron/tsumo witness for the next observed tenpai shape. Completion
score includes actual declared riichi, visible dora and applicable menzen
tsumo, without hypothetical ura, ippatsu or special-event bonuses. Recorded
score labels retain their four-seat evidence layout; the current completion
score head and loss consume only actor-relative seat zero. Yaku supervision
still covers all four seats. No record regeneration is required for this
output-head removal.

[training_data.py](training_data.py) also stores `critic/bonus` and
`critic/score`: categorical added-han labels and total-han/fu classes
conditional on added han, retained as recorded scoring evidence. The current
critic does not consume these sidebands: observed terminal payments supervise
its single total han/fu distribution through settlement likelihood. Recorded
future scoring values are never encoder inputs.

Action labels include `continuation_discard`. Metadata identifies the game,
hand, decision, actor and partition; sampling sidebands remain separate.
[sakigiri_regret.py](sakigiri_regret.py) supplies root-selection witnesses,
not restoration targets. `records.parse_batch` returns inputs and labels;
`training_data.parse_batch` additionally returns critic supervision.

Consumers validate schemas and scoring definitions against
`dataset_summary.json`. Rebuild into a new directory after a contract change;
editing metadata cannot convert existing records. See
[model architecture](../model/ARCHITECTURE.md) for prediction meanings.

## Prepared expert targets

Prepared records preserve the human action, same-hand outcomes and seven
belief labels. `expert/mortal_action` is a required scalar int16 byte field
containing Mortal's greedy choice in head-local action space. Decoding maps
it to the global critic candidate ID, including joint call/discard and open
kan actions.

The initializer performs three Q-weighted human imitation passes. Each stage
also trains observed settlement and belief losses. Settlement targets always
describe the human action actually played.
Expert weights compare the live learner's Q for the human action with its Q
for the recorded Mortal action; Q values are never cached.

The summary preserves `preparation.mortal_policy` provenance, source hashes,
partition counts and validation exclusions. Minibatch updates use prepared
choices without running Mortal. The main-loop MMD reference is separate.
