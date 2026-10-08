# Nejimakidori model card

Nejimakidori learns four-player riichi decisions from recorded games. One
encoder feeds the policy, opponent beliefs and current-hand settlement critic.
The [training pipeline](model/README.md) produces a verified export and a
frozen-opponent match.

## Inputs and outputs

The visible input schema is [model/feature_vector.py](model/feature_vector.py),
packed by [model/features.py](model/features.py). It includes candidate hands,
legal actions, rivers, melds, dora, scores, seats, wall and public safety clues.
Opponent concealed tiles and future outcomes are training labels only.

Discard policies have 37 tile classes, with red fives distinct. Riichi policies
have 74 dama/riichi × tile choices. Drawn and held copies share one policy
action; execution preserves red identity and forced-discard legality. Calls,
kan and riichi are masked to native legal choices.

The full export returns policy probabilities, opponent beliefs, current-hand
payment and rank distributions, per-action projected ST3 and four final-placement
probabilities. Final placement uses analytic continuation of the settlement
forecast. It is not a separately learned rank head. See
[the model contract](model/PAYMENT_CRITIC.md) and
[HTTP API](service/README.md).

## Data and training

The [dataset job](log_dataset/README.md) selects ranked player-games, excludes
suspected AFK hands and reserves complete games for validation and holdout.
Approved RiichiLab teachers can supplement the Houou corpus.

Training uses a persistent mixture of 75% natural and 25% focused records.
Focus reads decision-time features: tenpai and riichi choices, legal kans,
value-qualified calls, late-wall fights, visible opponent threats and bounded
own-yaku development. It does not read the recorded choice or future yaku.
Canonical rules and category weights live in
[records.py](log_dataset/records.py) and [focus_sampling.py](log_dataset/focus_sampling.py).
Validation remains natural.

Joint objectives combine imitation or detached projected-ST3 AWR, settlement
likelihood, hand-end rank likelihood, immediate legal-ron supervision, opponent
belief losses and completion-yaku supervision. One optimizer updates the shared
encoder and heads. [PAYMENT_CRITIC.md](model/PAYMENT_CRITIC.md) records the
loss, scoring and continuation contracts.

## Limitations

Public safety features are evidence, not guarantees of safety. AFK filtering
is a behavioral heuristic. Settlement support and continuation approximate
future play; a normalized placement forecast is not proof of calibration.
Smoke tests verify software behavior. Strength requires matched game results
against the intended opponents.
