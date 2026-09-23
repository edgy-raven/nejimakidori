# Suspected AFK kyokus

The dataset builder excludes a kyoku when either AFK rule qualifies:

- Two consecutive complete hands from one seat with at least eight discards
  each, all tsumogiri, and no tedashi, riichi declaration, call, kan or win.
  Every hand in the streak is removed, including its first hand. A shorter
  hand or active choice resets that seat's streak.
- A run of at least 12 unforced tsumogiris within one hand, ignoring at least
  two shanten-improving draws, while the held hand is not tenpai and no
  opponent has declared riichi during the run. This can start after earlier
  tedashi or calls and does not require another inactive hand.

For the second rule, an active choice ends that seat's run. Earlier active
play does not exempt the rest of the hand. Other players' calls do not reset
the run. A riichi declaration ends the declaring player's run and suppresses
all their forced post-riichi discards. Separate runs and multiple inactive
seats are detected; each kyoku is excluded once.

The detector replays each hand; only long-run candidates need shanten checks.
The held hand after the first tsumogiri remains fixed throughout the run.
Improving draws are actual discarded tiles in that hand's structural ukeire,
including red/ordinary five equivalence. The rule does not count unchanged
shanten upgrades or use hypothetical draws. Hidden tiles are used only for
offline data quality; these annotations are not policy inputs.

All four players' decisions from an excluded kyoku are removed before feature
generation. Other kyokus in the same hanchan are retained in its original
training corpus. A hanchan disappears only if no kyokus remain.
Retained kyokus use their actual starting state, payments and placement labels
from the original replay, including the original final match result and
first-dealer tie-breaking. Match results can still reflect AFK in other
kyokus; filtering does not invent a counterfactual result or mask those labels.

The sampled Houou MJAI archives contain no disconnect or timeout fields.
This is behavioral evidence of suspected AFK, not a confirmed disconnect.
Riichi and tenpai exclusions prevent obvious forced-play/dama false positives
in the new rule. A player folding against an open hand or repeatedly choosing
a different yaku route can still be a false positive. Brief AFK, late
departures, and runs during opponent riichi may be missed. The thresholds are
conservative heuristics, not a calibrated classifier. The whole-hand rule
retains its separate behavior. The filter does not identify or ban accounts.

The detector returns hand indices and seat/discard evidence. The expert record builder uses `log_dataset.afk.retained_hands` before generating
any examples. Dataset summaries declare `afk_filter.scope = "kyoku"`.
Existing datasets are unchanged; rebuild from logs to apply the filter.
