# Recorded-game datasets

Current training actions are tile classes: 37 discards and 74 dama/riichi
choices. Karagiri and tsumogiri are the same action. Recorded origin is a
separate `discard_origin` auxiliary label and still controls tedashi-only
focus for broader yaku/round contexts. Riichi threats, high-value opponents,
and the last 16 live-wall tiles retain either discard origin, as do existing
critical-position overrides. Rebuild cached labels, selection, and TFRecords for this
contract; old checkpoints/exports require their frozen source. See
[payment critic contract](../model/PAYMENT_CRITIC.md).

The [data and model integration contract](INTEGRATION_CONTRACT.md) records the
current end-to-end test boundary.

Download the fixed 2024/2025 Houou MJAI archives from the repository root:

```sh
python download_dataset.py --output /path/to/archives
```

Downloads resume from `.zip.part` files and verify the pinned SHA256 and MJAI
archive contents before renaming into place. Existing archives are verified
without downloading them again. No separate download cache is needed.

`afk.retained_hands(events)` removes **all four players' decisions** from a
suspected AFK hand, including decisions before inactivity began. It retains
other hands in the match. The expert record builder uses this filter.
See [AFK evidence and thresholds](AFK.md). Features and labels remain owned by
`model`; `log_dataset.shanten` owns the native structural analysis shared by
the filter and feature producer.
`log_dataset.scoring` owns completion-yaku evidence, its ordered yaku names,
and decision-time scored waits. Corpus sampling, training labels, visible
features, and inference metadata share this scoring code.

Training runs freeze this package beside their model source so workers use
the exact data filter recorded in the run plan.

## Historical player ranks

Before building training records, merge ranks once from the original Tenhou
SQLite databases (`2024.db`, `2025.db`, release `v1.2.0` of
[NikkeTryHard/tenhou-to-mjai](https://github.com/NikkeTryHard/tenhou-to-mjai/releases/tag/v1.2.0)):

```sh
python enrich_ranks.py --archives /path/to/archives \
  --databases /path/to/original-databases --output /path/to/merged
```

The command stages enriched archives in a fresh output directory. Verify its
`player_ranks.json` coverage report, then install the two archives and that
report together in the corpus directory, retaining the originals as backups.
The merge checks game IDs and all four ordered player names. Each MJAI
`start_game.player_ranks` contains historical `dan`, `rating`, and
`retained_seats`. Dan 11 denotes Tenhoui; missing metadata uses empty lists.
Game events and all opponents remain intact.

Selection keeps every 9+ dan player-game, 75% of 8-dan player-games and 50%
of 7-dan player-games (rounded down within each year/rank stratum). A fixed
SHA256 ordering selects whole seats for the entire game. Unverified games
and players below 7 dan contribute no training rows. The coverage report
includes verified games, retained games, and rank counts before/after selection.

Build records from the installed enriched archives:

```sh
python build_dataset.py --archives /path/to/archives --output /path/to/data
```

The rebuild writes a normalized preprocessing cache, `decisions.sqlite`, then
exports training TFRecords. [Cache stages and indexing](CACHE.md) describe the
independent import, replay, structural analysis, scoring, yaku, participation,
label and selection commands. Each stage can be rerun from persisted inputs;
changed prerequisites invalidate dependent results.

To export a completed cache without replay or feature calculations:

```sh
python export_dataset.py --database /path/to/data/decisions.sqlite \
  --output /path/to/new-data --workers 8
```

The cache contains component tensors, not finished training examples. Tile
participation and decision-time eligible yaku are cached separately from sampling
rules and retrospective whole-hand yaku evidence. Critic calculations remain on
the existing on-the-fly path.

The record builder consumes the saved seats for all three natural → focused → natural
training phases, then applies AFK exclusions and the focused-condition union. Rank is
sampling metadata, never a model input. It neither fetches ranks nor resamples.
The merge refuses existing outputs and already enriched inputs. The archive
downloader verifies installed enriched archives using the report's original
and merged checksums, and does not download them again. Rerun the affected cache stages and their dependents when analysis or labels
change. Re-export when only TFRecord packaging changes; modifying summaries
changes neither.

Forced-only discard positions produce no TFRecord, including no board or
auxiliary supervision. Their events remain in the MJAI archive for state and
outcome reconstruction. Automatic win/kyuushu opportunities likewise produce
no training rows; their terminal events still label earlier genuine choices.

## Approved RiichiLab teachers

Import completed ranked games involving our accounts and an explicit teacher
allowlist. Only teacher seats receive assigned training ranks; all other
seats have unknown ranks and are excluded from imitation. Complete game events
remain available for features and outcome labels. Games with any action-penalized
player are excluded before teacher sampling, even when another player is the
teacher. Original logs and metadata remain preserved; the manifest lists
`action_penalty_excluded_games`. Disconnected teacher seats are excluded.
The supplied dan is an assignment, not a claimed
historical Tenhou rank. The usual deterministic player-game sampling applies
(75% for 8d), with no additional teacher weighting.

```sh
python import_riichilab.py --own-bots 407 408 \
  --teachers 120:8 299:8 351:8 203:8 \
  --output /mnt/drv1/nejimakidori/corpora/riichilab-teachers-20261001-four
```

Each import creates a fresh snapshot containing original compressed logs,
public game metadata, an enriched ZIP, and its checksummed rank manifest.
Matches involving both our bots are deduplicated. Future teachers require
explicit `BOT_ID:DAN` entries; opponents are never enrolled automatically.
Refresh into a new snapshot directory, replacing the previous supplemental
input rather than combining overlapping snapshots.

The current penalty-filtered snapshot is
`/mnt/drv1/nejimakidori/corpora/riichilab-teachers-20261002-four-no-penalties/`.
It retains 153 of the 154 downloaded games; one game with an action penalty
is excluded for every seat before sampling. All four approved teachers remain.
Use this supplement for the next combined build. The unfiltered October 2
snapshot and October 1 snapshot below are historical inputs, not additional
supplements to concatenate with the current one.

The approved teachers are Mortal-v4b (120), nodoka-latest (299),
new_1 (351), and みーにょ6段 (203), all assigned 8d. To combine the current snapshot with the
main corpus on a subsequent build, use:

```sh
python build_critic_dataset.py \
  --archives /mnt/drv1/nejimakidori/corpora/tenhou-houou-mjai \
  --extra-archives /mnt/drv1/nejimakidori/corpora/riichilab-teachers-20261002-four-no-penalties \
  --reserve /path/to/reserve.json --output /path/to/new-training-data
```

`--extra-archives` is repeatable and rejects duplicate game IDs. The normal
whole-game train/validation/held-out partition remains unchanged. Existing
frozen runs and datasets are not modified by importing a supplemental corpus.

The matching materialized snapshot is
`/mnt/drv1/nejimakidori/datasets/riichilab-teachers-20261002-four-no-penalties/`.
Its `teacher-validation.json` audits every record against the archive's
retained teacher seats. This snapshot backfills only games involving accounts
407 or 408; it does not include teachers' games against unrelated tables.

## Training contract rebuild

The current record contract adds training-only attack/defense allocation,
includes all five yakuhai types in retrospective own-yaku eligibility, and gives
never-tenpai sakigiri labels the constant 14,689. Rebuild records, focused
metadata, and materialized datasets together. Existing summaries are rejected;
retain historical corpora separately. Use the matching input schema recorded
in the dataset summary.
See [the joint training contract](../model/PAYMENT_CRITIC.md).

Focused calling retains a legal chi/pon-to-tenpai opportunity when its
live-wait-weighted open base ron is strictly above 3,900, or its valid
closed-riichi counterfactual base ron is strictly below 7,700. The latter
requires a currently closed hand and a positive scored value; an unavailable
counterfactual encoded as zero does not qualify. Retain both recorded calls
and passes at qualifying opportunities. This replaces blanket actual-call
retention and unrestricted call-to-tenpai retention. Other critical clauses,
including own tenpai, riichi/dama, late keiten and actual own kan, still apply.
The thresholds exclude exactly 3,900 from the first branch and exactly 7,700
from the second; either branch may qualify the same position.

For an existing normalized cache, rerun selection and materialization to update
hand-focus and focused metadata. Cached per-yaku evidence already includes
yakuhai. Existing frozen materialized corpora retain their historical selection
contract; do not relabel their summaries as the new contract.
