# Recorded-game datasets

The [data and model integration contract](INTEGRATION_CONTRACT.md) records the
current end-to-end test boundary.

Download the fixed 2024/2025 Houou MJAI archives from the repository root:

```sh
python -m log_dataset.download --output /path/to/archives
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

Training runs freeze this package beside their model source so workers use
the exact data filter recorded in the run plan.

## Historical player ranks

Before building training records, merge ranks once from the original Tenhou
SQLite databases (`2024.db`, `2025.db`, release `v1.2.0` of
[NikkeTryHard/tenhou-to-mjai](https://github.com/NikkeTryHard/tenhou-to-mjai/releases/tag/v1.2.0)):

```sh
python -m log_dataset.player_ranks --archives /path/to/archives \
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
python -m log_dataset.rebuild --archives /path/to/archives --output /path/to/data
```

The rebuild writes `decisions.sqlite` and materializes training TFRecords from
it. The database is keyed by game, hand, decision and actor. Its `records`
table stores serialized examples; `positions` and `discard_actions` store
scalar shanten, ukeire, upgrade and scored-tenpai summaries for joins.
`log_dataset.shanten` owns the vendored Nyanten calculation used by both the
index and model feature producer.

`log_dataset.rebuild` is the one corpus rebuild job: it reads the archives,
populates the database, and materializes its initial TFRecords. Rank enrichment
remains the separate `log_dataset.player_ranks` prerequisite. To rematerialize
TFRecords after a record-layout change without rereading archives, run:

```sh
python -m log_dataset.materialize --database /path/to/data/decisions.sqlite \
  --output /path/to/new-data
```

The record builder consumes the saved seats for all three natural → focused → natural
training phases, then applies AFK exclusions and the focused-condition union. Rank is
sampling metadata, never a model input. It neither fetches ranks nor resamples.
The merge refuses existing outputs and already enriched inputs. The archive
downloader verifies installed enriched archives using the report's original
and merged checksums, and does not download them again. Rebuild SQLite when
archive-derived analysis or labels change. Rematerialize TFRecords when only
the record layout changes; modifying summaries changes neither.

Forced-only discard positions produce no TFRecord, including no board or
auxiliary supervision. Their events remain in the MJAI archive for state and
outcome reconstruction. Automatic win/kyuushu opportunities likewise produce
no training rows; their terminal events still label earlier genuine choices.
