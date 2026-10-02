# Preprocessing cache

`decisions.sqlite` is a disposable cache before TFRecord generation. It stores
recorded data and reusable calculations. Critic predictions, policy evaluations,
and self-play targets are computed at runtime; the critic generator continues
to use its existing on-the-fly path.

Build a fresh cache and export its TFRecords:

```sh
python build_dataset.py --archives /path/to/enriched-archives \
    --output /path/to/cache --workers 40
```

Run one stage using `--stage`. After import, the original ZIP files are no longer
required. Each stage reads committed prerequisites and replaces its own results
in one transaction. Rerunning a stage clears its dependent results and their
completion markers. Export refuses a cache whose selection stage is incomplete.
Use a fresh directory for the previous three-table cache format; there is no
migration or compatibility reader.

| Stage | Tables written | Prerequisites |
| --- | --- | --- |
| `import` | `games`, `game_players`, `hands` | Enriched archives and rank report |
| `replay` | `decisions`, `state` | Import |
| `analysis` | `analysis` | Replay |
| `scoring` | `scoring` | Analysis |
| `yaku` | `yaku`, `hand_yaku` | Replay |
| `participation` | `tile_participation` | Replay |
| `labels` | `labels`, `label_context` | Analysis, scoring, yaku |
| `selection` | `hand_selection`, `hand_focus`, `selection` | Labels, participation |

`--stage all` is the default and finishes by exporting TFRecords. Individual
stages do not export. For example, rerun participation and selection, then export:

```sh
python build_dataset.py --output /path/to/cache --stage participation
python build_dataset.py --output /path/to/cache --stage selection
python export_dataset.py --database /path/to/cache/decisions.sqlite \
    --output /path/to/tfrecords --workers 8
```

Import preserves historical ranks, the verified retained-seat selection, and all
hand events. Unverified rank metadata remains null. AFK filtering and focused
sampling are applied after computation, so changing selection does not require
recomputing features. The imported retained-seat selection is fixed by the rank
enrichment contract; `selection` does not resample the source ranks.

Feature stages reconstruct the native engine from cached hand events. This
lightweight replay avoids depending on a serialized native-engine object. Only
the requested family's calculations run: scoring consumes cached structural
results; selection consumes cached features, labels, participation and yaku.
Observed supervision belongs to `labels`, not to the runtime critic.

The rebuild worker selects its input components once per game. Selection reads
cached labels, threat context and participation together; it does not construct
the replay engine. Retrospective yaku sampling has its own replay pass.
`model.records.serialize_observation` owns direct TFRecord serialization;
`cache.encode_fields` and `cache.decode_fields` handle compressed components,
with SQL row keys supplied by the rebuild worker.

## Data ownership

Games and hands have integer primary keys. Decisions and their components share
`(hand_id, decision_index, actor)`, with foreign keys enforced during writes.
Original textual game/hand identifiers are stored once in the parent tables.

`state`, `analysis`, `scoring`, `yaku`, and `labels` each contain a `fields` BLOB
with disjoint named tensors in the existing numeric encoding, compressed with
zlib level 1 to keep sparse arrays and repeated field names compact. Each component
has protobuf framing, allowing export to concatenate the components and fresh
metadata using protobuf's message-merge semantics. There is no table of finished
training examples. This retains independent recomputation without encoding
hundreds of tensor fields on every export. Changes to a component's tensor
encoding require regenerating that component and its dependents.

`analysis` owns shanten, ukeire, upgrades, completion probabilities and efficiency.
`scoring` owns scored waits and closed-riichi alternatives. `yaku` owns public
eligibility for the actor, opponents and candidate actions. `hand_yaku.eligible`
is the retrospective per-seat union of eligible yaku over recorded tenpai states,
ordered by `log_dataset.scoring.NAMES`. It is sampling evidence, never
a model input. `hand_focus` derives the configured focus subset from that union.

`tile_participation.probabilities` holds 34 float32 values in base-tile order,
computed with the cached concealed counts, unavailable counts and legal discard
set. Negative values preserve the native search's unresolved result. Positions
without discard candidates have a zero vector and an empty cached discard mask.
The focused-sampling threshold is applied only by selection.

## Indexes and export

- `games`: integer primary key, unique external `game_id`, and
  `games_export(shard, game_id)` for shard enumeration.
- `hands`: integer primary key and unique `(game_id, kyoku_id)` for game reads.
- `game_players`, `hand_yaku`, `hand_focus`: clustered `(game_id, actor)` or
  `(hand_id, actor)` primary keys using `WITHOUT ROWID`.
- `decisions` and every decision component: clustered
  `(hand_id, decision_index, actor)` primary key using `WITHOUT ROWID`.
- `hand_selection`: integer `hand_id` primary key.
- `stages` and `metadata`: clustered text-name primary keys.

These are the indexes used by the stage readers and exporter. The old
`positions_phase`, `discard_actions_tile`, and `records_export` indexes no longer
apply to this schema. No secondary indexes are maintained on wide tensor BLOBs
or low-selectivity selection flags.

Export streams retained decisions in `(hand_id, decision_index, actor)` order,
using one joined query per shard. Imported games are assigned contiguous hand
IDs and shards, so a shard is a primary-key range. Each component lookup uses
its matching primary key; export needs no sort and no per-record SQL calls.
Eight export threads process independent shards by default; `--workers`
controls this count. Threads share the TensorFlow import and each owns its
SQLite connection and writer. The writer reuses its metadata
protobuf and concatenates the decompressed tensor components. GZIP TFRecords and their
parsed feature/label contract remain unchanged.

SQLite WAL allows workers to read committed prerequisite tables while the
coordinator writes one stage. Game IDs are read in batches of two per worker;
the coordinator does not construct a task list for the entire corpus.
A failed stage rolls back to the previous completed state. A successful upstream
rerun removes the output summary until export completes again. Running selection
alone never publishes a training summary for old TFRecord shards. Exports
publish each shard via a `.partial` file, then write a fresh summary after checking
the row count. A complete output directory is not published atomically; export
to a separate directory when another process is consuming existing TFRecords.
