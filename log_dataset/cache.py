"""Normalized, disposable preprocessing cache and its stage dependencies."""

import contextlib
import pathlib
import sqlite3
import zlib

import numpy
import tensorflow as tf

import feature_vector
from model import records

FEATURE_TABLES = {name: {} for name in ("state", "analysis", "scoring", "yaku")}
for _name, _definition in feature_vector.INPUTS.items():
    if "yaku_possibility" in _name:
        FEATURE_TABLES["yaku"][_name] = _definition
    elif "tenpai_values" in _name or "closed_riichi_values" in _name:
        FEATURE_TABLES["scoring"][_name] = _definition
    elif any(
        word in _name
        for word in ("shanten", "ukeire", "upgrade", "efficiency", "completion")
    ):
        FEATURE_TABLES["analysis"][_name] = _definition
    else:
        FEATURE_TABLES["state"][_name] = _definition

DEPENDENCIES = {
    "import": (),
    "replay": ("import",),
    "analysis": ("replay",),
    "scoring": ("analysis",),
    "yaku": ("replay",),
    "participation": ("replay",),
    "labels": ("analysis", "scoring", "yaku"),
    "selection": ("labels", "participation"),
}
TABLES = {
    "import": ("games", "game_players", "hands"),
    "replay": ("decisions", "state"),
    "analysis": ("analysis",),
    "scoring": ("scoring",),
    "yaku": ("yaku", "hand_yaku"),
    "participation": ("tile_participation",),
    "labels": ("labels", "label_context"),
    "selection": ("hand_selection", "hand_focus", "selection"),
}
KEY = "hand_id, decision_index, actor"


def connect(database):
    return contextlib.closing(
        sqlite3.connect(
            pathlib.Path(database).resolve().as_uri() + "?mode=ro", uri=True
        )
    )


def create(connection):
    connection.executescript("""
        CREATE TABLE stages (name TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE metadata (
            name TEXT PRIMARY KEY, value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE games (
            id INTEGER PRIMARY KEY, game_id TEXT NOT NULL UNIQUE,
            date TEXT NOT NULL, archive TEXT NOT NULL, member TEXT NOT NULL,
            shard INTEGER NOT NULL, hanchan TEXT NOT NULL
        );
        CREATE TABLE game_players (
            game_id INTEGER NOT NULL REFERENCES games(id),
            actor INTEGER NOT NULL, dan INTEGER,
            rating REAL, retained INTEGER NOT NULL,
            PRIMARY KEY (game_id, actor)
        ) WITHOUT ROWID;
        CREATE TABLE hands (
            id INTEGER PRIMARY KEY,
            game_id INTEGER NOT NULL REFERENCES games(id),
            kyoku_id TEXT NOT NULL, events TEXT NOT NULL,
            UNIQUE (game_id, kyoku_id)
        );
        CREATE TABLE decisions (
            hand_id INTEGER NOT NULL REFERENCES hands(id),
            decision_index INTEGER NOT NULL, actor INTEGER NOT NULL,
            event_index INTEGER NOT NULL, phase INTEGER NOT NULL,
            PRIMARY KEY (hand_id, decision_index, actor)
        ) WITHOUT ROWID;
        CREATE TABLE hand_yaku (
            hand_id INTEGER NOT NULL REFERENCES hands(id),
            actor INTEGER NOT NULL, eligible BLOB NOT NULL,
            PRIMARY KEY (hand_id, actor)
        ) WITHOUT ROWID;
        CREATE TABLE hand_focus (
            hand_id INTEGER NOT NULL REFERENCES hands(id),
            actor INTEGER NOT NULL, yaku_focus INTEGER NOT NULL,
            PRIMARY KEY (hand_id, actor)
        ) WITHOUT ROWID;
        CREATE TABLE hand_selection (
            hand_id INTEGER PRIMARY KEY REFERENCES hands(id),
            retained INTEGER NOT NULL
        );
        """)
    for table, columns in {
        **{
            table: "fields BLOB NOT NULL"
            for table in (*FEATURE_TABLES, "labels")
        },
        "label_context": "opponent_high_value BLOB NOT NULL",
        "tile_participation": "probabilities BLOB NOT NULL",
        "selection": "focused INTEGER NOT NULL, retained INTEGER NOT NULL",
    }.items():
        connection.execute(
            f'CREATE TABLE "{table}" ('
            "hand_id INTEGER NOT NULL, decision_index INTEGER NOT NULL, "
            f"actor INTEGER NOT NULL, {columns}, PRIMARY KEY ({KEY}), "
            f"FOREIGN KEY ({KEY}) REFERENCES decisions({KEY})) WITHOUT ROWID"
        )


def require(connection, stages):
    completed = {
        row[0] for row in connection.execute("SELECT name FROM stages")
    }
    missing = set(stages) - completed
    if missing:
        raise ValueError(
            f"incomplete cache stages: {', '.join(sorted(missing))}"
        )


def invalidate(connection, stage):
    invalid = {stage}
    for name, dependencies in DEPENDENCIES.items():
        if invalid.intersection(dependencies):
            invalid.add(name)
    for name in reversed(DEPENDENCIES):
        if name in invalid:
            for table in reversed(TABLES[name]):
                connection.execute(f'DELETE FROM "{table}"')
            connection.execute("DELETE FROM stages WHERE name = ?", (name,))


def feature_rows(connection, hand_id, tables):
    """Read each component once per hand, never once per decision."""
    result = {}
    for table in tables:
        for row in connection.execute(
            f'SELECT * FROM "{table}" WHERE hand_id = ? '
            "ORDER BY decision_index, actor",
            (hand_id,),
        ):
            key = row[1:3]
            if key not in result:
                result[key] = feature_vector.FeatureVector().tensors
            for name, value in decode_fields(
                row[3], FEATURE_TABLES[table], "feature"
            ).items():
                result[key][name][...] = value
    return result


def encode_fields(values, definitions, prefix):
    # Each family owns disjoint tensor fields. Protobuf message concatenation
    # merges these fields at export, without decoding or rebuilding tensors.
    fields = tf.train.Example()
    for name, definition in definitions.items():
        fields.features.feature[prefix + "/" + name].bytes_list.value.append(
            records.encode(
                prefix + "/" + name,
                numpy.asarray(values[name], dtype=definition["dtype"]).reshape(
                    definition["shape"]
                ),
            )
        )
    return zlib.compress(fields.SerializeToString(), level=1)


def decode_fields(raw, definitions, prefix):
    fields = tf.train.Example.FromString(zlib.decompress(raw)).features.feature
    return {
        name: records.decode(
            prefix + "/" + name,
            fields[prefix + "/" + name].bytes_list.value[0],
            definition,
        )
        for name, definition in definitions.items()
    }


def export_query():
    # The clustered decision key is the streaming driver; component lookups
    # use that same key. A shard is a contiguous range of imported hands.
    return (
        "SELECT g.shard, g.game_id, h.kyoku_id, "
        "s.decision_index, s.actor, hl.yaku_focus, s.focused, "
        + ", ".join(f"{table}.fields" for table in (*FEATURE_TABLES, "labels"))
        + " FROM selection AS s "
        "CROSS JOIN hands AS h ON h.id = s.hand_id "
        "CROSS JOIN games AS g ON g.id = h.game_id "
        "CROSS JOIN hand_focus AS hl "
        "ON hl.hand_id = s.hand_id AND hl.actor = s.actor "
        + " ".join(
            f'CROSS JOIN "{table}" ON '
            + " AND ".join(
                f"{table}.{key} = s.{key}" for key in KEY.split(", ")
            )
            for table in (*FEATURE_TABLES, "labels")
        )
        + " WHERE s.retained = 1"
        + " AND s.hand_id BETWEEN ? AND ?"
        + " ORDER BY s.hand_id, s.decision_index, s.actor"
    )


def summary(connection):
    require(connection, ("selection",))
    examples, focused = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(focused), 0) "
        "FROM selection WHERE retained = 1"
    ).fetchone()
    return {
        **records.DATA_CONTRACT,
        "examples": examples,
        "focused_examples": focused,
        "games": connection.execute("SELECT COUNT(*) FROM games").fetchone()[0],
        "shards": connection.execute(
            "SELECT COUNT(DISTINCT shard) FROM games"
        ).fetchone()[0],
    }
