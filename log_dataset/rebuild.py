"""Rebuild a queryable training corpus from enriched MJAI archives."""

import contextlib
import dataclasses
import gzip
import hashlib
import json
import pathlib
import sqlite3
import time
import zipfile

import numpy
import tensorflow as tf

import feature_vector
from model import records, replay

from . import afk, player_ranks, shanten
from .materialize import write as materialize


@dataclasses.dataclass
class Game:
    game_id: str
    date: object
    archive: object
    member: str


@dataclasses.dataclass
class ShardTask:
    index: int
    root: object
    games: tuple


INDEX_SCHEMA = """
    CREATE TABLE positions (
        game_id TEXT NOT NULL,
        kyoku_id TEXT NOT NULL,
        decision_index INTEGER NOT NULL,
        actor INTEGER NOT NULL,
        focused INTEGER NOT NULL,
        phase INTEGER NOT NULL,
        live_wall_count INTEGER NOT NULL,
        hand_yaku_focus INTEGER NOT NULL,
        tenpai_value REAL NOT NULL,
        PRIMARY KEY (game_id, kyoku_id, decision_index, actor)
    );
    CREATE TABLE records (
        game_id TEXT NOT NULL,
        kyoku_id TEXT NOT NULL,
        decision_index INTEGER NOT NULL,
        actor INTEGER NOT NULL,
        shard INTEGER NOT NULL,
        example BLOB NOT NULL,
        PRIMARY KEY (game_id, kyoku_id, decision_index, actor)
    );
    CREATE TABLE discard_actions (
        game_id TEXT NOT NULL,
        kyoku_id TEXT NOT NULL,
        decision_index INTEGER NOT NULL,
        actor INTEGER NOT NULL,
        tile INTEGER NOT NULL,
        shanten INTEGER NOT NULL,
        ukeire_count INTEGER NOT NULL,
        upgrade_type_count INTEGER NOT NULL,
        upgrade_tile_count INTEGER NOT NULL,
        upgrade_weighted_gain INTEGER NOT NULL,
        ron_mean_points REAL NOT NULL,
        riichi_ron_mean_points REAL NOT NULL,
        PRIMARY KEY (game_id, kyoku_id, decision_index, actor, tile)
    );
    CREATE INDEX discard_actions_tile ON discard_actions(tile);
    CREATE INDEX positions_phase ON positions(phase, focused);
"""


def serialize_observation(observation, focused):
    fields = {}
    for prefix, definitions in (
        ("feature", feature_vector.INPUTS),
        ("label", records.LABELS),
    ):
        for name, definition in definitions.items():
            value = (
                observation.features[name]
                if prefix == "feature"
                else getattr(observation, name)
            )
            raw = records.encode(
                prefix + "/" + name,
                numpy.asarray(value, dtype=definition["dtype"]).reshape(
                    definition["shape"]
                ),
            )
            fields[prefix + "/" + name] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[raw])
            )
    fields["meta/sample_weight"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=[1])
    )
    fields["meta/focused"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[int(focused)])
    )
    fields["meta/game_id"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[observation.game_id.encode()])
    )
    fields["meta/kyoku_id"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[observation.kyoku_id.encode()])
    )
    for name in ("decision_index", "actor", "hand_yaku_focus"):
        fields["meta/" + name] = tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=[int(getattr(observation, name))]
            )
        )
    return tf.train.Example(
        features=tf.train.Features(feature=fields)
    ).SerializeToString()


def process_shard(task):
    index_path = pathlib.Path(task.root) / f"train-{task.index:05d}.sqlite"
    examples = focused_examples = 0
    with contextlib.ExitStack() as stack:
        database = stack.enter_context(
            sqlite3.connect(str(index_path) + ".partial")
        )
        database.execute("PRAGMA journal_mode = OFF")
        database.execute("PRAGMA synchronous = OFF")
        database.executescript(INDEX_SCHEMA)
        archives = {
            archive: stack.enter_context(zipfile.ZipFile(archive))
            for archive in {game.archive for game in task.games}
        }
        for game in task.games:
            raw = archives[game.archive].read(game.member)
            if raw.startswith(b"\x1f\x8b"):
                raw = gzip.decompress(raw)
            events = [json.loads(line) for line in raw.splitlines()]
            seats = events[0]["player_ranks"]["retained_seats"]
            if not seats:
                continue
            hanchan = replay.hanchan_targets(events)
            for round_events in afk.retained_hands(events):
                state = replay.Replay(
                    game.game_id,
                    round_events,
                    hanchan=hanchan,
                )
                try:
                    for observation in state.observations():
                        if observation.actor not in seats:
                            continue
                        # Whole-hand yaku eligibility is sampling metadata only.
                        focused = retain_focused(observation)
                        example = serialize_observation(observation, focused)
                        key = (
                            observation.game_id,
                            observation.kyoku_id,
                            observation.decision_index,
                            observation.actor,
                        )
                        database.execute(
                            "INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                *key,
                                int(focused),
                                int(observation.features["decision_phase"]),
                                int(observation.features["live_wall_count"]),
                                int(observation.hand_yaku_focus),
                                float(observation.tenpai_value),
                            ),
                        )
                        database.execute(
                            "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)",
                            (*key, task.index, example),
                        )
                        for tile in numpy.flatnonzero(
                            observation.features["discard_action_legal"]
                        ):
                            values = observation.features[
                                "discard_tenpai_values"
                            ][tile]
                            database.execute(
                                "INSERT INTO discard_actions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (
                                    *key,
                                    int(tile),
                                    int(
                                        observation.features[
                                            "discard_action_all_shanten"
                                        ][tile]
                                    ),
                                    int(
                                        observation.features[
                                            "discard_action_all_ukeire_count"
                                        ][tile]
                                    ),
                                    int(
                                        observation.features[
                                            "discard_action_all_upgrade_type_count"
                                        ][tile]
                                    ),
                                    int(
                                        observation.features[
                                            "discard_action_all_upgrade_tile_count"
                                        ][tile]
                                    ),
                                    int(
                                        observation.features[
                                            "discard_action_all_upgrade_weighted_gain"
                                        ][tile]
                                    ),
                                    float(values[0, 4]),
                                    float(values[2, 4]),
                                ),
                            )
                        examples += 1
                        focused_examples += focused
                except ValueError as error:
                    error.add_note(
                        f"game={game.game_id} "
                        f"round={round_events[0]['bakaze']}"
                        f"{round_events[0]['kyoku']}-"
                        f"{round_events[0]['honba']} "
                        f"shard={task.index} archive={game.archive} "
                        f"member={game.member}"
                    )
                    raise
    pathlib.Path(str(index_path) + ".partial").replace(index_path)
    return {
        "index_path": str(index_path),
        "examples": examples,
        "focused_examples": int(focused_examples),
    }


def retain_focused(observation):
    if int(observation.features["decision_phase"]) == 3:
        return True
    if int(observation.response_action) in (1, 2):
        return True
    if (
        int(observation.features["decision_phase"]) == 0
        and int(observation.discard_action) != 37
        and (
            observation.hand_yaku_focus
            or (
                int(observation.features["prevailing_wind"]) == 1
                and int(observation.features["kyoku_number"]) in (2, 3)
            )
        )
    ):
        return True
    return bool(
        observation.features["riichi_state"][1:].any()
        or observation.opponent_high_value.any()
        or any(
            observation.features[name + "_self_shanten"] <= 1
            for name in ("normal", "chiitoitsu", "kokushi")
        )
        or observation.features["live_wall_count"] <= 16
        or (
            int(observation.features["decision_phase"]) == 0
            and floating_dora_tradeoff(observation.features)
        )
    )


def floating_dora_tradeoff(packed):
    """Low-participation bonus retention that costs shanten or ukeire.

    Compare the best efficiency cuts with cuts retaining strictly more bonus.
    Red and ordinary fives share structure but have separate bonus costs.
    Unresolved participation never qualifies as low participation.
    """
    cuts = numpy.flatnonzero(packed["discard_action_legal"])
    bonus = packed["dora_multiplicity"][shanten.BASE_IDS[:37]] + (
        numpy.arange(37) >= 34
    )
    best = min(
        cuts,
        key=lambda cut: (
            packed["discard_action_all_shanten"][cut],
            -packed["discard_action_all_ukeire_count"][cut],
            bonus[cut],
        ),
    )
    if bonus[cuts].min() >= bonus[best]:
        return False
    participation = shanten.tenpai_participation(
        packed["candidate_hand_counts"][0],
        packed["known_unavailable_counts"],
        cuts,
    )
    if (participation < 0).any():
        return False
    best_cuts = cuts[
        (
            packed["discard_action_all_shanten"][cuts]
            == packed["discard_action_all_shanten"][best]
        )
        & (
            packed["discard_action_all_ukeire_count"][cuts]
            == packed["discard_action_all_ukeire_count"][best]
        )
        & (bonus[cuts] == bonus[best])
    ]
    return bool(
        numpy.any(
            participation[shanten.BASE_IDS[best_cuts]]
            < records.FOCUSED_SAMPLING["dora_participation_below"]
        )
    )


def run(args):
    import concurrent.futures
    import datetime
    import re

    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rank_summary = json.loads(
        (pathlib.Path(args.archives) / "player_ranks.json").read_text()
    )
    if rank_summary["sampling"] != player_ranks.SAMPLING:
        raise ValueError("player rank sampling contract mismatch")
    games = []
    for archive_path in sorted(pathlib.Path(args.archives).glob("*.zip")):
        with archive_path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != rank_summary["years"][archive_path.stem]["merged_sha256"]:
            raise ValueError(f"rank archive mismatch: {archive_path.name}")
        with zipfile.ZipFile(archive_path) as archive:
            for member in sorted(archive.namelist()):
                match = re.search(r"(202[45])(\d{2})(\d{2})", member)
                if match and member.endswith((".mjson", ".jsonl")):
                    date = datetime.date(*map(int, match.groups()))
                    games.append(
                        Game(
                            game_id=pathlib.Path(member).stem,
                            date=date,
                            archive=archive_path,
                            member=member,
                        )
                    )
    if args.limit_games and len(games) > args.limit_games:
        games = [
            games[index]
            for index in numpy.linspace(
                0, len(games) - 1, args.limit_games, dtype=int
            )
        ]
    if not games:
        raise ValueError("no eligible 2024/2025 games")
    (root / "games.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "game_id": game.game_id,
                    "date": game.date.isoformat(),
                    "archive": str(pathlib.Path(game.archive).resolve()),
                    "member": game.member,
                }
            )
            + "\n"
            for game in games
        )
    )
    tasks = [
        ShardTask(
            index, root, tuple(games[start : start + args.games_per_shard])
        )
        for index, start in enumerate(
            range(0, len(games), args.games_per_shard)
        )
    ]
    shanten.native()
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        results = []
        for result in executor.map(process_shard, tasks):
            results.append(result)
            if (
                len(results) == 1
                or len(results) % 10 == 0
                or len(results) == len(tasks)
            ):
                print(
                    f"records shards={len(results)}/{len(tasks)} "
                    f"examples={sum(row['examples'] for row in results)} "
                    f"elapsed_seconds={time.monotonic() - started:.3f}",
                    flush=True,
                )
    database = root / "decisions.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.executescript(INDEX_SCHEMA)
    for shard_path in [result["index_path"] for result in results]:
        connection.execute("ATTACH DATABASE ? AS shard", (shard_path,))
        for table in ("positions", "records", "discard_actions"):
            connection.execute(
                f"INSERT INTO {table} SELECT * FROM shard.{table}"
            )
        connection.commit()
        connection.execute("DETACH DATABASE shard")
        pathlib.Path(shard_path).unlink()
    connection.close()
    (root / "dataset_summary.json").write_text(
        json.dumps(
            {
                "examples": sum(row["examples"] for row in results),
                "focused_examples": sum(
                    row["focused_examples"] for row in results
                ),
                **records.DATA_CONTRACT,
                "games": len(games),
                "shards": len(results),
            },
            indent=2,
        )
    )
    materialize(database, root)
    (root / "player_rank_selection.json").write_text(
        json.dumps(rank_summary, indent=2)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", action="store_true")
    arguments = parser.parse_args()
    arguments.limit_games = 2 if arguments.experiment else None
    arguments.workers = 1 if arguments.experiment else 40
    arguments.games_per_shard = 32
    run(arguments)
