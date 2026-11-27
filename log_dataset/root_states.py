"""Prepare and load persistent simulator states for human-root continuations."""

import argparse
import collections
import concurrent.futures
import fcntl
import functools
import gzip
import hashlib
import json
import lzma
import multiprocessing
import pathlib
import shutil
import sqlite3
import time

import numpy

from log_dataset import (
    _native_collection,
    afk,
    archives,
    call_states,
    features,
    records,
    replay,
    sakigiri_regret,
)


class Store:
    """Read saved roots and selection witnesses without replaying source logs."""

    def __init__(self, directory):
        self.directory = pathlib.Path(directory).resolve()
        self.index = sqlite3.connect(
            f"file:{self.directory / 'index.sqlite'}?mode=ro", uri=True
        )
        self.connections = collections.OrderedDict()

    def close(self):
        self.index.close()
        for connection in self.connections.values():
            connection.close()
        self.hand.cache_clear()
        self.witnesses.cache_clear()

    def connection(self, game_id):
        row = self.index.execute(
            "SELECT shard FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"No prepared roots for game {game_id}")
        if row[0] not in self.connections:
            if len(self.connections) == 32:
                self.connections.popitem(last=False)[1].close()
            self.connections[row[0]] = sqlite3.connect(
                f"file:{self.directory / row[0]}?mode=ro", uri=True
            )
        self.connections.move_to_end(row[0])
        return self.connections[row[0]]

    @functools.lru_cache(maxsize=4)
    def hand(self, game_id, kyoku_id):
        payload = (
            self.connection(game_id)
            .execute(
                "SELECT roots FROM hands WHERE game_id = ? AND kyoku_id = ?",
                (game_id, kyoku_id),
            )
            .fetchone()
        )
        if payload is None:
            raise KeyError(f"No prepared roots for {game_id}/{kyoku_id}")
        return {
            tuple(key): value
            for key, value in json.loads(lzma.decompress(payload[0]))
        }

    @functools.lru_cache(maxsize=256)
    def witnesses(self, game_id, kyoku_id):
        payload = (
            self.connection(game_id)
            .execute(
                "SELECT witnesses FROM hands "
                "WHERE game_id = ? AND kyoku_id = ?",
                (game_id, kyoku_id),
            )
            .fetchone()
        )
        if payload is None:
            raise KeyError(f"No prepared witnesses for {game_id}/{kyoku_id}")
        return {tuple(key): value for key, value in json.loads(payload[0])}

    def root(self, game_id, kyoku_id, decision_index, actor, phase):
        return _native_collection.NativeRoot.loads(
            self.hand(game_id, kyoku_id)[decision_index, actor, phase][
                "root"
            ].encode()
        )

    def witness(self, game_id, kyoku_id, decision_index, actor, phase):
        return self.witnesses(game_id, kyoku_id).get(
            (decision_index, actor, phase)
        )


def prepare_hand(game_id, events, actors=(0, 1, 2, 3)):
    """Verify and snapshot each retained policy decision once during building."""
    state = replay.Replay(game_id, events)
    pending = collections.defaultdict(list)
    for event, actor, phase, _, _ in state.policy_decisions():
        if actor in actors:
            pending[event + 1, actor].append(
                (
                    (state.decision_index, actor, features.PHASES[phase]),
                    call_states.batch([features.pack(state, actor, phase)]),
                )
            )
    witnesses = {
        (index, events[index]["actor"]): witness
        for index, witness in sakigiri_regret.round_pairs(
            game_id, events
        ).items()
    }
    collector = _native_collection.NativeCollector.from_hand(
        json.dumps(events), 941
    )
    prepared = {}
    while pending and (request := collector.next()) is not None:
        for decision, identity in enumerate(
            zip(request["event"], request["seat"])
        ):
            for key, expected in pending.pop(identity, []):
                root = collector.snapshot(decision, phase=int(key[2]))
                actual = call_states.select(
                    call_states.from_request(root.fork(1, 941, False).next()),
                    [root.decision],
                )
                for name in expected:
                    numpy.testing.assert_array_equal(
                        actual[name], expected[name], err_msg=f"{key}/{name}"
                    )
                prepared[key] = {
                    "root": root.dumps().decode(),
                    "event": identity[0],
                    "witness": witnesses.get(identity),
                }
        if pending:
            collector.submit_recorded()
    if pending:
        raise ValueError(
            f"Missing roots in {game_id}/{state.kyoku_id}: {pending.keys()}"
        )
    return prepared


def build_shard(task):
    index, directory, games = task
    path = pathlib.Path(directory) / f"roots-{index:05d}.sqlite"
    receipt = path.with_suffix(".json")
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result["sakigiri"] != sakigiri_regret.DEFINITION:
            raise ValueError(f"Outdated sakigiri witnesses: {path}")
        if path.stat().st_size != result["bytes"]:
            raise ValueError(f"Incomplete root-state shard: {path}")
        return result
    if shutil.disk_usage(directory).free < 15 * 1024**3:
        raise OSError("Root-state build requires at least 15 GiB free space")
    temporary = path.with_suffix(".sqlite.partial")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.execute(
        "CREATE TABLE hands (game_id TEXT, kyoku_id TEXT, roots BLOB, "
        "witnesses TEXT, PRIMARY KEY (game_id, kyoku_id)) WITHOUT ROWID"
    )
    count = 0
    hands = 0
    raw_bytes = 0
    for game in games:
        payload = archives.open_archive(game["archive"]).read(game["member"])
        if payload.startswith(b"\x1f\x8b"):
            payload = gzip.decompress(payload)
        events = [json.loads(line) for line in payload.splitlines()]
        for hand in afk.retained_hands(events):
            prepared = prepare_hand(
                game["game_id"],
                hand,
                actors=events[0]["player_ranks"]["retained_seats"],
            )
            payload = json.dumps(list(prepared.items()), separators=(",", ":"))
            connection.execute(
                "INSERT INTO hands VALUES (?, ?, ?, ?)",
                (
                    game["game_id"],
                    f"{hand[0]['bakaze']}{hand[0]['kyoku']}-{hand[0]['honba']}",
                    lzma.compress(payload.encode(), preset=3),
                    json.dumps(
                        [
                            (key, value["witness"])
                            for key, value in prepared.items()
                            if value["witness"] is not None
                        ]
                    ),
                ),
            )
            count += len(prepared)
            hands += 1
            raw_bytes += len(payload)
    connection.commit()
    connection.close()
    temporary.replace(path)
    result = {
        "path": path.name,
        "games": [game["game_id"] for game in games],
        "hands": hands,
        "roots": count,
        "raw_bytes": raw_bytes,
        "bytes": path.stat().st_size,
        "sakigiri": sakigiri_regret.DEFINITION,
    }
    receipt.with_suffix(".json.partial").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    receipt.with_suffix(".json.partial").replace(receipt)
    return result


def run(args):
    directory = pathlib.Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = pathlib.Path(args.data).resolve()
        games = [
            json.loads(line)
            for line in (source / "games.jsonl").read_text().splitlines()
        ]
        games = [
            game
            for game in games
            if records.game_partition(game["game_id"]) < 7
        ]
        if args.limit_games is not None:
            games = games[: args.limit_games]
        if not games:
            raise ValueError("No training games selected for root states")
        plan = {
            "data": str(source),
            "compression": "lzma-preset-3",
            "input_schema": _native_collection.INPUTS,
            "seed": 941,
            "sakigiri": sakigiri_regret.DEFINITION,
            "summary_sha256": hashlib.sha256(
                (source / "dataset_summary.json").read_bytes()
            ).hexdigest(),
            "games_sha256": hashlib.sha256(
                json.dumps(games, sort_keys=True).encode()
            ).hexdigest(),
            "games": len(games),
            "limit_games": args.limit_games,
        }
        plan_path = directory / "build-plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Root-state inputs changed; use a fresh output")
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        tasks = [
            (index, str(directory), games[start : start + 32])
            for index, start in enumerate(range(0, len(games), 32))
        ]
        started = time.monotonic()
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            for result in pool.map(build_shard, tasks):
                results.append(result)
                status = {
                    "shards": len(results),
                    "total_shards": len(tasks),
                    "roots": sum(row["roots"] for row in results),
                    "bytes": sum(row["bytes"] for row in results),
                    "elapsed_seconds": time.monotonic() - started,
                }
                (directory / "status.json").write_text(
                    json.dumps(status, indent=2) + "\n"
                )
                print(json.dumps(status), flush=True)
        temporary = directory / "index.sqlite.partial"
        temporary.unlink(missing_ok=True)
        connection = sqlite3.connect(temporary)
        connection.execute(
            "CREATE TABLE games (game_id TEXT PRIMARY KEY, shard TEXT) "
            "WITHOUT ROWID"
        )
        connection.executemany(
            "INSERT INTO games VALUES (?, ?)",
            [
                (game_id, result["path"])
                for result in results
                for game_id in result["games"]
            ],
        )
        connection.commit()
        connection.close()
        temporary.replace(directory / "index.sqlite")
        (directory / "complete.json").write_text(
            json.dumps({**plan, **status}, indent=2) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-games", type=int)
    args = parser.parse_args()
    if args.workers < 1 or (
        args.limit_games is not None and args.limit_games < 1
    ):
        parser.error("workers and limit-games must be positive")
    run(args)


if __name__ == "__main__":
    main()
