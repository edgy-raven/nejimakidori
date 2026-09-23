"""Create a fixed player-game selection from original Tenhou databases."""

import collections
import gzip
import hashlib
import json
import pathlib
import re
import sqlite3
import urllib.parse
import xml.etree.ElementTree
import zipfile

SAMPLING = {
    "unit": "game_id_seat",
    "rank": "historical_tenhou_dan",
    "rates": {"7": 0.5, "8": 0.75, "9+": 1.0},
    "allocation": "floor_per_year_rank_sha256_order",
    "seed": "nejimakidori-player-ranks-20260921",
    "unverified": "exclude",
}


def select(rows):
    strata = collections.defaultdict(list)
    for row in rows:
        row["seats"] = []
        for seat, dan in enumerate(row["dan"]):
            if dan >= 7:
                strata[(row["game_id"][:4], min(dan, 9))].append((row, seat))
    for (_, dan), players in strata.items():
        players.sort(
            key=lambda player: hashlib.sha256(
                f"{SAMPLING['seed']}:{player[0]['game_id']}:{player[1]}".encode()
            ).digest()
        )
        rate = SAMPLING["rates"][str(dan) if dan < 9 else "9+"]
        for row, seat in players[: int(len(players) * rate)]:
            row["seats"].append(seat)
    for row in rows:
        row["seats"].sort()


def run(archives, databases, output):
    if (pathlib.Path(archives) / "player_ranks.json").exists():
        raise ValueError("archives already stratified")
    root = pathlib.Path(output)
    # An existing selection is never overwritten or sampled again.
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    summary = {"sampling": SAMPLING, "years": {}}
    for path in sorted(pathlib.Path(archives).glob("*.zip")):
        database = pathlib.Path(databases) / f"{path.stem}.db"
        with path.open("rb") as stream, database.open("rb") as db_stream:
            year = {
                "archive_sha256": hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest(),
                "database_sha256": hashlib.file_digest(
                    db_stream, "sha256"
                ).hexdigest(),
                "database": str(database.resolve()),
            }
        with zipfile.ZipFile(path) as archive:
            members = {
                pathlib.Path(member).stem: member
                for member in archive.namelist()
                if member.endswith(".mjson")
            }
            year["games"] = len(members)
            matched = set()
            connection = sqlite3.connect(
                f"file:{database.resolve()}?mode=ro", uri=True
            )
            try:
                for game_id, payload in connection.execute(
                    "SELECT id, log FROM logs WHERE log IS NOT NULL"
                ):
                    if game_id not in members:
                        continue
                    player = xml.etree.ElementTree.fromstring(
                        re.search(
                            rb"<UN\s[^>]+/>", gzip.decompress(payload)
                        ).group()
                    ).attrib
                    names = [
                        urllib.parse.unquote(player[f"n{seat}"])
                        for seat in range(4)
                    ]
                    raw = archive.read(members[game_id])
                    if raw.startswith(b"\x1f\x8b"):
                        raw = gzip.decompress(raw)
                    start = json.loads(raw.split(b"\n", 1)[0])
                    if names != start["names"]:
                        raise ValueError(f"seat names mismatch: {game_id}")
                    ranks = list(map(int, player["dan"].split(",")))
                    if len(ranks) != 4 or any(
                        rank < 0 or rank > 20 for rank in ranks
                    ):
                        raise ValueError(f"invalid ranks: {game_id} {ranks}")
                    rows.append(
                        {
                            "game_id": game_id,
                            # Tenhou codes 10..19 = 1..10 dan; 20 = Tenhoui.
                            "dan": [rank - 9 for rank in ranks],
                            "rating": list(
                                map(float, player["rate"].split(","))
                            ),
                        }
                    )
                    matched.add(game_id)
            finally:
                connection.close()
            year["verified_games"] = len(matched)
            year["missing_games"] = len(members) - len(matched)
            rows.extend(
                {"game_id": game_id, "dan": [], "rating": []}
                for game_id in sorted(members.keys() - matched)
            )
        summary["years"][path.stem] = year
        print(json.dumps({path.stem: year}), flush=True)
    select(rows)
    for year, counts in summary["years"].items():
        games = [row for row in rows if row["game_id"].startswith(year)]
        counts["retained_games"] = sum(bool(row["seats"]) for row in games)
        counts["rank_counts"] = dict(
            collections.Counter(dan for row in games for dan in row["dan"])
        )
        counts["retained_rank_counts"] = dict(
            collections.Counter(
                row["dan"][seat] for row in games for seat in row["seats"]
            )
        )
    lookup = {row["game_id"]: row for row in rows}
    for path in sorted(pathlib.Path(archives).glob("*.zip")):
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(
            root / path.name,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
        ) as target:
            for member in source.namelist():
                raw = source.read(member)
                if raw.startswith(b"\x1f\x8b"):
                    raw = gzip.decompress(raw)
                first, rest = raw.split(b"\n", 1)
                start = json.loads(first)
                if "player_ranks" in start:
                    raise ValueError(f"already stratified: {path} {member}")
                row = lookup[pathlib.Path(member).stem]
                start["player_ranks"] = {
                    "dan": row["dan"],
                    "rating": row["rating"],
                    "retained_seats": row["seats"],
                }
                target.writestr(
                    member,
                    json.dumps(start, ensure_ascii=False).encode()
                    + b"\n"
                    + rest,
                )
        with (root / path.name).open("rb") as stream:
            summary["years"][path.stem]["merged_sha256"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    (root / "player_ranks.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", required=True)
    parser.add_argument("--databases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.archives, args.databases, args.output)
