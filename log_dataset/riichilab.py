"""Import allowlisted RiichiLab opponents as explicitly assigned dan seats."""

import collections
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import pathlib
import subprocess
import zipfile

from . import player_ranks

TEACHERS = {120: 8, 351: 8, 269: 8, 411: 8}


def fetch(url):
    return subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", url],
        check=True,
        capture_output=True,
    ).stdout


def api(path):
    response = json.loads(fetch("https://api.riichi.dev/api/v1/" + path))
    if not response["ok"]:
        raise ValueError(response["error"])
    return response["data"]


def history(bot, since=None):
    games = {}
    offset = 0
    while True:
        page = api(f"bots/{bot}/games?limit=50&offset={offset}")
        for game in page["games"]:
            if since is None or game["played_at"] >= since:
                games[game["game_id"]] = game
        offset += len(page["games"])
        if offset >= page["total"] or not page["games"]:
            return games
        if since is not None and page["games"][-1]["played_at"] < since:
            return games


def normalize_draw_payments(events):
    """Remove native draw deltas' already-paid riichi deposits.

    Nagashi and zero-transfer exhaustive draws can retain the per-seat
    deposit delta. MJAI reach_accepted already accounts for that payment.
    Accept only the observed, conservation-checked native defect.
    """
    for event in events:
        if event["type"] == "start_kyoku":
            deposits = [0] * 4
        elif event["type"] == "reach_accepted":
            deposits[event["actor"]] += 1000
        elif event["type"] == "ryukyoku" and sum(event["deltas"]):
            corrected = [
                delta + deposit
                for delta, deposit in zip(event["deltas"], deposits)
            ]
            if sum(corrected) or not (
                event["reason"] == "nagashimangan"
                or (
                    event["reason"] == "exhaustive_draw"
                    and corrected == [0] * 4
                )
            ):
                raise ValueError(f"Invalid RiichiLab draw payment: {event}")
            event["deltas"] = corrected


def enrich(game, events, teachers):
    """Retain complete context, but assign ranks only to approved teachers."""
    if events[0]["type"] != "start_game" or events[-1]["type"] != "end_game":
        raise ValueError(f"incomplete game: {game['id']}")
    players = sorted(game["players"], key=lambda player: player["seat"])
    if [player["seat"] for player in players] != list(range(4)):
        raise ValueError(f"not a four-player game: {game['id']}")
    normalize_draw_payments(events)
    ranks = [
        (
            teachers[player["bot_id"]]
            if player["bot_id"] in teachers
            and not player["is_disconnected"]
            and not player["is_penalized"]
            else None
        )
        for player in players
    ]
    events[0].update(
        {
            "names": [player["bot_name"] for player in players],
            "player_ranks": {
                "dan": ranks,
                "rating": [],
                "retained_seats": [],
            },
            "riichilab": {
                "game_id": game["id"],
                "bot_ids": [player["bot_id"] for player in players],
                "rank_source": "user_assigned_training_dan",
            },
        }
    )
    return {
        "game_id": game["played_at"][:10].replace("-", "")
        + "-riichilab-"
        + game["id"],
        "dan": ranks,
        "events": events,
        "metadata": game,
    }


def write_archive(root, rows, own_bots, teachers):
    games = [row["metadata"] for row in rows]
    excluded_games = sorted(
        game["id"]
        for game in games
        if any(player["is_penalized"] for player in game["players"])
    )
    rows = [row for row in rows if row["metadata"]["id"] not in excluded_games]
    player_ranks.select(rows)
    counts = collections.Counter()
    with zipfile.ZipFile(
        root / "riichilab.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for row in sorted(rows, key=lambda row: row["game_id"]):
            row["events"][0]["player_ranks"]["retained_seats"] = row["seats"]
            counts.update(str(row["dan"][seat]) for seat in row["seats"])
            archive.writestr(
                row["game_id"] + ".jsonl",
                "".join(json.dumps(event) + "\n" for event in row["events"]),
            )
    with (root / "riichilab.zip").open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = {
        "sampling": player_ranks.SAMPLING,
        "years": {"riichilab": {"merged_sha256": checksum}},
        "source": "https://riichi.dev",
        "rank_source": "user_assigned_training_dan",
        "own_bots": sorted(own_bots),
        "teacher_dan": teachers,
        "downloaded_games": len(games),
        "action_penalty_excluded_games": excluded_games,
        "games": len(rows),
        "retained_games": sum(bool(row["seats"]) for row in rows),
        "retained_rank_counts": dict(counts),
        "captured_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    (root / "player_ranks.json").write_text(json.dumps(summary, indent=2))
    (root / "games.json").write_text(json.dumps(games, indent=2))
    return summary


def run(args):
    own_bots = set(args.own_bots)
    teachers = {
        int(bot): int(dan)
        for bot, dan in (item.split(":") for item in args.teachers)
    }
    if own_bots & teachers.keys():
        raise ValueError("our bots cannot be imitation teachers")
    if not teachers or any(dan < 7 or dan > 11 for dan in teachers.values()):
        raise ValueError("teachers must have an assigned dan from 7 through 11")
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "raw").mkdir()
    own_games = {}
    for bot in sorted(own_bots):
        own_games.update(history(bot))
    if not own_games:
        raise ValueError("our bots have no completed games")
    since = min(game["played_at"] for game in own_games.values())
    matches = set()
    for bot in sorted(teachers):
        matches.update(set(history(bot, since)) & own_games.keys())

    def download(game_id):
        game = api("games/" + game_id)
        bot_ids = {player["bot_id"] for player in game["players"]}
        if (
            game["game_type"] != "ranked"
            or game["player_count"] != 4
            or not bot_ids & own_bots
            or not bot_ids & teachers.keys()
        ):
            raise ValueError(f"game outside the requested scope: {game_id}")
        date = game["played_at"][:10].replace("-", "/")
        raw = fetch(
            f"https://logs.riichi.dev/mjai-logs/{date}/{game_id}.jsonl.gz"
        )
        (root / "raw" / (game_id + ".jsonl.gz")).write_bytes(raw)
        events = [
            json.loads(line) for line in gzip.decompress(raw).splitlines()
        ]
        return enrich(game, events, teachers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(download, sorted(matches)))
    if not rows:
        raise ValueError("no games against the allowlisted teachers")
    print(json.dumps(write_archive(root, rows, own_bots, teachers), indent=2))
