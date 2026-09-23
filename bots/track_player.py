"""Poll Amae-Koromo and reply with new game review scores on Discord."""

import argparse
import hashlib
import json
import pathlib
import time

import requests

from . import fleet


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def player_games(session, settings, start, end):
    games = {}
    cursor = start
    while cursor < end:
        response = session.get(
            f"{settings['amae_url']}/pl4/player_records/"
            f"{settings['player_id']}/{cursor}/{end}",
            params={"mode": "8,9,11,12,15,16", "limit": 500},
            timeout=60,
        )
        response.raise_for_status()
        page = response.json()
        for game in page:
            games[game["uuid"]] = game
        if len(page) < 500:
            break
        cursor = (max(game["startTime"] for game in page) + 1) * 1000
    return sorted(games.values(), key=lambda game: game["startTime"])


def review_game(session, settings, record):
    link_id = 1358437 + ((7 * settings["player_id"] + 1117113) ^ 86216345)
    link = (
        "https://mahjongsoul.game.yo-star.com/?paipu="
        f"{record['uuid']}_a{link_id}"
    )
    with session.post(
        settings["review_url"] + "/api/replay",
        json={"url": link, "seat": "auto"},
        stream=True,
        timeout=(10, 180),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            message = json.loads(line)
            if message["type"] == "error":
                raise RuntimeError(message["message"])
            if message["type"] != "complete":
                continue
            game = message["game"]
            player = next(
                player
                for player in game["players"]
                if player["accountId"] == settings["player_id"]
            )
            if game["seat"] != player["seat"]:
                raise ValueError("Reviewer selected the wrong player")
            grade = game["grades"][player["seat"]]
            if grade["score"] is None:
                raise ValueError("Review contains no graded decisions")
            placement = next(
                index + 1
                for index, player in enumerate(
                    sorted(record["players"], key=lambda row: -row["score"])
                )
                if player["accountId"] == settings["player_id"]
            )
            return (
                f"**{settings['nickname']} — log score: "
                f"{grade['score']:.2f}/100**\n"
                f"Placement: {placement}/4 · "
                f"{grade['graded']} graded decisions\n"
                f"Played: <t:{record['startTime']}:f>\n"
                f"Replay: <{link}>"
            )
    raise RuntimeError("Reviewer stream ended without a completed score")


def recover_message(session, settings, report, bot_id):
    # A post may have succeeded just before a timeout or process crash.
    # Check Discord before retrying, even beyond its short nonce window.
    after = str((report["attempted_at"] - 1420070400000) << 22)
    params = {"limit": 100}
    while True:
        response = session.get(
            f"{settings['discord_url']}/channels/"
            f"{settings['channel_id']}/messages",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        messages = response.json()
        for message in messages:
            if (
                message["author"]["id"] == bot_id
                and message["content"] == report["content"]
            ):
                return message["id"]
        if len(messages) < 100:
            return None
        oldest = min(int(message["id"]) for message in messages)
        if oldest <= int(after):
            return None
        params["before"] = str(oldest)


def poll(settings, state_path):
    state = json.loads(state_path.read_text())
    with requests.Session() as public, requests.Session() as discord:
        discord.headers["Authorization"] = (
            "Bot " + pathlib.Path(settings["token_path"]).read_text().strip()
        )
        response = discord.get(
            settings["discord_url"] + "/users/@me", timeout=30
        )
        response.raise_for_status()
        bot_id = response.json()["id"]
        for record in player_games(
            public, settings, state["start_time"], int(time.time() * 1000)
        ):
            if record["uuid"] in state["seen"]:
                continue
            if record["uuid"] not in state["pending"]:
                state["pending"][record["uuid"]] = {"record": record}
        save_state(state_path, state)
        for uuid, report in list(state["pending"].items()):
            if "content" not in report:
                report["content"] = review_game(
                    public, settings, report["record"]
                )
                save_state(state_path, state)
            message_id = None
            if "attempted_at" in report:
                message_id = recover_message(discord, settings, report, bot_id)
            if message_id is None:
                report["attempted_at"] = int(time.time() * 1000) - 1000
                save_state(state_path, state)
                response = discord.post(
                    f"{settings['discord_url']}/channels/"
                    f"{settings['channel_id']}/messages",
                    json={
                        "content": report["content"],
                        "allowed_mentions": {
                            "parse": [],
                            "replied_user": False,
                        },
                        "message_reference": {
                            "message_id": settings["reply_to"],
                            "channel_id": settings["channel_id"],
                            "guild_id": settings["guild_id"],
                        },
                        "nonce": hashlib.sha256(uuid.encode()).hexdigest()[:24],
                        "enforce_nonce": True,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                message_id = response.json()["id"]
            state["seen"][uuid] = message_id
            del state["pending"][uuid]
            save_state(state_path, state)
            print(f"Posted {uuid}: {message_id}", flush=True)
        state["checked_at"] = int(time.time() * 1000)
        save_state(state_path, state)
        print(f"Checked {settings['nickname']}; no pending games", flush=True)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    state_path = args.config.with_name("state.json")
    with fleet.locked(args.config.with_name("poll.lock")):
        if args.initialize:
            if state_path.exists():
                raise FileExistsError(state_path)
            # Baseline already indexed games; retain a day of overlap for
            # games in progress and delayed indexing at installation time.
            now = int(time.time() * 1000)
            with requests.Session() as session:
                games = player_games(session, settings, now - 86400000, now)
            save_state(
                state_path,
                {
                    "start_time": now - 86400000,
                    "checked_at": now,
                    "seen": {game["uuid"]: None for game in games},
                    "pending": {},
                },
            )
            print(f"Baseline saved: {len(games)} existing games")
        else:
            poll(settings, state_path)


if __name__ == "__main__":
    run()
