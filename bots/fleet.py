"""Manage three isolated friendly controllers without touching main autoplay."""

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import pathlib
import secrets
import shlex
import signal
import subprocess
import sys
import time

import aiohttp

BOT_IDS = ("friendly-1", "friendly-2", "friendly-3")


class FleetError(Exception):
    """An expected account, controller or process failure."""


@contextlib.contextmanager
def locked(path):
    with open(path, "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FleetError("Another friendly operation is running") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def initialize(root):
    root = pathlib.Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    config = root / "config.json"
    settings = (
        json.loads(config.read_text())
        if config.exists()
        else {
            "controller_directory": str(
                pathlib.Path(__file__).resolve().parents[1] / "live/web"
            ),
            "model_url": "http://127.0.0.1:8501",
            "accounts": {
                bot: {"account_id": None, "nickname": None} for bot in BOT_IDS
            },
            "discord": {"guild_id": None, "channel_id": None, "role_id": None},
        }
    )
    config.write_text(json.dumps(settings, indent=2) + "\n")
    for number, bot in enumerate(BOT_IDS, 8801):
        directory = root / bot
        directory.mkdir(exist_ok=True, mode=0o700)
        (directory / "profile").mkdir(exist_ok=True, mode=0o700)
        pin = directory / "control-pin"
        if not pin.exists():
            with open(
                os.open(pin, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w"
            ) as stream:
                stream.write(f"{secrets.randbelow(10000):04d}\n")
        values = {
            "MAHJONG_SOUL_INSTANCE_ID": bot,
            "MAHJONG_SOUL_BOT_ACCOUNT_IDS": ",".join(
                str(account["account_id"])
                for account in settings["accounts"].values()
                if account["account_id"] is not None
            ),
            "MAHJONG_SOUL_PROFILE": str(directory / "profile"),
            "MAHJONG_SOUL_SCREENSHOT": str(directory / "screenshot.png"),
            "MAHJONG_SOUL_LIVE_STATE": str(directory / "live-state.json"),
            "MAHJONG_SOUL_BUTTON_CAPTURES": str(directory / "button-captures"),
            "MAHJONG_SOUL_CONTROL_URL": f"http://127.0.0.1:{number}",
            "MODEL_URL": settings["model_url"],
            "MAHJONG_SOUL_AUTOPLAY": "on",
        }
        if settings.get("chrome_path"):
            values["CHROME_PATH"] = settings["chrome_path"]
        for field, key in (
            ("account_id", "MAHJONG_SOUL_EXPECTED_ACCOUNT_ID"),
            ("nickname", "MAHJONG_SOUL_EXPECTED_NICKNAME"),
        ):
            if settings["accounts"][bot][field] is not None:
                values[key] = str(settings["accounts"][bot][field])
        (directory / "vision.sh").write_text(
            "#!/bin/sh\nset -eu\ncd "
            + shlex.quote(settings["controller_directory"])
            + "\n"
            + "\n".join(
                f"export {name}={shlex.quote(value)}"
                for name, value in values.items()
            )
            + '\nexport MAHJONG_SOUL_CONTROL_PIN="$(cat '
            + shlex.quote(str(pin))
            + ')"\n'
            + "exec node --env-file-if-exists=.env.local mahjongsoul_friendly_live.mjs"
            + " >>"
            + shlex.quote(str(directory / "vision.log"))
            + " 2>&1\n"
        )
        (directory / "vision.sh").chmod(0o700)
    return settings


class Fleet:
    def __init__(self, root, settings, session):
        self.root = pathlib.Path(root)
        self.settings = settings
        self.session = session
        self.recovery = {}
        if set(settings["accounts"]) != set(BOT_IDS):
            raise FleetError("Exactly the three friendly accounts are required")
        for field in ("account_id", "nickname"):
            values = [
                account[field]
                for account in settings["accounts"].values()
                if account[field] is not None
            ]
            if len(values) != len(set(values)):
                raise FleetError(f"Friendly {field} values must be distinct")
        if any(
            (account["nickname"] or "").lower() == "nejimakidori"
            for account in settings["accounts"].values()
        ):
            raise FleetError("Main account cannot join the friendly fleet")

    def directory(self, bot_id):
        if bot_id not in BOT_IDS:
            raise FleetError("Unknown friendly account")
        return self.root / bot_id

    def operation(self):
        return locked(self.root / "operation.lock")

    async def request(self, bot_id, method, path, payload=None):
        pin = (self.directory(bot_id) / "control-pin").read_text().strip()
        try:
            async with self.session.request(
                method,
                f"http://127.0.0.1:{8801 + BOT_IDS.index(bot_id)}{path}",
                headers={"Authorization": f"Bearer {pin}"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as response:
                result = await response.json()
                if response.status != 200:
                    raise FleetError(result["error"])
                return result
        except (aiohttp.ClientError, TimeoutError) as error:
            raise FleetError(
                f"Controller connection failed: {error}"
            ) from error

    async def status(self, bot_id):
        result = await self.request(bot_id, "GET", "/status")
        if result["instanceId"] != bot_id or not result["friendlyOnly"]:
            raise FleetError(
                "Controller identity does not match friendly account"
            )
        return result

    async def command(self, bot_id, payload):
        await self.status(bot_id)
        return await self.request(bot_id, "POST", "/command", payload)

    async def start(self, bot_id):
        with locked(self.directory(bot_id) / "startup.lock"):
            result = subprocess.run(
                ["tmux", "has-session", "-t", bot_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode:
                result = subprocess.run(
                    [
                        "tmux",
                        "new-session",
                        "-d",
                        "-s",
                        bot_id,
                        str(self.directory(bot_id) / "vision.sh"),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    raise FleetError(result.stderr.strip())
            deadline = time.monotonic() + 10
            while True:
                try:
                    return await self.status(bot_id)
                except FleetError as error:
                    if not isinstance(
                        error.__cause__, (aiohttp.ClientError, TimeoutError)
                    ):
                        raise
                    if time.monotonic() >= deadline:
                        raise
                    await asyncio.sleep(0.25)

    async def clear_game_cache(self, bot_id):
        process = await asyncio.create_subprocess_exec(
            "node",
            str(pathlib.Path(__file__).with_name("clear_game_cache.mjs")),
            self.settings.get("chrome_path", "google-chrome"),
            str(self.directory(bot_id) / "profile"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 80)
        except (TimeoutError, asyncio.CancelledError) as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), 3)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            if isinstance(error, TimeoutError):
                raise FleetError("Game cache maintenance timed out") from error
            raise
        if process.returncode:
            raise FleetError(stderr.decode().strip() or stdout.decode().strip())

    async def summon(self, room_id, count=3):
        if type(room_id) is not int or not 1 <= room_id <= 999999:
            raise FleetError("Room must be an integer from 1 to 999999")
        if type(count) is not int or not 1 <= count <= 3:
            raise FleetError("Count must be an integer from 1 to 3")
        for bot in BOT_IDS[:count]:
            if any(
                self.settings["accounts"][bot][name] is None
                for name in ("account_id", "nickname")
            ):
                raise FleetError(f"Configure account ID and nickname for {bot}")

        async def summon_account(bot):
            try:
                status = await self.start(bot)
                if status["status"] == "hibernating":
                    await self.clear_game_cache(bot)
                result = await self.command(
                    bot, {"action": "summon", "roomId": room_id}
                )
                return {**result, "bot_id": bot, "ok": True}
            except FleetError as error:
                return {"bot_id": bot, "ok": False, "error": str(error)}

        with self.operation():
            return await asyncio.gather(
                *(summon_account(bot) for bot in BOT_IDS[:count])
            )

    async def hibernate(self, bot_id):
        with self.operation():
            if (await self.status(bot_id))["status"] == "playing":
                raise FleetError(
                    "Account is playing; finish the friendly game first"
                )
            return await self.command(bot_id, {"action": "hibernate"})

    async def recover_stalled(self, now):
        try:
            with self.operation():
                for bot in BOT_IDS:
                    try:
                        status = await self.status(bot)
                        if status["status"] not in (
                            "waiting_for_round",
                            "reconnecting",
                        ):
                            self.recovery.pop(bot, None)
                            continue
                        if not status.get("accountVerified") or not status.get(
                            "roomId"
                        ):
                            continue
                        if bot not in self.recovery:
                            self.recovery[bot] = {"since": now, "attempts": 0}
                        if now - self.recovery[bot]["since"] < 45:
                            continue
                        verified = await self.status(bot)
                        if (
                            verified["status"] != status["status"]
                            or verified.get("roomId") != status["roomId"]
                            or not verified.get("accountVerified")
                        ):
                            continue
                        await self.command(bot, {"action": "hibernate"})
                        if self.recovery[bot]["attempts"] < 2:
                            await self.clear_game_cache(bot)
                            await self.command(bot, {"action": "login"})
                            self.recovery[bot]["attempts"] += 1
                            self.recovery[bot]["since"] = now
                    except FleetError:
                        continue
        except FleetError:
            return


async def run(args):
    settings = initialize(args.root)
    if args.action == "init":
        print(f"Initialized friendly fleet in {args.root}")
        return
    async with aiohttp.ClientSession() as session:
        fleet = Fleet(args.root, settings, session)
        if args.action == "summon":
            result = await fleet.summon(int(args.target), args.count)
        elif args.action in ("login", "control"):
            if args.target is None:
                raise FleetError("Choose a friendly account")
            await fleet.start(args.target)
            result = await fleet.command(
                args.target,
                (
                    {"action": "login"}
                    if args.action == "login"
                    else json.load(sys.stdin)
                ),
            )
        else:
            result = {}
            for bot in (args.target,) if args.target else BOT_IDS:
                result[bot] = await getattr(fleet, args.action)(bot)
        print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".local/state/mahjong-friendly",
    )
    parser.add_argument(
        "action",
        choices=(
            "init",
            "start",
            "status",
            "login",
            "control",
            "summon",
            "hibernate",
        ),
    )
    parser.add_argument("target", nargs="?")
    parser.add_argument("--count", type=int, default=3)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
