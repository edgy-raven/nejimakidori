"""Friendly commands against an isolated external-controller HTTP boundary."""

import json
import pathlib
import tempfile
import types
import unittest
import unittest.mock

import aiohttp
import aiohttp.web

import bots.discord_bot
import bots.fleet


class FriendlyWorkflows(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.settings = bots.fleet.initialize(self.root)
        for number, bot_id in enumerate(bots.fleet.BOT_IDS, 1):
            self.settings["accounts"][bot_id] = {
                "account_id": number,
                "nickname": f"Guest{number}",
            }
        self.states = {bot_id: "hibernating" for bot_id in bots.fleet.BOT_IDS}
        self.commands = []
        self.identity = "friendly-1"
        self.error = None
        app = aiohttp.web.Application()
        app.router.add_route("*", "/{bot_id}/{operation}", self.handle)
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self.session = aiohttp.ClientSession()
        self.addAsyncCleanup(self.session.close)
        self.fleet = bots.fleet.Fleet(
            self.root,
            self.settings,
            types.SimpleNamespace(request=self.request),
        )

    def request(self, method, url, **kwargs):
        # Route external controller URLs to a test-owned ephemeral server.
        # The actual Fleet request, authentication, and JSON handling run intact.
        expected = "http://127.0.0.1:"
        self.assertTrue(url.startswith(expected))
        port, path = url[len(expected) :].split("/", 1)
        self.assertIn(int(port), (8801, 8802, 8803))
        bot_id = f"friendly-{int(port) - 8800}"
        return self.session.request(
            method, f"http://127.0.0.1:{self.port}/{bot_id}/{path}", **kwargs
        )

    async def handle(self, request):
        bot_id = request.match_info["bot_id"]
        self.assertEqual(
            request.headers["Authorization"],
            "Bearer "
            + (self.root / bot_id / "control-pin").read_text().strip(),
        )
        if request.method == "GET":
            return aiohttp.web.json_response(
                {
                    "instanceId": (
                        self.identity if bot_id == "friendly-1" else bot_id
                    ),
                    "friendlyOnly": True,
                    "status": self.states[bot_id],
                    "accountVerified": True,
                    "roomId": 12345,
                    "account": {
                        "nickname": self.settings["accounts"][bot_id][
                            "nickname"
                        ]
                    },
                }
            )
        payload = await request.json()
        self.commands.append((bot_id, payload))
        if self.error and bot_id == "friendly-2":
            return aiohttp.web.json_response({"error": self.error}, status=409)
        if payload["action"] == "hibernate":
            self.states[bot_id] = "hibernating"
        elif payload["action"] == "login":
            self.states[bot_id] = "reconnecting"
        return aiohttp.web.json_response({"roomId": 12345, "ready": True})

    async def test_initialization_summon_and_identity_safety(self):
        pins = [
            (self.root / bot / "control-pin").read_bytes()
            for bot in bots.fleet.BOT_IDS
        ]
        (self.root / "config.json").write_text(json.dumps(self.settings))
        bots.fleet.initialize(self.root)
        for number, bot_id in enumerate(bots.fleet.BOT_IDS):
            directory = self.root / bot_id
            self.assertEqual(
                (directory / "control-pin").read_bytes(), pins[number]
            )
            self.assertEqual(
                (directory / "control-pin").stat().st_mode & 0o777, 0o600
            )
            launcher = (directory / "vision.sh").read_text()
            self.assertIn(str(directory / "profile"), launcher)
            self.assertIn(f"127.0.0.1:{8801 + number}", launcher)
            self.assertIn("mahjongsoul_friendly_live.mjs", launcher)
            self.assertTrue(
                (
                    pathlib.Path(self.settings["controller_directory"])
                    / "mahjongsoul_friendly_live.mjs"
                ).is_file()
            )
            self.assertNotIn("8795", launcher)
        for room, count in ((True, 3), (0, 3), (12345, 4)):
            with self.assertRaises(bots.fleet.FleetError):
                await self.fleet.summon(room, count)
        with self.assertRaises(bots.fleet.FleetError):
            self.fleet.directory("nejimakidori")
        self.identity = "nejimakidori"
        with self.assertRaises(bots.fleet.FleetError):
            await self.fleet.command("friendly-1", {"action": "login"})
        self.assertEqual(self.commands, [])
        self.identity = "friendly-1"
        self.error = "Login required"
        process = types.SimpleNamespace(
            returncode=0,
            communicate=unittest.mock.AsyncMock(return_value=(b"", b"")),
        )
        with unittest.mock.patch(
            "bots.fleet.subprocess.run",
            return_value=types.SimpleNamespace(returncode=0),
        ) as tmux:
            with unittest.mock.patch(
                "bots.fleet.asyncio.create_subprocess_exec",
                return_value=process,
            ) as chrome:
                results = await self.fleet.summon(12345)
        self.assertEqual(
            [item["bot_id"] for item in results], list(bots.fleet.BOT_IDS)
        )
        self.assertEqual([item["ok"] for item in results], [True, False, True])
        self.assertEqual(results[1]["error"], "Login required")
        self.assertEqual(chrome.call_count, 3)
        self.assertEqual(tmux.call_count, 3)
        for call in chrome.call_args_list:
            self.assertTrue(call.args[3].startswith(str(self.root)))
        self.states["friendly-1"] = "playing"
        before = list(self.commands)
        with self.assertRaises(bots.fleet.FleetError):
            await self.fleet.hibernate("friendly-1")
        self.assertEqual(self.commands, before)
        with self.fleet.operation():
            with self.assertRaises(bots.fleet.FleetError):
                await self.fleet.summon(12345)

    async def test_recovery_and_discord_continue_around_failed_accounts(self):
        self.states.update(
            {"friendly-1": "reconnecting", "friendly-2": "playing"}
        )
        process = types.SimpleNamespace(
            returncode=0,
            communicate=unittest.mock.AsyncMock(return_value=(b"", b"")),
        )
        with unittest.mock.patch(
            "bots.fleet.asyncio.create_subprocess_exec", return_value=process
        ) as chrome:
            await self.fleet.recover_stalled(0)
            await self.fleet.recover_stalled(44)
            self.assertEqual(self.commands, [])
            for now in (45, 50, 95, 100, 145):
                await self.fleet.recover_stalled(now)
        self.assertEqual(chrome.call_count, 2)
        self.assertEqual(
            self.commands,
            [
                ("friendly-1", {"action": action})
                for action in (
                    "hibernate",
                    "login",
                    "hibernate",
                    "login",
                    "hibernate",
                )
            ],
        )
        self.assertEqual(self.states["friendly-1"], "hibernating")
        client = bots.discord_bot.FriendlyBot(
            self.fleet, {"guild_id": 1, "channel_id": 2, "role_id": 3}
        )
        self.addAsyncCleanup(client.close)
        interaction = types.SimpleNamespace(
            guild_id=1,
            channel_id=99,
            permissions=types.SimpleNamespace(manage_guild=False),
            user=types.SimpleNamespace(roles=[types.SimpleNamespace(id=3)]),
            response=types.SimpleNamespace(
                send_message=unittest.mock.AsyncMock(),
                defer=unittest.mock.AsyncMock(),
            ),
            followup=types.SimpleNamespace(send=unittest.mock.AsyncMock()),
        )
        before = list(self.commands)
        await client.dismiss(interaction)
        self.assertEqual(self.commands, before)
        interaction.response.send_message.assert_awaited_once()
        self.assertTrue(
            interaction.response.send_message.call_args.kwargs["ephemeral"]
        )
        interaction.channel_id = 2
        await client.status(interaction)
        self.assertIn(
            "friendly-2: playing; Guest2; account verified",
            interaction.followup.send.call_args.args[0],
        )
        await client.dismiss(interaction)
        self.assertEqual(
            interaction.followup.send.call_args.args[0],
            "friendly-1: stopped\nfriendly-2: Account is playing; finish "
            "the friendly game first\nfriendly-3: stopped",
        )
        self.assertTrue(interaction.followup.send.call_args.kwargs["ephemeral"])
        self.assertTrue(client.intents.guilds)
        self.assertFalse(client.intents.members)
        self.assertFalse(client.intents.message_content)
        client.settings.update(channel_id=None, role_id=1)
        interaction.user.roles = [types.SimpleNamespace(id=1)]
        interaction.channel_id = 99
        with unittest.mock.patch.object(
            self.fleet, "summon", new_callable=unittest.mock.AsyncMock
        ) as summon:
            summon.return_value = [
                {
                    "bot_id": "friendly-1",
                    "ok": True,
                    "ready": True,
                    "roomId": 12345,
                }
            ]
            await client.summon(interaction, 12345, 1)
            summon.assert_awaited_once_with(12345, 1)
            self.assertEqual(
                interaction.followup.send.call_args.args[0],
                "friendly-1: ready in room 12345",
            )
            interaction.guild_id = 4
            await client.summon(interaction, 12345, 1)
            self.assertEqual(summon.await_count, 1)
