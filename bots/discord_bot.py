"""Guild-scoped slash commands for the isolated friendly fleet."""

import asyncio
import contextlib
import json
import pathlib
import time

import aiohttp
import discord
import discord.app_commands

import bots.fleet


class FriendlyBot(discord.Client):
    def __init__(self, fleet, settings):
        super().__init__(
            intents=discord.Intents(guilds=True),
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.fleet = fleet
        self.settings = settings
        self.tree = discord.app_commands.CommandTree(self)
        self.recovery_task = None
        guild = discord.Object(id=int(settings["guild_id"]))
        for name, description, callback in (
            (
                "mahjong-summon",
                "Invite friendly accounts to a room",
                self.summon,
            ),
            ("mahjong-status", "Show friendly account status", self.status),
            ("mahjong-dismiss", "Stop idle friendly accounts", self.dismiss),
        ):
            self.tree.add_command(
                discord.app_commands.Command(
                    name=name, description=description, callback=callback
                ),
                guild=guild,
            )

    async def setup_hook(self):
        self.tree.clear_commands(guild=None)
        await self.tree.sync()
        await self.tree.sync(
            guild=discord.Object(id=int(self.settings["guild_id"]))
        )
        self.recovery_task = asyncio.create_task(self.recover())

    async def on_ready(self):
        for guild in self.guilds:
            if guild.id != int(self.settings["guild_id"]):
                self.tree.clear_commands(guild=guild)
                await self.tree.sync(guild=guild)

    async def recover(self):
        while True:
            await self.fleet.recover_stalled(time.monotonic())
            await asyncio.sleep(5)

    async def close(self):
        if self.recovery_task is not None:
            self.recovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.recovery_task
        await super().close()

    async def authorize(self, interaction):
        role = self.settings["role_id"]
        if (
            interaction.guild_id != int(self.settings["guild_id"])
            or (
                self.settings["channel_id"] is not None
                and interaction.channel_id != int(self.settings["channel_id"])
            )
            or not (
                interaction.permissions.manage_guild
                or (
                    role is not None
                    and any(
                        item.id == int(role) for item in interaction.user.roles
                    )
                )
            )
        ):
            await interaction.response.send_message(
                "This command is restricted to the configured friendly channel and role.",
                ephemeral=True,
            )
            return False
        await interaction.response.defer(ephemeral=True, thinking=True)
        return True

    async def summon(
        self,
        interaction: discord.Interaction,
        room: discord.app_commands.Range[int, 1, 999999],
        count: discord.app_commands.Range[int, 1, 3] = 3,
    ):
        if not await self.authorize(interaction):
            return
        try:
            results = await self.fleet.summon(room, count)
            lines = []
            for result in results:
                if not result["ok"]:
                    message = result["error"]
                elif result["ready"]:
                    message = f"ready in room {result['roomId']}"
                else:
                    needs = result["needs"]
                    message = f"room {result['roomId']}: " + (
                        ", ".join(needs)
                        if isinstance(needs, list)
                        else str(needs)
                    )
                lines.append(f"{result['bot_id']}: {message}")
            message = "\n".join(lines)
        except bots.fleet.FleetError as error:
            message = str(error)
        await interaction.followup.send(message, ephemeral=True)

    async def status(self, interaction: discord.Interaction):
        if not await self.authorize(interaction):
            return
        lines = []
        for bot in bots.fleet.BOT_IDS:
            try:
                status = await self.fleet.status(bot)
                nickname = (status.get("account") or {}).get("nickname")
                lines.append(
                    f"{bot}: {status['status']}; {nickname or 'login required'}; "
                    + (
                        "account verified"
                        if status.get("accountVerified")
                        else "account unverified"
                    )
                )
            except bots.fleet.FleetError as error:
                lines.append(f"{bot}: {error}")
        await interaction.followup.send("\n".join(lines), ephemeral=True)

    async def dismiss(self, interaction: discord.Interaction):
        if not await self.authorize(interaction):
            return
        lines = []
        for bot in bots.fleet.BOT_IDS:
            try:
                await self.fleet.hibernate(bot)
                lines.append(f"{bot}: stopped")
            except bots.fleet.FleetError as error:
                lines.append(f"{bot}: {error}")
        await interaction.followup.send("\n".join(lines), ephemeral=True)


def run(config_path):
    config_path = pathlib.Path(config_path)
    settings = json.loads(config_path.read_text())
    if not settings["discord"]["guild_id"]:
        raise bots.fleet.FleetError("Configure Discord guild first")
    token_path = config_path.with_name("discord-token")
    if token_path.stat().st_mode & 0o777 != 0o600:
        raise bots.fleet.FleetError("discord-token must have mode 0600")

    async def connect():
        async with aiohttp.ClientSession() as session:
            fleet = bots.fleet.Fleet(config_path.parent, settings, session)
            async with FriendlyBot(fleet, settings["discord"]) as client:
                await client.start(token_path.read_text().strip())

    asyncio.run(connect())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    run(parser.parse_args().config)
