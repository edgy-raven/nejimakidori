# Friendly-room accounts

Guests `friendly-1` through `friendly-3` share the model server but have
separate browser profiles and control ports (8801–8803). Never copy the main
account's profile or use its control port 8795. Runtime state and `config.json`
live outside the checkout.

Friendly launchers enable autoplay by default; account verification and the
friendly-room authorization still gate every automated action.

## RiichiLab arena client

`python -m bots.riichilab` plays four-player games with
a frozen release policy. It runs independently of the friendly controllers.
Keep the bot token in a private file (mode `0600`), outside the repository.
The source argument points to a writable copy of the release's `source/model`
directory, with its sibling feature and dataset modules; the native shanten
loader needs a writable build directory. Keep the original release intact.
Older flat releases use the copied `source` directory directly. Set
`--memory-mb` to cap TensorFlow GPU memory (default: 4096).

```sh
python docker/model_entrypoint.py python -m bots.riichilab \
  --model /path/to/release/saved_model \
  --source /path/to/release-source/model \
  --device GPU-UUID --token-file /private/path/token \
  --output /mnt/drv1/nejimakidori/riichilab/validation --mode validate
```

Every invocation first plays a local East game using masked observations to
warm up inference and check legal actions. Validation waits for the server's
`validation_result`, after `end_game`. Once activated, use `--mode ranked
--games 10` for a bounded run, or `--games 0` for continuous ranked play.
Timestamped game filenames preserve previous logs across service restarts.
One bot token supports one connection
at a time. Requests, replies, latency and action receipts are saved as JSONL;
credentials are excluded. Ranked sessions stop after a game with non-accepted
action receipts. The client exits on disconnection or errors; the persistent
user service restarts it after 60 seconds to join a new game.
See the [RiichiLab protocol](https://riichi.dev/docs/protocol).

The host's `riichilab.service` unit lives in `../codex_env/riichilab.service`.
It runs on the RTX 2080 Ti alongside model serving, joins ranked games
continuously, and starts at boot through the lingering user manager.
Use `systemctl --user status riichilab` and `journalctl --user -u riichilab -f`
to inspect it, or `systemctl --user stop riichilab` to stop it.
Game logs live under `/mnt/drv1/nejimakidori/riichilab/ranked/`.
The initial handover waits for `riichilab-trial.service` to finish so its active
game is not interrupted.

`../codex_env/riichilab-v0.service` runs the September 12 optimized release
as a separate bot on the same GPU with a 2048 MiB TensorFlow limit. Its private
token is in `../codex_env/state/riichilab-v0/token`, and its game logs are in
`/mnt/drv1/nejimakidori/riichilab/v0-ranked/`. Manage it with
`systemctl --user stop/start/status riichilab-v0`.

## Fleet commands

Run from the repository root:

```sh
fleet_python=python
"$fleet_python" -m bots.fleet init
"$fleet_python" -m bots.fleet start
"$fleet_python" -m bots.fleet status
"$fleet_python" -m bots.fleet login friendly-1
printf '%s\n' '{"action":"screenshot"}' | "$fleet_python" -m bots.fleet control friendly-1
"$fleet_python" -m bots.fleet summon 12345 --count 3
"$fleet_python" -m bots.fleet hibernate friendly-1
```

`start`, `status` and `hibernate` accept a guest ID or apply to all guests.
`login` and `control` require one. `summon` defaults to three guests.
Put `--root /path/to/fleet` before the action to use another runtime directory;
The controller image supplies Chromium and Python vision dependencies.

`init` preserves profiles and PINs and regenerates container launchers. The
image runs `live/web/mahjongsoul_friendly_live.mjs` from this checkout. See
[controller operations](../live/web/README.md).

## Account onboarding

Profiles do not create accounts. After `init` and `start`, open `login` for
one guest and complete official registration with its own email and verification
code. Inspect fresh screenshots; send input through local stdin, never Discord:

```json
{"action":"screenshot"}
{"action":"click","x":640,"y":360}
{"action":"type","value":"text to enter"}
```

Coordinates use the full 1280×720 screenshot. After login, copy status fields
`account.accountId` and `account.nickname` into that guest's `account_id` and
`nickname` config fields. Reload only while idle:

```sh
"$fleet_python" -m bots.fleet init
"$fleet_python" -m bots.fleet hibernate friendly-1
# Continue only after hibernate succeeds; active games must finish first.
"$fleet_python" -m bots.fleet start friendly-1
"$fleet_python" -m bots.fleet login friendly-1
"$fleet_python" -m bots.fleet status friendly-1
```

Check `accountVerified:true`, then repeat for the other guests. Unconfigured
guests cannot join or play; configured identities must match. A summon verifies
identity and lobby, joins, then confirms Ready. Inspect `needs` and a screenshot
if `ready:false`. A human starts the game; guests cannot autoqueue or auto-start.

Configure guest identities in the private fleet configuration. Reviewer and
ranked accounts remain separate.

## Discord

Configure numeric `discord.guild_id`, `discord.channel_id` and `discord.role_id`.
A null channel allows commands throughout the server. Using the server ID as
the role grants everyone access; a null role requires Manage Server permission.
Install the application with `bot` and `applications.commands` scopes. Store
its token in private `discord-token` beside the config, mode `0600`.

```sh
"$fleet_python" -m bots.discord_bot /path/to/config.json
```

Startup replaces the bot's global and per-server command sets; only the
configured server receives Mahjong commands. The bridge uses the Guilds intent
for role authorization and does not read messages. Replies are private:

- `/mahjong-summon room:<id> count:<1–3>` summons idle guests. Playing guests or
  guests in another room refuse; repeating an invitation to a ready room is safe.
- `/mahjong-status` reports guest state.
- `/mahjong-dismiss` closes idle browsers and refuses playing guests.

These commands leave main-account autoplay alone. The bridge launcher and logs
remain private runtime state.

Friendly controllers run in Docker Compose containers. `docker compose -p
nejimakidori ps` shows them; `docker compose -p nejimakidori logs -f
friendly-1` follows a controller log. The
reverse SSH forwarder runs as a separate service:

```sh
FLEET_ROOT=/home/fen/projects/codex_env/state/nejimakidori/friendly-bots \
  docker compose -f compose.yml -p nejimakidori up -d reverse-ssh
```

It reads SSH configuration and keys from `~/.ssh` and forwards host port 8793
to the configured Thoth host. Set `THOTH_SSH`, `LOCAL_PORT`, `REMOTE_PORT` or
`SSH_DIRECTORY` in the environment to override its defaults. Follow it with
`docker compose -p nejimakidori logs -f reverse-ssh`.

The Compose stack includes the three friendly containers and the Discord bot.
Guest autoplay is off at startup. The bot can summon guests when requested.

## Recovery and room lifecycle

Dante checks guests every five seconds. A verified guest loading a room for
45 seconds gets at most two recovery attempts: close Chrome, clear Unity game
databases and reopen the saved login. Cold summons also clear those databases;
cookies and login storage remain. Main and reviewer accounts are excluded.

The controller checks notification-maintained membership every 1.5 seconds.
A known all-bot roster closes the browser; unknown membership or a disconnect
does not. After results, guests confirm Ready again while humans remain.
They vote Yes on active cancellation votes without initiating cancellation,
retrying every five seconds until acknowledged or expired, even with autoplay off.

## Scheduled player log scores

`python -m bots.track_player /path/to/config.json` polls Amae-Koromo's four-player
records in the configured mode, requests the selected player's review
score, and replies on Discord with score, placement and replay link. Identity
must match the native record and review seat before posting.

Config fields: `player_id`, `mode`, `nickname`, `amae_url`, `review_url`, `discord_url`,
`guild_id`, `channel_id`, `reply_to`, and `token_path`. Keep runtime config,
ledger and logs outside the checkout; tokens stay private.
Use mode `9` for Gold South, matching the last component of the profile URL.
Use `8,9,11,12,15,16` for all indexed four-player ranked modes.
Run once with `--initialize` to record existing games without posting, then
schedule the ordinary command with `*/30 * * * *`.

Polls lock the ledger and save pending scores and message IDs atomically.
Interrupted posts are reconciled against Discord before retrying. Paginated
polls reread the fixed start boundary, including a day of overlap for games
already in progress, so delayed indexing is caught. Only indexed matches are
visible; older history is not backfilled.
