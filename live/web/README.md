# Deployed Mahjong Soul controllers

Ranked and friendly automation share game inference, transport and vision,
with separate lifecycle policies. The official client's WebSocket supplies
exact state; vision verifies UI readiness and click targets. The separate
[`review/`](../../review/README.md) product owns replay review and live display.

## Setup and operation

Install JavaScript dependencies here with `npm ci` and Python vision
dependencies from `../vision/requirements.txt`. Use [.env.example](.env.example)
for private session configuration. Runtime launchers, profiles and credentials
belong outside the repository, for example under `~/.local/state/nejimakidori/`.
PIN files and `.env.local` are ignored; never embed their values in public
snapshots, logs or documentation.

- `npm run live` starts `mahjongsoul_live.mjs` for ranked play.
- `npm run friendly` starts `mahjongsoul_friendly_live.mjs` for friendly rooms.
- `npm run control -- status` reads controller status using the locally
  configured PIN.

Entry points select the role. Each account needs separate profile, state,
screenshot, control endpoint and identity settings. See
[friendly fleet setup](../../bots/README.md). Never share a browser profile
between simultaneous clients. Stop with SIGTERM to close Chrome and vision.
Model and main control defaults are loopback ports 8792 and 8795.

The deployed ranked account is supervised by a user service:

```sh
systemctl --user restart nejimakidori-ranked.service
journalctl --user -u nejimakidori-ranked.service
```

It opens the saved session and restarts after process exit. Do not also
launch a controller against that account in another container. Inspect private
launchers for the active profile and model export rather than assuming checkout
changes are deployed. UI changes need refresh; process code changes need restart.

## Ranked rendering and previews

Automation on means unattended operation: ranked animation callbacks are
capped at 15 fps, and periodic lobby preview captures are disabled. Game-state
messages, decision screenshots, overlay checks and click verification continue
normally. Turning automation off restores the browser's normal animation rate
and 250 ms lobby previews; explicit screenshot requests remain available.

`CHROME_HARDWARE_RENDERING=on` selects headless Vulkan rendering. The host
deployment uses `../codex_env/ranked-rendering.compose.yml` (relative to the
repository root) to expose GPU 1, the RTX 2080 Ti, with NVIDIA graphics libraries
and memory devices. Its Vulkan manifest uses NVIDIA's EGL entry point because
the runtime-injected GLX entry point failed initialization on this host.
Controllers without that setting retain software rendering.

Apply ranked image changes only after a completed game, with requeueing held.
Restore the prior automation settings after verifying login and screenshots.

## Ranked queue and recovery

The ranked controller clears the `/idbfs` and `UnityCache` game databases
only when it reaches the lobby and hibernates for a scheduled break or closed
queue hours. It closes Chrome before clearing the cache. Startup, login and
consecutive matches retain the cache; cookies and login storage are preserved.
`game_cache_cleared` records each completed cleanup. Maintenance errors pause
automation.

Autoqueue defaults on and selects the highest rank-eligible room:
Bronze/Silver East, Gold/Jade/Throne South. Insufficient Copper stops queueing
without selecting a cheaper room. Status reports rank, Copper, room and any
blocking reason. `ranked_rooms.json` contains entry thresholds; regenerate
with `node refresh_ranked_rooms.mjs config.proto lqc.lqbin` using the official
client's current configuration files.

`ranked_schedule.json` is the ranked play policy: timezone, open-hour ranges,
continuation decay and session-break bounds all live there. Ranges include
their start and exclude their end; an existing game always finishes. The
browser hibernates at the verified lobby during breaks and closed windows.
Wake-up requires enabled queueing and no unresolved automation block. Manual
hibernation lasts until manual login or service restart.

`ranked-runtime.json` preserves autoplay, queue stops, continuation count and
break expiry. Restarting must not override manual stops or unresolved incidents.
One More Match is used only while the current mode and Copper remain eligible;
its confirmation dialog submits matchmaking. Rejected or unacknowledged queue
requests are not blindly retried.

Recognized result/reward controls require stable observations. A clicked
control has two five-second observation intervals to advance before manual
help is required. Rank-up and
unknown screens require manual help. In-game failures abandon the failed
decision, save evidence, and disable requeueing while allowing autoplay on
subsequent server decisions. Postgame failures stop automatic clicks. Inspect
`automationBlock` and `lastActionMismatch` when diagnosing a visible button.
A persisted gameplay pause may allow result confirmations solely to return
to the lobby; a failed return removes that permission.

For a durable automation pause, fix the screen and use **Resume automation**,
or `{"action":"autoqueue","resume":true,"enabled":true}`. Ordinary login,
enable commands and manual clicks do not clear the incident. Failed decision
records use `<live-state-path>.gameplay-failures.jsonl`; action mismatches use
`.mismatches.jsonl`. Private `.failures/<incident-id>/` directories retain
screenshots, recognition output and position evidence.

## Friendly accounts

Set `MAHJONG_SOUL_EXPECTED_ACCOUNT_ID` and `MAHJONG_SOUL_EXPECTED_NICKNAME`
for each guest. Missing identity allows onboarding but blocks joining and
play; a mismatch blocks clicks. Status reports `accountVerified` and `ready`.
Use the [fleet CLI](../../bots/README.md) for onboarding and invitations.

`{"action":"summon","roomId":12345}` wakes a guest, verifies identity and
lobby, joins the room, and confirms Ready through the official response.
Allow a 90-second client timeout. Active games and other rooms are refused;
`ready:false` includes a `needs` message for manual inspection. A human starts
the room. Guests cannot enable ranked queueing or automatic room starts.
Friendly automation failures retain their pause policy; local recovery uses
`resume`. Room lifecycle and Discord commands are documented with the fleet.

`MAHJONG_SOUL_RIICHI_ONLY=on` is for dedicated four-riichi test accounts:
force offered riichi, otherwise discard or skip; never claim calls or wins.
It permits all-bot test rooms. Normal friendly guests require a human host.

## Discord reports

`MAHJONG_SOUL_DISCORD_CONFIG` points to private JSON containing `channelId`,
`tokenPath` and `statePath`. Ranked reports include placement, final score,
rank/points, Copper and game UUID after account updates settle. Friendly
controllers do not load the ranked notifier.

The private ledger retains delivered message IDs and manual-help incidents
across restarts. Status exposes delivery errors. Network/server failures and
rate limits retry notification delivery, never gameplay clicks.

## Implementation and verification

Shared gameplay and prediction scheduling live in `mahjongsoul_gameplay.mjs`;
result confirmation in `mahjongsoul_result_controls.mjs`; protocol state in
`mahjongsoul_protocol.mjs` and `mahjongsoul_live_state.mjs`. `BrowserTransport`
owns Chrome startup, sessions and shutdown. `ControllerIO` serializes screenshots,
vision requests and atomic status writes. Ranked and friendly controllers retain
their own account checks, room workflows and recovery policies.
Vision workers live in `../../vision/`. Public assets are served through a
startup-generated route map, not arbitrary filesystem paths.

Preserve native 1280×720 screenshots and coordinates. Discards require two
agreeing frames. Only server-confirmed controls enter the learned button bank;
ambiguous targets require inspection. Call composition follows the server's
original menu order, including red-free alternatives even when model policy
prefers consuming red fives. Public snapshots retain complete predictions.

Learned button references are cropped PNG images named by their control and
stored under `MAHJONG_SOUL_BUTTON_CAPTURES`. Unresolved controls and call
attempts are also saved as PNG images in their respective subdirectories.

From this directory:

```sh
npm run test:controller
PYTHONPATH=../.. python \
  -m unittest discover -s ../../integration -p test_vision_workflows.py
```

For browser parity against a preserved release, use
`node ../../review/test/ui-parity.mjs /path/to/reference/web`. It uses synthetic state,
isolated profiles and temporary servers without account login or game input;
evidence goes to `../../artifacts/ui-parity/`.
