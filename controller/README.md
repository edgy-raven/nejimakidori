# Play locally in Mahjong Soul

The client opens a visible Chromium window and uses the model API to play.
You sign in and join a game manually. Autoplay is disabled until you enable it.

## Setup

Start [the model service](../README.md#run-v1) first. On the same machine,
install Node.js 22, Chromium and Python 3.13, then run from the repository root:

```sh
python -m pip install -r controller/vision/requirements.txt
cd controller
npm ci
cp .env.example .env.local
```

Edit `.env.local`:

| Setting | Value |
| --- | --- |
| `MODEL_URL` | `http://127.0.0.1:8792` for the local model. |
| `CHROME_PATH` | Path to Chromium, commonly `/usr/bin/chromium`. |
| `PYTHON_PATH` | Python executable with the vision requirements installed. |
| `MAHJONG_SOUL_CONTROL_PIN` | A four-digit PIN for local controls. |
| `MAHJONG_SOUL_EXPECTED_ACCOUNT_ID` | The account to allow for autoplay. |
| `MAHJONG_SOUL_EXPECTED_NICKNAME` | That account's exact nickname. |
| `MAHJONG_SOUL_AUTOPLAY` | Keep `off` for initial setup. |

```sh
npm start
```

Log in through the browser. In a second terminal, from `controller/`, inspect
the detected account:

```sh
npm run control -- status
```

If you did not know the account ID, copy it into `.env.local` with the exact
nickname, then restart the client. Join a game and enable autoplay:

```sh
printf '%s' '{"action":"autoplay","enabled":true}' | npm run control
```

Keep the game viewport at 1280×720; vision and click targets use those
coordinates. Use one client per account and avoid manual moves during autoplay.

## Controls and recovery

Disable autoplay:

```sh
printf '%s' '{"action":"autoplay","enabled":false}' | npm run control
```

A failure pauses automation and saves evidence beside the live-state file.
Inspect the failure before resuming:

```sh
printf '%s' '{"action":"autoplay","enabled":true,"resume":true}' \
  | npm run control
```

The control API binds to `127.0.0.1:8795` and requires the configured PIN.
`npm run control` reads JSON commands from stdin; `status` is a command-line
argument. Other commands include `screenshot`, `click`, `wheel`, `type`,
`hibernate` and `login`. Their payloads are defined in
[mahjongsoul_live.mjs](mahjongsoul_live.mjs).

Stop the client with Ctrl-C. Shutdown closes Chromium and the vision worker.
The client does not automatically queue another game.

## Local state and development

The browser profile defaults to `~/.local/state/nejimakidori/profile`.
Learned buttons live beside it. These variables override local state paths:
`MAHJONG_SOUL_PROFILE`, `MAHJONG_SOUL_BUTTON_CAPTURES`,
`MAHJONG_SOUL_SCREENSHOT` and `MAHJONG_SOUL_LIVE_STATE`.
Keep profiles, credentials and screenshots outside Git.

Run `npm run test:controller` for the client tests. `domain/` contains tile
and replay-state utilities; `vision/` contains the Python vision worker and
its references.

The optional [controller image](../docker/Dockerfile.controller) needs a
local graphical display, `DISPLAY`, access to the X11 socket and authorization
file, and a writable `/state` volume. A normal desktop launch avoids display
forwarding setup.
