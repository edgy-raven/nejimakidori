# Play through the local client

The client opens Chromium and connects to the [model API](../service/README.md).
You log in and join a game manually, then enable autoplay for the verified
account. It does not queue the next game automatically.

## Install and configure

Start the model service first. Install Node.js 22, Chromium and Python 3.13,
then run from the repository root:

```sh
python -m pip install -r controller/vision/requirements.txt
cd controller
npm ci
cp .env.example .env.local
```

Set these values in `.env.local`:

| Setting | Value |
| --- | --- |
| `MODEL_URL` | Local API address, usually `http://127.0.0.1:8792`. |
| `CHROME_PATH` | Chromium executable, such as `/usr/bin/chromium`. |
| `PYTHON_PATH` | Python executable with the vision requirements installed. |
| `MAHJONG_SOUL_CONTROL_PIN` | Four digits used to authenticate local controls. |
| `MAHJONG_SOUL_EXPECTED_ACCOUNT_ID` | Account ID allowed to use autoplay. |
| `MAHJONG_SOUL_EXPECTED_NICKNAME` | That account's exact nickname. |
| `MAHJONG_SOUL_AUTOPLAY` | Leave `off` while setting up. |

Run all following commands from `controller/`.

## Start playing

```sh
npm start
```

Log in through the browser. In a second terminal, check the detected account:

```sh
npm run control -- status
```

Copy the account ID and nickname into `.env.local` if needed, then restart the
client. Join a game and enable autoplay:

```sh
printf '%s' '{"action":"autoplay","enabled":true}' | npm run control
```

Keep the game viewport at 1280×720. Use one client per account and disable
autoplay before making manual moves:

```sh
printf '%s' '{"action":"autoplay","enabled":false}' | npm run control
```

Ctrl-C stops the client and closes Chromium and the vision worker.

## Recover from a pause

A failure pauses automation and saves evidence at `<live-state-path>.failures`.
Inspect it before explicitly resuming:

```sh
printf '%s' '{"action":"autoplay","enabled":true,"resume":true}' \
  | npm run control
```

The control endpoint defaults to `http://127.0.0.1:8795`. The CLI reads the PIN
from `.env.local` and sends it with each request. Additional commands and their
payloads are defined in [mahjongsoul_live.mjs](mahjongsoul_live.mjs).

## Local files

| Variable | Default path |
| --- | --- |
| `MAHJONG_SOUL_PROFILE` | `~/.local/state/nejimakidori/profile` |
| `MAHJONG_SOUL_BUTTON_CAPTURES` | `~/.local/state/nejimakidori/button-captures` |
| `MAHJONG_SOUL_SCREENSHOT` | `/tmp/mahjongsoul-live.png` |
| `MAHJONG_SOUL_LIVE_STATE` | `/tmp/nejimakidori-live.json` |

Keep browser profiles and captured state outside Git. Run
`npm run test:controller` for client tests. Browser control lives in this folder;
`domain/` holds game-state utilities and `vision/` holds the Python worker.

The optional [Docker image](../docker/Dockerfile.controller) requires a graphical
display, X11 socket and authorization access, and a writable `/state` volume.
