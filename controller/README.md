# Local Mahjong Soul client

The client opens a visible Chrome window. Log in and join a game manually;
autoplay uses the model API, server game messages and the owned vision worker.
It does not queue games, manage other accounts or send Discord messages.

Install Node 22, Chromium and Python requirements in `vision/requirements.txt`.
From this directory run `npm ci`, copy `.env.example` to `.env.local`, set a
four-digit local control PIN and your expected account ID and nickname, then
run `npm start`. Start the public model service first (`MODEL_URL`, default
`http://127.0.0.1:8792`). A graphical desktop is required for Chrome.

Autoplay starts disabled. Use `npm run control -- status` to inspect the logged-in
account, set both expected identity values and restart before enabling play.
Only one client should use an account at a time.

```sh
printf '%s' '{"action":"autoplay","enabled":true}' | npm run control
```

The local API binds to 127.0.0.1:8795 and requires the configured PIN.
`autoplay`, `screenshot`, `click`, `wheel`, `type`, `hibernate` and `login`
commands accept JSON on stdin. Explicitly resume after inspecting a failure:
`{"action":"autoplay","enabled":true,"resume":true}`.
Keep the Chrome game viewport at 1280×720; screenshots and click targets use
those coordinates. Do not manually play while autoplay is enabled.

The profile defaults to `~/.local/state/nejimakidori/profile`; learned buttons
live beside it. `MAHJONG_SOUL_PROFILE`, `MAHJONG_SOUL_BUTTON_CAPTURES`,
`MAHJONG_SOUL_SCREENSHOT` and `MAHJONG_SOUL_LIVE_STATE` override state paths.
SIGINT/SIGTERM closes Chrome and the vision worker. Failures pause automation
and save evidence beside the live-state file; they never automatically requeue.

`domain/` owns pure tile and replay-state contracts shared with the reviewer.
Vision references live in `vision/static/` and `vision/datasets/`.

Run `npm run test:controller` for protocol, playback, targeting, gameplay,
transport, control and browser-rendering checks.

The optional `docker/Dockerfile.controller` image also needs access to your
local graphical display. Build from the repository root, provide `DISPLAY`
and mount your X11 socket and authorization file read-only, plus a writable
`/state` volume. Keep the model/control ports on loopback; with host networking
the same local `MODEL_URL` works. The normal desktop launch avoids display
forwarding setup.
