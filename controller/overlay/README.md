# Windows overlay

The app opens Mahjong Soul in its own game window and shows model advice in a
separate always-on-top window. Tracking reads the game window's WebSocket
traffic through Electron's debugger. It does not capture another browser,
read screen pixels, or play moves automatically.

From PowerShell, with Node.js and npm installed:

```powershell
cd controller
npm ci
cd overlay
npm ci
npm start
```

Enter the model API base URL and its dedicated bearer token, select **Connect**,
then **Open Mahjong Soul**. The default URL is
`https://ogame.squareqrow.com/nejimakidori/api/live/model`; this requires the
hosted model proxy to be deployed and configured. Connect verifies the authenticated `<base>/health` endpoint. The app posts to
`<base>/predict-live`. HTTPS is required except for localhost development.
The API token and URL remain in memory for this app session. Game login cookies
persist in Electron's local application profile.

Log in and start a game in the app's game window. Tracking starts before the
game loads; reconnect traffic rebuilds the current game when supported by the
server. Advice clears when the game position changes, the socket closes, or
tracking fails. Opening game developer tools can detach the tracker; close
and reopen the game window to attach again.

Drag the overlay by its title bar. **Ctrl+Shift+O** toggles click-through;
**Ctrl+Shift+H** hides or shows it. The × button exits both windows. Use windowed
or borderless play; an overlay may not appear over exclusive fullscreen.

## Windows portable build

Run on Windows after installing dependencies above:

```powershell
npm run build:windows
```

The portable executable is written to `overlay/dist/`. The builder packages
only the overlay, the controller's protocol/state/domain modules and schema,
and the controller's production dependencies. It does not bundle game login
cookies, API tokens, training artifacts, or hosted configuration. The executable
is unsigned. A Windows interactive smoke test is required to verify game login,
a complete game, reconnects, hotkeys, and display scaling before distribution.
