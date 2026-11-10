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

Enter the API URL and access token supplied by your server operator, select
**Connect**, then **Open Mahjong Soul**. Hosted API access is separate from the
model download. The default address is
`https://ogame.squareqrow.com/nejimakidori/api/live/model`.

The app uses `<base>/health` and `<base>/predict-live`. Remote servers require
HTTPS; localhost can use HTTP. The URL and token are kept for the app session.
Your Mahjong Soul login persists in Electron's local profile.

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

The portable executable is written to `overlay/dist/`. It includes the client
and its dependencies; users supply their own API access and game login.
The executable is unsigned.
