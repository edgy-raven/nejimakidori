import { app, BrowserWindow, globalShortcut, ipcMain } from "electron";
import { fileURLToPath } from "node:url";
import { GameTracker } from "./tracker.mjs";

const overlayURL = new URL("./index.html", import.meta.url).href;
const gameURL = "https://mahjongsoul.game.yo-star.com/";
let overlay;
let game;
let clickThrough = false;
let clickThroughShortcut = false;
let settings = {
  base: "https://ogame.squareqrow.com/nejimakidori/api/live/model",
  token: "",
};
let snapshot;

function publish(value) {
  snapshot = value;
  if (overlay && !overlay.isDestroyed()) {
    overlay.webContents.send("snapshot", {...snapshot, clickThrough});
  }
}

const tracker = new GameTracker({
  publish,
  async predict(body, signal) {
    if (!settings.token) throw new Error("Set the model API token first.");
    const response = await fetch(`${settings.base}/predict-live`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${settings.token}`,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.any([signal, AbortSignal.timeout(30000)]),
      redirect: "error",
    });
    if (!response.ok) throw new Error(`Model API returned HTTP ${response.status}`);
    return response.json();
  },
});

function assertOverlay(event) {
  if (event.sender !== overlay.webContents ||
      event.senderFrame !== overlay.webContents.mainFrame ||
      event.senderFrame.url !== overlayURL) {
    throw new Error("Invalid IPC sender");
  }
}

function toggleClickThrough() {
  if (!clickThroughShortcut) {
    throw new Error("Ctrl+Shift+O is unavailable; click-through is disabled.");
  }
  clickThrough = !clickThrough;
  overlay.setIgnoreMouseEvents(clickThrough, {forward: true});
  publish(snapshot);
}

async function startGame() {
  if (!settings.token) throw new Error("Set the model API token first.");
  if (game && !game.isDestroyed()) {
    game.show();
    game.focus();
    return;
  }
  tracker.reset();
  game = new BrowserWindow({
    width: 1280, height: 800, title: "Mahjong Soul",
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true, sandbox: true, nodeIntegration: false,
      partition: "persist:mahjong-soul",
    },
  });
  game.webContents.session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  game.webContents.session.setPermissionCheckHandler(() => false);
  game.webContents.setWindowOpenHandler(() => ({action: "deny"}));
  game.webContents.on("will-navigate", (event, url) => {
    if (new URL(url).origin !== new URL(gameURL).origin) event.preventDefault();
  });
  game.webContents.on("did-start-navigation", (_event, _url, inPlace, mainFrame) => {
    if (mainFrame && !inPlace) tracker.reset();
  });
  game.webContents.debugger.on("message", (_event, method, params) => {
    tracker.handle({method, params});
  });
  game.webContents.debugger.on("detach", () => tracker.reset("disconnected"));
  game.on("closed", () => {
    game = null;
    tracker.close();
  });
  game.webContents.debugger.attach("1.3");
  await game.webContents.debugger.sendCommand("Network.enable");
  await game.loadURL(gameURL);
}

app.whenReady().then(async () => {
  overlay = new BrowserWindow({
    width: 360, height: 490, minWidth: 320, minHeight: 300,
    frame: false, alwaysOnTop: true, backgroundColor: "#131c26",
    title: "Nejimakidori", autoHideMenuBar: true,
    webPreferences: {
      preload: fileURLToPath(new URL("./preload.cjs", import.meta.url)),
      contextIsolation: true, sandbox: true, nodeIntegration: false,
    },
  });
  overlay.webContents.setWindowOpenHandler(() => ({action: "deny"}));
  overlay.webContents.on("will-navigate", event => event.preventDefault());
  ipcMain.handle("overlay", async (event, command, value) => {
    assertOverlay(event);
    switch (command) {
      case "ready": return {...snapshot, clickThrough, base: settings.base};
      case "configure": {
        if (!value || typeof value.base !== "string" ||
            typeof value.token !== "string" || value.token.length > 4096 ||
            !/^[\x21-\x7e]+$/.test(value.token)) {
          throw new Error("Enter a valid API URL and token.");
        }
        const url = new URL(value.base);
        const local = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
        if ((url.protocol !== "https:" && !(local && url.protocol === "http:")) ||
            url.username || url.password || url.search || url.hash) {
          throw new Error("Use HTTPS, or HTTP on localhost.");
        }
        const base = url.href.replace(/\/+$/, "");
        const response = await fetch(`${base}/health`, {
          headers: {Authorization: `Bearer ${value.token}`},
          signal: AbortSignal.timeout(10000), redirect: "error",
        });
        if (!response.ok) {
          throw new Error(`Model API returned HTTP ${response.status}`);
        }
        settings = {base, token: value.token};
        if (tracker.snapshot.position?.phase) {
          tracker.update(tracker.snapshot.position);
        }
        return {base: settings.base};
      }
      case "start": await startGame(); return;
      case "click-through": toggleClickThrough(); return;
      case "quit": app.quit(); return;
      default: throw new Error("Unknown overlay command");
    }
  });
  await overlay.loadURL(overlayURL);
  const shortcuts = [
    ["CommandOrControl+Shift+O", toggleClickThrough],
    ["CommandOrControl+Shift+H", () => overlay.isVisible() ? overlay.hide() : overlay.showInactive()],
  ];
  for (const [key, callback] of shortcuts) {
    const registered = globalShortcut.register(key, callback);
    if (key === "CommandOrControl+Shift+O") clickThroughShortcut = registered;
    if (!registered) {
      publish({...snapshot, error: `Shortcut unavailable: ${key}`});
    }
  }
  overlay.on("closed", () => app.quit());
});
app.on("will-quit", () => {
  tracker.close();
  settings.token = "";
  globalShortcut.unregisterAll();
});
app.on("window-all-closed", () => app.quit());
