import { FailureCapture } from "./failure_capture.mjs";
import { createGameplay } from "./mahjongsoul_gameplay.mjs";
import { BrowserTransport } from "./browser_transport.mjs";
import { ControllerIO } from "./controller_io.mjs";
import { randomUUID } from "node:crypto";
import { appendFileSync } from "node:fs";
import { setTimeout as wait } from "node:timers/promises";

import { startLiveControl } from "./live_control.mjs";
import { MahjongSoulLiveState } from "./mahjongsoul_live_state.mjs";
import { VisionWorker } from "./mahjongsoul_vision.mjs";
import {
  decodeNotification, decodeGameResponse,
} from "./mahjongsoul_protocol.mjs";

// Standalone client: the user logs in and chooses games in Chrome.
let account = null;
function accountVerified() {
  return Boolean(account &&
    account.accountId === Number(process.env.MAHJONG_SOUL_EXPECTED_ACCOUNT_ID) &&
    account.nickname === process.env.MAHJONG_SOUL_EXPECTED_NICKNAME);
}
const chromePath =
  process.env.CHROME_PATH || "/usr/bin/google-chrome";
const profilePath =
  process.env.MAHJONG_SOUL_PROFILE ||
  process.env.HOME + "/.local/state/nejimakidori/profile";
const screenshotPath =
  process.env.MAHJONG_SOUL_SCREENSHOT || "/tmp/mahjongsoul-live.png";
const liveStatePath =
  process.env.MAHJONG_SOUL_LIVE_STATE || "/tmp/nejimakidori-live.json";
const modelUrl = process.env.MODEL_URL || "http://127.0.0.1:8792";
const modelHealth = await fetch(`${modelUrl}/health`, {
  signal: AbortSignal.timeout(5000),
});
if (!modelHealth.ok || !(await modelHealth.json()).ok) {
  throw new Error(`Model service is not ready: ${modelUrl}`);
}
let autoplay = process.env.MAHJONG_SOUL_AUTOPLAY === "on";
const pythonPath =
  process.env.PYTHON_PATH || "python3";
const vision = new VisionWorker(pythonPath);
const buttonCapturePath = process.env.MAHJONG_SOUL_BUTTON_CAPTURES ||
  process.env.HOME + "/.local/state/nejimakidori/button-captures";
const failureCapture = new FailureCapture(`${liveStatePath}.failures`);
let screenshotTimeouts = [];
let browserRestartScheduled = false;


const browser = new BrowserTransport(handleGameMessage);
const io = new ControllerIO({
  send, vision, failureCapture, screenshotPath, liveStatePath, output,
  resetViewport: false,
});

let positionVersion = 0;
let overlayWatchPending = false;


let screenCapturePending = false;
let gameId = null;
let automationBlock = null;
let afkClicked = false;

const gameRequests = new Map();
let state = new MahjongSoulLiveState();
let liveState = {
  status: "starting",
  adapterPid: process.pid,
  startedAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  position: null,
  prediction: null,
};

const gameplay = createGameplay({
  get automationBlock() {return automationBlock;},
  get positionVersion() {return positionVersion;},
  get autoplay() {return autoplay;},
  get liveState() {return liveState;},
  casual: false,
  canPlay: () => accountVerified(),
  buttonCapturePath, modelUrl, captureControl: io.capture, send, wait, click, output,
  publishLiveState, pauseAutomation: handleGameplayFailure, handleActionMismatch,
});

function publishLiveState(values) {
  liveState = {
    ...liveState,
    ...values,
    controller: "standalone", instanceId: "nejimakidori",
    account, accountVerified: accountVerified(),
    lastHandResult: state.lastHandResult,
    finalMatchResult: state.finalMatchResult,
    automationBlock,
    autoplay: {
      enabled: autoplay,
      friendlyGame: false,
      ready: autoplay && liveState.status === "playing" && Boolean(state.round),
    },
    updatedAt: new Date().toISOString(),
  };
  io.publish(liveState);
}

function output(type, value = {}) {
  process.stdout.write(`${JSON.stringify({ type, ...value })}\n`);
}

function send(method, params = {}, targetSession = browser.sessionId) {
  return browser.send(method, params, targetSession).catch((error) => {
    if (error.message.startsWith(
      "Browser command timed out: Page.captureScreenshot",
    )) {
      const now = Date.now();
      screenshotTimeouts = screenshotTimeouts.filter(
        (timestamp) => now - timestamp < 45000,
      );
      screenshotTimeouts.push(now);
      if (screenshotTimeouts.length >= 3 && !browserRestartScheduled) {
        browserRestartScheduled = true;
        output("browser_restart", {
          reason: "Chrome stopped responding to screenshot requests.",
          timeouts: screenshotTimeouts.length,
        });
        setImmediate(() => process.exit(1));
      }
    }
    throw error;
  });
}

async function captureLobbyScreen() {
  if (autoplay || screenCapturePending || io.capturePending || controlServer.busy ||
      (!["connected", "in_room", "finished", "logged_off"].includes(liveState.status) &&
       !(automationBlock && liveState.status === "playing"))) return;
  screenCapturePending = true;
  try {
    await io.capture("preview");
  } finally {
    screenCapturePending = false;
  }
}

function handleActionMismatch(verification) {
  autoplay = false;
  automationBlock = {reason: "Action mismatch; inspect saved evidence."};
  const mismatch = {id: randomUUID(), at: new Date().toISOString(), gameId,
    ...verification, position: liveState.position,
    prediction: liveState.prediction};
  mismatch.capture = failureCapture.save(mismatch, liveState);
  appendFileSync(`${liveStatePath}.mismatches.jsonl`,
    `${JSON.stringify(mismatch)}\n`, {mode: 0o600});
  publishLiveState({lastActionMismatch: mismatch});
}

function handleGameplayFailure(reason) {
  autoplay = false;
  automationBlock = {reason};
  positionVersion += 1;
  gameplay.expectedAction = null;
  gameplay.expectedButton = null;
  const failure = {id: randomUUID(), at: new Date().toISOString(), gameId,
    reason: reason.slice(0, 600), position: liveState.position};
  failure.capture = failureCapture.save(failure, liveState);
  appendFileSync(`${liveStatePath}.gameplay-failures.jsonl`,
    `${JSON.stringify(failure)}\n`, {mode: 0o600});
  publishLiveState({lastGameplayFailure: failure});
  output("gameplay_failure", {reason: failure.reason, capture: failure.capture});
}

function pauseAutomation(reason) {
  if (automationBlock) return;
  handleGameplayFailure(reason);
}

async function watchOverlays() {
  if (automationBlock) return;
  if (!accountVerified() || !autoplay || liveState.status !== "playing" || overlayWatchPending) {
    return;
  }
  overlayWatchPending = true;
  try {
    const control = await io.capture("overlays");
    if (accountVerified() && autoplay && liveState.status === "playing" && control.overlays.afk) {
      if (afkClicked) {
        if (Date.now() - afkClicked < 5000) return;
        pauseAutomation("The AFK overlay remained after its dismissal click.");
        return;
      }
      afkClicked = Date.now();
      const [x, y] = control.overlays.afk.center;
      await click(x, y);
      output("afk_cleared", { x, y });
    } else {
      afkClicked = false;
    }
  } finally {
    overlayWatchPending = false;
  }
}

function handleGameMessage(message) {
  if (message.method === "Runtime.exceptionThrown") {
    output("browser_exception", {
      text: message.params.exceptionDetails.text,
      description: message.params.exceptionDetails.exception?.description,
    });
    return;
  }
  if (message.method === "Log.entryAdded" &&
      ["error", "warning"].includes(message.params.entry.level)) {
    output("browser_log", {
      level: message.params.entry.level,
      text: message.params.entry.text,
    });
    return;
  }
  if (message.method === "Network.loadingFailed") {
    output("network_failure", {
      error: message.params.errorText,
      url: message.params.requestId,
    });
    return;
  }
  if (
    !["Network.webSocketFrameReceived", "Network.webSocketFrameSent"].includes(message.method) ||
    message.params.response.opcode !== 2
  ) {
    return;
  }
  try {
    const gameResponse = decodeGameResponse(
      message.params.response.payloadData, gameRequests, message.params.requestId,
    );
    if (gameResponse && !gameResponse.errorCode &&
        ["login", "emailLogin", "oauth2Login", "logout"].includes(gameResponse.method)) {
      account = gameResponse.account || null;
      positionVersion += 1;
      gameplay.expectedAction = null;
      publishLiveState({
        status: gameResponse.method === "logout" ? "logged_off" : "connected",
        position: null, prediction: null, roomId: null,
      });
    }
    if (gameResponse?.method === "authGame" && !gameResponse.errorCode) {
      if (gameResponse.selfSeat < 0) throw new Error("Authenticated player has no seat");
      afkClicked = false;
      gameId = gameResponse.gameId;
      state = new MahjongSoulLiveState();
      state.selfSeat = gameResponse.selfSeat;
      state.accountRanks = gameResponse.accountRanks;
      state.players = gameResponse.players;
      state.eastOnly = gameResponse.eastOnly;
      publishLiveState({
        status: "reconnecting", roomId: gameResponse.roomId,
        position: null, prediction: null,
      });
      output("autoplay_authorization", { autoplay });
    }
    if (gameResponse?.method === "syncGame" && !gameResponse.errorCode) {
      let position = null;
      for (const action of gameResponse.actions) {
        position = state.apply(action);
        position.receivedAt = Date.now();
      }
      positionVersion += 1;
      publishLiveState({
        status: state.round ? "playing" : "reconnecting", position,
      });
      if (position?.phase) {
        gameplay.prediction(position, positionVersion);
      }
      output("round_restored", { actions: gameResponse.actions.length });
    }
    if (message.method === "Network.webSocketFrameSent") return;
    const action = decodeNotification(message.params.response.payloadData);
    if (!action) {
      return;
    }
    if (["NotifyAccountLogout", "NotifyAnotherLogin"].includes(action.name)) {
      account = null;
      positionVersion += 1;
      gameplay.expectedAction = null;
      publishLiveState({
        status: "logged_off", position: null, prediction: null, roomId: null,
      });
      output("logged_off", { reason: action.name });
      return;
    }
    if (["NotifyRoomGameStart", "NotifyMatchGameStart"].includes(action.name)) {
      publishLiveState({status: "waiting_for_round"});
      return;
    }
    if (
      ["NotifyGameEndResult", "NotifyGameTerminate"].includes(action.name)
    ) {
      output("terminal", { action });
      if (action.name === "NotifyGameEndResult") {
        state.apply(action);
      }
      positionVersion += 1;
      publishLiveState({ status: "finished", prediction: null });
      return;
    }
    if (!action.name.startsWith("Action")) {
      return;
    }
    if (action.name === "ActionNewRound") {
      gameplay.expectedAction = null;
    }
    gameplay.verifyExpectedAction(action);
    positionVersion += 1;
    const position = state.apply(action);
    position.receivedAt = Date.now();
    output("action", { action });
    if (position.phase) {
      output("position", { position });
    }
    publishLiveState({
      status: position.round ? "playing" : "reconnecting",
      position,
      prediction: null,
    });
    gameplay.prediction(position, positionVersion);
  } catch (error) {
    pauseAutomation(`Game message could not be processed: ${error.message}`);
    output("decode_error", { message: error.message });
  }
}

async function click(x, y, manual = false) {
  if (automationBlock && !manual) throw new Error("Automation is paused; manual help required.");
  if (!manual && account && !accountVerified()) throw new Error("Unexpected live account identity.");
  await send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  await wait(8 + Math.floor(Math.random() * 5));
  if (automationBlock && !manual) throw new Error("Automation is paused; manual help required.");
  await send("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x,
    y,
    button: "left",
    buttons: 1,
    clickCount: 1,
  });
  await wait(16 + Math.floor(Math.random() * 8));
  await send("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x,
    y,
    button: "left",
    buttons: 0,
    clickCount: 1,
  });
}

async function openBrowser() {
  account = null;
  publishLiveState({ status: "starting", stopReason: null });
  state = new MahjongSoulLiveState();
  gameRequests.clear();
  await browser.open({
    chromePath, profilePath, output, ranked: true, unattended: false,
    headless: false,
    url: process.env.MAHJONG_SOUL_URL || "https://mahjongsoul.game.yo-star.com",
  });
  output("ready", { screenshotPath, autoplay });

  publishLiveState({ status: "connected" });
}

async function hibernate(reason = null) {
  account = null;
  positionVersion += 1;
  gameplay.expectedAction = null;
  publishLiveState({
    status: "hibernating", position: null, prediction: null, roomId: null,
    stopReason: reason,
  });
  await io.captureChain;
  await browser.stop();
}

const controlServer = startLiveControl(controlCommand, () => liveState);
publishLiveState({
  status: "hibernating", controls: { joinRoom: false, screenshotIntervalMs: 250 },
});

const screenRefresh = setInterval(() => {
  captureLobbyScreen().catch((error) =>
    output("screen_capture_error", { message: error.message }),
  );
}, 250);
const heartbeat = setInterval(() => {
  publishLiveState({});
  if (!automationBlock && gameplay.expectedAction && gameplay.expectedActionAt !== null && Date.now() - gameplay.expectedActionAt >= 15000) {
    handleGameplayFailure("The attempted move was not acknowledged by the server.");
    return;
  }
  watchOverlays().catch((error) =>
    pauseAutomation(`Overlay controls failed: ${error.message}`),
  );
}, 1500);


async function shutdown() {
  autoplay = false;
  positionVersion += 1;
  clearInterval(heartbeat);
  clearInterval(screenRefresh);
  controlServer.close();
  vision.close();
  browser.close();
  publishLiveState({ status: "offline", controls: {}, prediction: null });
  await io.stateWrite;
  await gameplay.buttonReferenceWrite;
  process.exit(0);
}
process.once("SIGINT", shutdown);
process.once("SIGTERM", shutdown);
process.once("exit", () => {
  browser.close();
  vision.close();
});
async function controlCommand(command) {
  if (command.action === "autoplay" && command.enabled && automationBlock) {
    if (command.resume !== true) throw new Error("Inspect the failure and explicitly resume automation.");
    automationBlock = null;
  }
  if (command.action === "login") {
    if (liveState.status === "hibernating") await openBrowser();
    else if (liveState.status === "logged_off") {
      await hibernate();
      await openBrowser();
    }
    return { status: liveState.status };
  }
  if (command.action === "hibernate") {
    await hibernate();
    return { status: liveState.status };
  }
  if (liveState.status === "hibernating") {
    throw new Error("Browser is closed. Log in to open it.");
  }
  if (command.action === "autoplay") {
    if (typeof command.enabled !== "boolean") {
      throw new TypeError("enabled must be boolean");
    }
    autoplay = command.enabled;
    positionVersion += 1;
    if (autoplay && liveState.status === "playing" && liveState.position?.phase) {
      gameplay.prediction(liveState.position, positionVersion);
    }
    publishLiveState({});
    return { enabled: autoplay };
  }
  if (command.action === "lobby_click") {
    if (!["connected", "in_room", "finished", "logged_off"].includes(liveState.status) &&
        !(automationBlock && liveState.status === "playing")) {
      throw new Error("Mouse control is available outside games only.");
    }
    command = { ...command, action: "click" };
  }
  if (command.action === "click" || command.action === "wheel") {
    if (!Number.isFinite(command.x) || !Number.isFinite(command.y) ||
        command.x < 0 || command.x >= 1280 ||
        command.y < 0 || command.y >= 720) {
      throw new TypeError("x and y must be within the 1280×720 viewport");
    }
    if (command.action === "click") {
      await click(command.x, command.y, true);
      await io.capture("preview");
    } else {
      if (!Number.isFinite(command.deltaY)) {
        throw new TypeError("deltaY must be a finite number");
      }
      await send("Input.dispatchMouseEvent", {
        type: "mouseWheel", x: command.x, y: command.y,
        deltaX: 0, deltaY: command.deltaY,
      });
    }
    return {};
  }
  if (command.action === "type") {
    if (typeof command.value !== "string") {
      throw new TypeError("value must be a string");
    }
    await send("Input.insertText", {text: command.value});
    return {};
  }
  if (command.action === "screenshot") {
    return { control: await io.capture("lobby"), screenshotPath };
  }
  throw new TypeError("Unknown command");
}

await openBrowser();
