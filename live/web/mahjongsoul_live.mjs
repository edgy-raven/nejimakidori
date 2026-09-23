import { FailureCapture } from "./failure_capture.mjs";
import { RankedRuntime } from "./ranked_runtime.mjs";
import { createGameplay } from "./mahjongsoul_gameplay.mjs";
import { BrowserTransport } from "./browser_transport.mjs";
import { once } from "node:events";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { appendFileSync, readFileSync } from "node:fs";
import { MatchNotifier } from "./match_notifier.mjs";
import { rename, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { startLiveControl } from "./live_control.mjs";
import { MahjongSoulLiveState } from "./mahjongsoul_live_state.mjs";
import { Autoqueue } from "./mahjongsoul_ui_controls.mjs";
import { VisionWorker } from "./mahjongsoul_vision.mjs";
import {
  decodeNotification, decodeGameResponse,
} from "./mahjongsoul_protocol.mjs";

// Ranked live account only. Friendly guests have their own entry point.
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
let autoplay = process.env.MAHJONG_SOUL_AUTOPLAY !== "off";
const pythonPath =
  process.env.PYTHON_PATH || "/home/fen/projects/codex_env/qrow/bin/python";
const visionPath = fileURLToPath(
  new URL("../../mahjongsoul.py", import.meta.url),
);
const vision = new VisionWorker(pythonPath, visionPath);
const buttonCapturePath = process.env.MAHJONG_SOUL_BUTTON_CAPTURES ||
  process.env.HOME + "/.local/state/nejimakidori/button-captures";
const failureCapture = new FailureCapture(`${liveStatePath}.failures`);
let chrome;
let scheduledSleep = process.env.MAHJONG_SOUL_AUTOSTART === "on";

const browser = new BrowserTransport(handleGameMessage);
let sessionId;
let privateGame = false;
let positionVersion = 0;
let overlayWatchPending = false;
let roomWatchPending = false;
let controlVersion = 0;
let captureChain = Promise.resolve();
let capturePending = 0;
let lastPreviewTimingAt = -Infinity;
let screenCapturePending = false;
let decisionChain = Promise.resolve();
const autoqueue = new Autoqueue();
const runtime = new RankedRuntime(process.env.MAHJONG_SOUL_RUNTIME_STATE ||
  `${liveStatePath}.runtime.json`);
autoplay = runtime.restore(autoqueue, autoplay);
const matchNotifier = process.env.MAHJONG_SOUL_DISCORD_CONFIG
  ? new MatchNotifier(JSON.parse(readFileSync(process.env.MAHJONG_SOUL_DISCORD_CONFIG, "utf8")))
  : null;
let gameId = null;
let automationBlock = matchNotifier?.attention || null;
let afkClicked = false;
if (automationBlock) autoplay = false;
const gameRequests = new Map();
let state = new MahjongSoulLiveState();
const liveStateTemporaryPath = `${liveStatePath}.${process.pid}.tmp`;
let liveState = {
  status: "starting",
  adapterPid: process.pid,
  startedAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  position: null,
  prediction: null,
};
let liveStateWrite = Promise.resolve();

const gameplay = createGameplay({
  get automationBlock() {return automationBlock;},
  get positionVersion() {return positionVersion;},
  get autoplay() {return autoplay;},
  get liveState() {return liveState;},
  casual: false,
  canPlay: () => accountVerified() && !privateGame,
  buttonCapturePath, modelUrl, captureControl, send, wait, click, output,
  publishLiveState, pauseAutomation: handleGameplayFailure, handleActionMismatch,
});

function publishLiveState(values) {
  if (values.status !== "offline") runtime.save(autoplay, autoqueue);
  liveState = {
    ...liveState,
    ...values,
    controller: "ranked", instanceId: "nejimakidori", friendlyOnly: false,
    account, accountVerified: accountVerified(),
    lastHandResult: state.lastHandResult,
    finalMatchResult: state.finalMatchResult,
    autoqueue: autoqueue.snapshot,
    discord: matchNotifier?.snapshot || {enabled: false},
    automationBlock,
    autoplay: {
      enabled: autoplay,
      friendlyGame: false,
      ready: autoplay && liveState.status === "playing" && Boolean(state.round),
    },
    updatedAt: new Date().toISOString(),
  };
  const body = JSON.stringify(liveState);
  liveStateWrite = liveStateWrite
    .then(() => writeFile(liveStateTemporaryPath, body))
    .then(() => rename(liveStateTemporaryPath, liveStatePath))
    .catch((error) => {
      process.stderr.write(`live state write failed: ${error.message}\n`);
    });
}

function output(type, value = {}) {
  process.stdout.write(`${JSON.stringify({ type, ...value })}\n`);
}

function send(method, params = {}, targetSession = sessionId) {
  return browser.send(method, params, targetSession);
}

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function captureLobbyScreen() {
  if (screenCapturePending || capturePending || controlServer.busy ||
      (!["connected", "in_room", "finished", "logged_off"].includes(liveState.status) &&
       !(automationBlock && liveState.status === "playing"))) return;
  screenCapturePending = true;
  try {
    await captureControl("preview");
  } finally {
    screenCapturePending = false;
  }
}

function captureControl(mode = "control", position = null) {
  const queuedAt = performance.now();
  capturePending += 1;
  const capture = captureChain.then(async () => {
    const started = performance.now();
    const preview = mode === "preview";
    const result = await send("Page.captureScreenshot", {
      format: preview ? "jpeg" : "png",
      // Scaled clips resize the game's canvas and invalidate click positions.
      ...(preview ? { quality: 60 } : {}),
      captureBeyondViewport: false,
    });
    const capturedAt = performance.now();
    const image = Buffer.from(result.data, "base64");
    failureCapture.observe(mode, image);
    const path = preview ? `${screenshotPath}.jpg` : screenshotPath;
    await writeFile(`${path}.tmp`, image);
    await rename(`${path}.tmp`, path);
    const savedAt = performance.now();
    const detection = preview ? {width: 1280, height: 720} : await vision.analyze(path, mode, position ? {
      hand: position.hand, buttons: position.buttons,
    } : null);
    failureCapture.observe(mode, image, detection);
    const finishedAt = performance.now();
    if (!preview || finishedAt - lastPreviewTimingAt >= 30000) {
      output("capture_timing", {
        mode, queue_ms: started - queuedAt,
        screenshot_ms: capturedAt - started, save_ms: savedAt - capturedAt,
        vision_ms: finishedAt - savedAt, total_ms: finishedAt - queuedAt,
      });
      if (preview) lastPreviewTimingAt = finishedAt;
    }
    return {
      ...detection,
      ...(mode === "control" ? { image } : {}),
    };
  }).finally(() => {
    capturePending -= 1;
  });
  captureChain = capture.catch(() => {});
  return capture;
}

function handleActionMismatch(verification) {
  autoqueue.stop("Action mismatch saved; requeue disabled.");
  const mismatch = {id: randomUUID(), at: new Date().toISOString(), gameId,
    ...verification, position: liveState.position,
    prediction: liveState.prediction};
  mismatch.capture = failureCapture.save(mismatch, liveState);
  appendFileSync(`${liveStatePath}.mismatches.jsonl`,
    `${JSON.stringify(mismatch)}\n`, {mode: 0o600});
  publishLiveState({lastActionMismatch: mismatch});
}

function handleGameplayFailure(reason) {
  autoqueue.stop("Gameplay failure saved; requeue disabled.");
  positionVersion += 1;
  gameplay.expectedAction = null;
  gameplay.expectedButton = null;
  const failure = {id: randomUUID(), at: new Date().toISOString(), gameId,
    reason: reason.slice(0, 600), position: liveState.position};
  failure.capture = failureCapture.save(failure, liveState);
  appendFileSync(`${liveStatePath}.gameplay-failures.jsonl`,
    `${JSON.stringify(failure)}\n`, {mode: 0o600});
  publishLiveState({lastGameplayFailure: failure});
  matchNotifier?.gameplayFailure(failure, autoqueue.account?.nickname || "Live bot");
  output("gameplay_failure", {reason: failure.reason, capture: failure.capture});
}

function pauseAutomation(reason) {
  if (liveState.status === "playing" && !automationBlock) {
    handleGameplayFailure(reason);
    return;
  }
  if (automationBlock) {
    if (automationBlock.returnToLobby && liveState.status === "finished") {
      automationBlock.returnToLobby = false;
      automationBlock.returnError = reason.slice(0, 600);
      automationBlock.returnCapture = failureCapture.save(
        {id: randomUUID(), reason, status: liveState.status}, liveState);
      matchNotifier?.save();
      publishLiveState({});
      output("return_to_lobby_stopped", {reason});
    }
    return;
  }
  automationBlock = {id: randomUUID(), reason: reason.slice(0, 600),
    status: liveState.status, at: new Date().toISOString(),
    returnToLobby: liveState.status === "playing",
    settings: {autoplay, autoqueue: autoqueue.enabled}};
  autoplay = false;
  autoqueue.stop(automationBlock.reason);
  positionVersion += 1;
  controlVersion += 1;
  gameplay.expectedAction = null;
  gameplay.expectedButton = null;
  automationBlock.capture = failureCapture.save(automationBlock, liveState);
  publishLiveState({});
  matchNotifier?.requestHelp(automationBlock, autoqueue.account?.nickname || "Live bot");
  output("automation_paused", {reason: automationBlock.reason});
}

async function watchRankedLobby() {
  if (automationBlock && (!automationBlock.returnToLobby ||
      liveState.status !== "finished")) return;
  if (!accountVerified()) return;
  if (liveState.status === "playing") autoqueue.next("playing", []);
  if (!["connected", "finished", "logged_off"].includes(liveState.status)) {
    return;
  }
  if (controlServer.busy || roomWatchPending) return;
  roomWatchPending = true;
  let controlsLocked = false;
  const status = liveState.status;
  const version = controlVersion;
  try {
    const control = await captureControl("ranked");
    if (controlServer.busy || controlVersion !== version ||
        liveState.status !== status) return;
    if (control.error) {
      pauseAutomation(control.error.message);
      return;
    }
    if (autoqueue.enabled && autoqueue.fresh && liveState.status === "connected" &&
        autoqueue.snapshot.blockedReason) {
      pauseAutomation(autoqueue.snapshot.blockedReason);
      return;
    }
    if (status === "finished" && control.screen === "home") {
      state.finalMatchResult = null;
      if (automationBlock) {
        automationBlock.returnToLobby = false;
        matchNotifier?.save();
      }
      publishLiveState({status: "connected", roomId: null, position: null});
    }
    if (!automationBlock && control.screen === "home" &&
        !autoqueue.pending && autoqueue.shouldIdleBrowser()) {
      controlServer.busy = controlsLocked = true;
      scheduledSleep = true;
      autoqueue.next("connected", [], Date.now(), control);
      await hibernate("Waiting for queue hours or the scheduled break to end.");
      return;
    }
    if (["finished", "connected", "in_room"].includes(status)) {
      const next = autoqueue.next(liveState.status, control.targets, Date.now(), control);
      if (next) {
        controlServer.busy = controlsLocked = true;
        if (next.control === "refresh account") {
          publishLiveState({status: "reconnecting", position: null, prediction: null});
          await send("Page.reload", {});
        } else if (next.control === "scroll rooms") {
          await scrollRankedRooms(next.deltaY);
        } else {
          // Keep the click near the detected center, inside its text box.
          await click(...next.center.map((value, axis) => value +
            (Math.random() * 2 - 1) * Math.min(3,
              next.box ? next.box[axis + 2] / 4 : 3)),
            false, Boolean(automationBlock));
        }
        output("postgame_clicked", { control: next.control });
      }
      // With no result controls, check whether the client returned to lobby.
      if (control.targets.length) return;
    }
  } finally {
    if (controlsLocked) controlServer.busy = false;
    roomWatchPending = false;
    publishLiveState({});
  }
}

async function watchOverlays() {
  if (automationBlock) return;
  if (!accountVerified() || privateGame || !autoplay || liveState.status !== "playing" || overlayWatchPending) {
    return;
  }
  overlayWatchPending = true;
  try {
    const control = await captureControl("overlays");
    if (accountVerified() && !privateGame && autoplay && liveState.status === "playing" && control.overlays.afk) {
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
      autoqueue.updateAccount(gameResponse.account || null);
      autoqueue.pending = false;
      positionVersion += 1;
      gameplay.expectedAction = null;
      privateGame = false;
      publishLiveState({
        status: gameResponse.method === "logout" ? "logged_off" : "connected",
        position: null, prediction: null, roomId: null,
      });
    }
    if (gameResponse?.method === "fetchAccountInfo" && !gameResponse.errorCode) {
      autoqueue.account = gameResponse.account;
      matchNotifier?.updateAccount(gameId, autoqueue.account);
      publishLiveState({});
    }
    if (gameResponse?.method === "matchGame") {
      if (gameResponse.errorCode) pauseAutomation(
        `Matchmaking rejected by the server (${gameResponse.errorCode}).`);
      else {
        autoqueue.pending = true;
        autoqueue.acknowledged = true;
      }
      publishLiveState({});
    }
    if (gameResponse?.method === "cancelMatch" && !gameResponse.errorCode) {
      autoqueue.stop("Matchmaking cancelled.");
      publishLiveState({});
    }
    if (gameResponse?.method === "authGame" && !gameResponse.errorCode) {
      if (gameResponse.selfSeat < 0) throw new Error("Authenticated player has no seat");
      afkClicked = false;
      gameId = gameResponse.gameId;
      if (typeof matchNotifier !== "undefined") {
        matchNotifier?.begin(gameId, autoqueue.account);
      }
      autoqueue.currentModeId = gameResponse.modeId;
      autoqueue.pending = false;
      state = new MahjongSoulLiveState();
      state.selfSeat = gameResponse.selfSeat;
      state.accountRanks = gameResponse.accountRanks;
      state.players = gameResponse.players;
      state.eastOnly = gameResponse.eastOnly;
      privateGame = gameResponse.roomId > 0;
      if (privateGame) {
        pauseAutomation("A friendly game was opened in the ranked vision.");
        return;
      }
      publishLiveState({
        status: "reconnecting", roomId: gameResponse.roomId,
        position: null, prediction: null,
      });
      output("autoplay_authorization", { privateGame, autoplay });
    }
    if (gameResponse?.method === "syncGame" && !gameResponse.errorCode) {
      let position = null;
      for (const action of gameResponse.actions) {
        position = state.apply(action);
        position.receivedAt = Date.now();
      }
      positionVersion += 1;
      const version = positionVersion;
      publishLiveState({
        status: state.round ? "playing" : "reconnecting", position,
      });
      if (position?.phase) {
        decisionChain = decisionChain.then(
          () => gameplay.prediction(position, version),
        ).catch((error) =>
          handleGameplayFailure(`Prediction failed: ${error.message}`),
        );
      }
      output("round_restored", { actions: gameResponse.actions.length });
    }
    if (["joinRoom", "fetchRoom"].includes(gameResponse?.method) && gameResponse.roomId) {
      pauseAutomation("A friendly room was opened in the ranked vision.");
      return;
    }
    if (message.method === "Network.webSocketFrameSent") return;
    const action = decodeNotification(message.params.response.payloadData);
    if (!action) {
      return;
    }
    if (action.name === "NotifyMatchTimeout") pauseAutomation("Matchmaking timed out.");
    autoqueue.notification(action);
    if (["NotifyAccountUpdate", "NotifyAccountLevelChange"].includes(action.name)) {
      matchNotifier?.updateAccount(gameId, autoqueue.account);
    }
    if (action.name.startsWith("NotifyAccount") || action.name === "NotifyMatchTimeout") publishLiveState({});
    if (["NotifyAccountLogout", "NotifyAnotherLogin"].includes(action.name)) {
      account = null;
      autoqueue.updateAccount(null);
      positionVersion += 1;
      gameplay.expectedAction = null;
      privateGame = false;
      publishLiveState({
        status: "logged_off", position: null, prediction: null, roomId: null,
      });
      output("logged_off", { reason: action.name });
      return;
    }
    if (action.name === "NotifyRoomGameStart") {
      privateGame = true;
      pauseAutomation("A friendly game was opened in the ranked vision.");
      return;
    }
    if (action.name === "NotifyMatchGameStart") {
      privateGame = false;
      publishLiveState({ status: "waiting_for_round" });
      output("autoplay_authorization", { privateGame, autoplay });
      return;
    }
    if (
      ["NotifyGameEndResult", "NotifyGameTerminate"].includes(action.name)
    ) {
      autoqueue.finish(gameId);
      output("terminal", { action });
      if (action.name === "NotifyGameEndResult") {
        state.apply(action);
      }
      positionVersion += 1;
      publishLiveState({ status: "finished", prediction: null });
      if (action.name === "NotifyGameEndResult") {
        matchNotifier?.finish(gameId, autoqueue.account, state.finalMatchResult);
      }
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
    const version = positionVersion;
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
    decisionChain = decisionChain.then(
      () => gameplay.prediction(position, version),
    ).catch((error) =>
      handleGameplayFailure(`Prediction failed: ${error.message}`),
    );
  } catch (error) {
    pauseAutomation(`Game message could not be processed: ${error.message}`);
    output("decode_error", { message: error.message });
  }
}

async function click(x, y, manual = false, exitOnly = false) {
  if (automationBlock && !manual && !(exitOnly &&
      automationBlock.returnToLobby && liveState.status === "finished" &&
      !autoqueue.enabled)) throw new Error("Automation is paused; manual help required.");
  if (!manual && account && !accountVerified()) throw new Error("Unexpected live account identity.");
  await send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  await wait(8 + Math.floor(Math.random() * 5));
  if (automationBlock && !manual && !(exitOnly &&
      automationBlock.returnToLobby && liveState.status === "finished" &&
      !autoqueue.enabled)) throw new Error("Automation is paused; manual help required.");
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

async function scrollRankedRooms(deltaY) {
  if (automationBlock) throw new Error("Automation is paused; manual help required.");
  const startY = deltaY > 0 ? 540 : 270;
  const endY = deltaY > 0 ? 270 : 540;
  await send("Input.dispatchMouseEvent", {type: "mouseMoved", x: 960, y: startY});
  await send("Input.dispatchMouseEvent", {type: "mousePressed", x: 960,
    y: startY, button: "left", buttons: 1, clickCount: 1});
  for (let step = 1; step <= 12 && !automationBlock; step += 1) {
    await send("Input.dispatchMouseEvent", {type: "mouseMoved", x: 960,
      y: startY + (endY - startY) * step / 12, buttons: 1});
    await wait(20);
  }
  await send("Input.dispatchMouseEvent", {type: "mouseReleased", x: 960,
    y: endY, button: "left", buttons: 0, clickCount: 1});
}

async function typeText(value) {
  await send("Input.insertText", { text: value });
}

async function openBrowser() {
  scheduledSleep = false;
  account = null;
  autoqueue.updateAccount(null);
  publishLiveState({ status: "starting", stopReason: null });
  sessionId = undefined;
  state = new MahjongSoulLiveState();
  gameRequests.clear();
  chrome = spawn(
    chromePath,
    [
      "--headless=new",
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--enable-webgl",
      "--enable-unsafe-swiftshader",
      "--autoplay-policy=no-user-gesture-required",
      "--ignore-gpu-blocklist",
      "--use-angle=swiftshader",
      "--remote-debugging-pipe",
      `--user-data-dir=${profilePath}`,
      "--window-size=1280,720",
      "about:blank",
    ],
    { stdio: ["ignore", "ignore", "ignore", "pipe", "pipe"] },
  );

  browser.attach(chrome);
  const { targetId } = await send("Target.createTarget", {
    url: "about:blank",
  }, undefined);
  output("startup", { stage: "target_created" });
  ({ sessionId } = await send(
    "Target.attachToTarget",
    { targetId, flatten: true },
    undefined,
  ));
  output("startup", { stage: "target_attached" });
  await send("Page.enable");
  output("startup", { stage: "page_enabled" });
  await send("Network.enable");
  output("startup", { stage: "network_enabled" });
  await send("Runtime.enable");
  await send("Log.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1280,
    height: 720,
    deviceScaleFactor: 1,
    mobile: false,
  });
  output("startup", { stage: "metrics_configured" });
  await send("Page.navigate", {
    url: process.env.MAHJONG_SOUL_URL ||
      "https://mahjongsoul.game.yo-star.com",
  });
  output("ready", { screenshotPath, autoplay });

  publishLiveState({ status: "connected" });
}

async function hibernate(reason = null) {
  account = null;
  autoqueue.updateAccount(null);
  positionVersion += 1;
  gameplay.expectedAction = null;
  privateGame = false;
  publishLiveState({
    status: "hibernating", position: null, prediction: null, roomId: null,
    stopReason: reason,
  });
  await captureChain;
  if (chrome) {
    const exited = once(chrome, "exit");
    chrome.kill("SIGTERM");
    await exited;
    chrome = undefined;
  }
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
  matchNotifier?.flush().catch(error => output("discord_report_error", {message: error.message}));
  publishLiveState({});
  if (!automationBlock && gameplay.expectedAction && gameplay.expectedActionAt !== null && Date.now() - gameplay.expectedActionAt >= 15000) {
    handleGameplayFailure("The attempted move was not acknowledged by the server.");
    return;
  }
  if (scheduledSleep && liveState.status === "hibernating" &&
      !controlServer.busy && !roomWatchPending && !automationBlock &&
      autoqueue.enabled && !autoqueue.shouldIdleBrowser()) {
    openBrowser().catch(error => pauseAutomation(`Login failed: ${error.message}`));
    return;
  }
  watchRankedLobby().catch((error) =>
    pauseAutomation(`Room controls failed: ${error.message}`),
  );
  watchOverlays().catch((error) =>
    pauseAutomation(`Overlay controls failed: ${error.message}`),
  );
}, 1500);

if (scheduledSleep && (automationBlock ||
    (autoqueue.enabled && !autoqueue.shouldIdleBrowser()))) {
  openBrowser().catch(error => pauseAutomation(`Login failed: ${error.message}`));
}

async function shutdown() {
  autoplay = false;
  positionVersion += 1;
  clearInterval(heartbeat);
  clearInterval(screenRefresh);
  controlServer.close();
  chrome?.kill("SIGTERM");
  vision.close();
  browser.close();
  publishLiveState({ status: "offline", controls: {}, prediction: null });
  await liveStateWrite;
  await gameplay.buttonReferenceWrite;
  process.exit(0);
}
process.once("SIGINT", shutdown);
process.once("SIGTERM", shutdown);
process.once("exit", () => {
  chrome?.kill("SIGTERM");
  vision.close();
});
async function controlCommand(command) {
  controlVersion += 1;
  if (command.action === "autoqueue" && command.resume === true) {
    if (!automationBlock) throw new Error("Automation is not paused.");
    if (liveState.status === "hibernating") throw new Error("Log in before resuming automation.");
    const settings = automationBlock.settings;
    matchNotifier?.resolveHelp(automationBlock.id);
    automationBlock = null;
    afkClicked = false;
    autoqueue.error = null;
    autoqueue.pending = false;
    autoqueue.observed = null;
    autoqueue.clickedControl = null;
    autoqueue.clickedAt = -Infinity;
    autoqueue.stalledSince = null;
    autoqueue.enabled = settings.autoqueue;
    command = {action: "autoplay", enabled: settings.autoplay};
  } else if (automationBlock && (["ranked_scroll"].includes(command.action) ||
      (["autoplay", "autoqueue"].includes(command.action) && command.enabled))) {
    throw new Error("Fix the issue, then explicitly Resume automation.");
  }
  if (["summon", "join", "room_monitor"].includes(command.action)) {
    throw new Error("Friendly-room commands require the friendly vision.");
  }
  if (command.action === "autoqueue") {
    if (typeof command.enabled !== "boolean") {
      throw new TypeError("enabled must be boolean");
    }
    if (command.enabled && autoqueue.error) {
      autoqueue.error = null;
      autoqueue.fresh = false;
    }
    autoqueue.observed = null;
    autoqueue.enabled = command.enabled;
    publishLiveState({});
    return { enabled: autoqueue.enabled };
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
    scheduledSleep = false;
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
      const position = liveState.position;
      const version = positionVersion;
      decisionChain = decisionChain.then(
        () => gameplay.prediction(position, version),
      ).catch((error) =>
        handleGameplayFailure(`Prediction failed: ${error.message}`),
      );
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
      await captureControl("preview");
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
  if (command.action === "ranked_scroll") {
    if (!["connected", "finished"].includes(liveState.status) ||
        (await captureControl("ranked")).screen !== "ranked rooms") {
      throw new Error("Ranked room scrolling requires the room list.");
    }
    if (!["up", "down"].includes(command.direction)) {
      throw new TypeError("direction must be up or down");
    }
    await scrollRankedRooms(command.direction === "up" ? -360 : 360);
    return {};
  }
  if (command.action === "type") {
    if (typeof command.value !== "string") {
      throw new TypeError("value must be a string");
    }
    await typeText(command.value);
    return {};
  }
  if (command.action === "screenshot") {
    return { control: await captureControl("lobby"), screenshotPath };
  }
  throw new TypeError("Unknown command");
}
