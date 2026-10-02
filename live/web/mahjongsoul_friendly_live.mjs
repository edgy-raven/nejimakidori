import { existsSync, readFileSync } from "node:fs";
import { FailureCapture } from "./failure_capture.mjs";
import { createGameplay } from "./mahjongsoul_gameplay.mjs";
import { BrowserTransport } from "./browser_transport.mjs";
import { ControllerIO } from "./controller_io.mjs";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { setTimeout as wait } from "node:timers/promises";

import { startLiveControl } from "./live_control.mjs";
import { MahjongSoulLiveState } from "./mahjongsoul_live_state.mjs";
import { FriendlySession } from "./mahjongsoul_friendly.mjs";
import { ResultControls } from "./mahjongsoul_result_controls.mjs";
import { VisionWorker } from "./mahjongsoul_vision.mjs";
import {
  decodeNotification, decodeGameResponse,
} from "./mahjongsoul_protocol.mjs";

// Private-room guests and the reviewer use this entry point exclusively.
const friendly = new FriendlySession({...process.env, MAHJONG_SOUL_FRIENDLY_ONLY: "on"});
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
let lastActivityAt = Date.now();
const idleTimeoutMs = Number(process.env.MAHJONG_SOUL_IDLE_MS || 600000);

const browser = new BrowserTransport(handleGameMessage);
const send = browser.send.bind(browser);
const io = new ControllerIO({
  send, vision, failureCapture, screenshotPath, liveStatePath, output,
  resetViewport: true,
});
let friendlyRoom = false;
let positionVersion = 0;
let overlayWatchPending = false;
let roomWatchPending = false;
let controlVersion = 0;
let screenCapturePending = false;
let joiningRoom = null;
const postgame = new ResultControls();
let automationBlock = existsSync(liveStatePath)
  ? JSON.parse(readFileSync(liveStatePath, "utf8")).automationBlock : null;
if (automationBlock) autoplay = false;
let afkClicked = false;
let readyClickedAt = null;
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
  casual: true,
  riichiOnly: process.env.MAHJONG_SOUL_RIICHI_ONLY === "on",
  canPlay: () => friendly.canPlay(friendlyRoom) && !friendly.voteActive(),
  buttonCapturePath, modelUrl, captureControl: io.capture, send, wait, click, output,
  publishLiveState, pauseAutomation,
  handleActionMismatch: () => pauseAutomation(
    "The server action did not match the attempted move."),
});

function publishLiveState(values) {
  liveState = {
    ...liveState,
    ...values,
    controller: "friendly", instanceId: friendly.instanceId,
    friendlyOnly: friendly.friendlyOnly,
    account: friendly.account,
    accountVerified: friendly.accountVerified,
    ready: friendly.ready,
    cancelVote: friendly.cancelVote,
    roomPlayers: friendly.roomPlayers,
    lastHandResult: state.lastHandResult,
    finalMatchResult: state.finalMatchResult,
    automationBlock,
    autoplay: {
      enabled: autoplay,
      friendlyGame: friendlyRoom,
      ready: autoplay && liveState.status === "playing" && Boolean(state.round),
    },
    updatedAt: new Date().toISOString(),
  };
  io.publish(liveState);
}

function output(type, value = {}) {
  process.stdout.write(`${JSON.stringify({ type, ...value })}\n`);
}

async function captureLobbyScreen() {
  if (screenCapturePending || io.capturePending || controlServer.busy ||
      (!["connected", "in_room", "finished", "logged_off"].includes(liveState.status) &&
       !(automationBlock && liveState.status === "playing"))) return;
  screenCapturePending = true;
  try {
    await io.capture("preview");
  } finally {
    screenCapturePending = false;
  }
}

async function requireLobbyScreen(screen, timeoutMs = 0) {
  let control;
  const deadline = Date.now() + timeoutMs;
  do {
    control = await io.capture("lobby");
    if (control.screen === screen) break;
    if (timeoutMs) await wait(100);
  } while (Date.now() < deadline);
  if (!["connected", "finished"].includes(liveState.status)) {
    throw new Error("Cannot join while a game is active or starting.");
  }
  if (control.screen !== screen) {
    throw new Error(`Expected ${screen}; found ${control.screen}. ` +
      "Return to the home lobby manually before joining.");
  }
}

async function joinRoom(roomId) {
  friendly.requireIdentity();
  if (!Number.isInteger(roomId) || roomId < 1 || roomId > 999999) {
    throw new TypeError("Enter a numeric room ID (1–999999).");
  }
  if (!["connected", "finished"].includes(liveState.status)) {
    throw new Error("Cannot join while a game is active or starting.");
  }
  if (joiningRoom) throw new Error("A room join is already in progress.");
  joiningRoom = { response: null };
  try {
    await requireLobbyScreen("home");
    await click(935, 504);
    await requireLobbyScreen("friendly_menu", 2000);
    await click(935, 390);
    await requireLobbyScreen("join_dialog", 2000);
    await click(467, 490);
    await wait(400);
    for (const digit of String(roomId)) {
      const value = Number(digit);
      await click(
        value ? 467 + ((value - 1) % 3) * 111 : 578,
        value ? 310 + Math.floor((value - 1) / 3) * 60 : 490,
      );
      // The keypad drops fast consecutive presses, especially repeated digits.
      await wait(400);
    }
    await requireLobbyScreen("join_dialog");
    await click(580, 572);
    for (let attempt = 0; attempt < 80 && !joiningRoom.response; attempt += 1) {
      await wait(100);
    }
    if (!joiningRoom.response) {
      throw new Error("Room join was not confirmed by Mahjong Soul.");
    }
    if (joiningRoom.response.errorCode) {
      throw new Error(`Mahjong Soul rejected the room: ${joiningRoom.response.errorCode}`);
    }
    if (joiningRoom.response.roomId !== roomId) {
      throw new Error("Joined room does not match the requested room ID.");
    }
    publishLiveState({ status: "in_room", roomId, position: null, prediction: null });
    return { roomId };
  } finally {
    joiningRoom = null;
  }
}

function pauseAutomation(reason) {
  if (automationBlock) return;
  automationBlock = {id: randomUUID(), reason: reason.slice(0, 600),
    status: liveState.status, at: new Date().toISOString(),
    settings: {autoplay}};
  autoplay = false;
  positionVersion += 1;
  controlVersion += 1;
  gameplay.expectedAction = null;
  gameplay.expectedButton = null;
  automationBlock.capture = failureCapture.save(automationBlock, liveState);
  publishLiveState({});
  output("automation_paused", {reason: automationBlock.reason});
}

async function watchRoom() {
  if (automationBlock || !friendly.accountVerified) return;
  if (liveState.status === "playing") postgame.reset();
  if (!["connected", "in_room", "finished"].includes(liveState.status)) return;
  if (controlServer.busy || roomWatchPending) return;
  roomWatchPending = true;
  let controlsLocked = false;
  const status = liveState.status;
  const version = controlVersion;
  try {
    const control = await io.capture("autoqueue");
    if (controlServer.busy || controlVersion !== version || liveState.status !== status) return;
    if (control.error) {
      pauseAutomation(control.error.message);
      return;
    }
    const next = postgame.next(status, control.targets);
    if (next) {
      controlServer.busy = controlsLocked = true;
      await click(...next.center);
      lastActivityAt = Date.now();
      output("postgame_clicked", {control: next.control});
    }
    if (control.targets.length) return;
    const room = await io.capture("room");
    if (controlServer.busy || controlVersion !== version || liveState.status !== status) return;
    if (status === "finished" && (room.in_room ||
        (!room.match_end_confirm && room.screen === "home"))) {
      state.finalMatchResult = null;
      publishLiveState({status: room.in_room ? "in_room" : "connected",
        roomId: room.in_room ? liveState.roomId : null, position: null});
    }
    if (room.in_room && liveState.status === "in_room" &&
        friendly.roomAction(liveState) === "ready") {
      if (readyClickedAt !== null) {
        if (Date.now() - readyClickedAt >= 5000) pauseAutomation("Ready was not acknowledged.");
        return;
      }
      const readyControl = await io.capture("ready");
      const target = readyControl.targets.find(target => target.control === "ready");
      if (target && !controlServer.busy && controlVersion === version &&
          liveState.status === "in_room" && friendly.roomAction(liveState) === "ready") {
        controlServer.busy = controlsLocked = true;
        readyClickedAt = Date.now();
        await click(...target.center);
        output("friendly_ready_clicked", {roomId: liveState.roomId});
      }
    }
  } finally {
    if (controlsLocked) controlServer.busy = false;
    roomWatchPending = false;
    publishLiveState({});
  }
}

async function watchFriendlyOverlays() {
  if (automationBlock) return;
  if (!friendly.canPlay(friendlyRoom) || (!autoplay && !friendly.shouldVoteYes()) || liveState.status !== "playing" || overlayWatchPending) {
    return;
  }
  overlayWatchPending = true;
  try {
    if (friendly.voteActive()) {
      if (controlServer.busy || !friendly.shouldVoteYes()) return;
      const version = controlVersion;
      const vote = await io.capture("cancel_vote");
      const target = vote.targets.find(target => ["yes", "agree"].includes(target.control));
      if (target && !controlServer.busy && controlVersion === version &&
          friendly.canPlay(friendlyRoom) && liveState.status === "playing" &&
          friendly.shouldVoteYes()) {
        if (Number.isFinite(friendly.voteClickedAt)) {
          pauseAutomation("The vote button remained after its confirmation click.");
          return;
        }
        controlServer.busy = true;
        try {
          if (!await click(...target.center, false, true)) return;
          friendly.voteClickedAt = Date.now();
          output("cancel_vote_yes_clicked", {control: target.control});
        } finally {
          controlServer.busy = false;
        }
      }
      return;
    }
    const control = await friendly.captureOverlays(io.capture);
    if (!control) return;
    if (friendly.canPlay(friendlyRoom) && autoplay && liveState.status === "playing" && control.overlays.afk) {
      if (afkClicked) {
        if (Date.now() - afkClicked < 5000) return;
        pauseAutomation("The AFK overlay remained after its dismissal click.");
        return;
      }
      const [x, y] = control.overlays.afk.center;
      if (!await click(x, y)) return;
      afkClicked = Date.now();
      output("afk_cleared", { x, y });
    } else {
      afkClicked = false;
    }
  } finally {
    overlayWatchPending = false;
  }
}

function handleGameMessage(message) {
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
      friendly.account = gameResponse.account || null;
      friendly.ready = false;
      positionVersion += 1;
      gameplay.expectedAction = null;
      friendlyRoom = false;
      publishLiveState({
        status: gameResponse.method === "logout" ? "logged_off" : "connected",
        position: null, prediction: null, roomId: null,
      });
    }
    if (gameResponse?.method === "authGame" && !gameResponse.errorCode) {
      if (gameResponse.selfSeat < 0) throw new Error("Authenticated player has no seat");
      readyClickedAt = null;
      afkClicked = false;
      postgame.reset();
      state = new MahjongSoulLiveState();
      state.selfSeat = gameResponse.selfSeat;
      state.accountRanks = gameResponse.accountRanks;
      state.players = gameResponse.players;
      state.eastOnly = gameResponse.eastOnly;
      friendlyRoom = gameResponse.roomId > 0;
      if (!friendlyRoom) {
        pauseAutomation("Ranked play is not supported by the friendly vision.");
        return;
      }
      publishLiveState({
        status: "reconnecting", roomId: gameResponse.roomId,
        position: null, prediction: null,
      });
      output("autoplay_authorization", { friendlyRoom, autoplay });
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
    if (gameResponse?.method === "leaveRoom" && !gameResponse.errorCode) {
      friendly.ready = false;
      friendly.roomPlayers = null;
      friendlyRoom = false;
      positionVersion += 1;
      gameplay.expectedAction = null;
      publishLiveState({
        status: "connected", roomId: null, position: null, prediction: null,
      });
    }
    if (gameResponse?.method === "readyPlay" && !gameResponse.errorCode) {
      friendly.ready = gameResponse.ready;
      publishLiveState({});
    }
    const roomResponse = ["joinRoom", "fetchRoom"].includes(gameResponse?.method)
      ? gameResponse : null;
    if (roomResponse?.method === "joinRoom" && joiningRoom) joiningRoom.response = roomResponse;
    if (roomResponse && !roomResponse.errorCode && roomResponse.roomId) {
      readyClickedAt = null;
      friendly.roomPlayers = roomResponse.accountIds;
      friendly.ready = false;
      publishLiveState({
        status: "in_room", roomId: roomResponse.roomId,
        position: null, prediction: null,
      });
    }
    if (message.method === "Network.webSocketFrameSent") return;
    const action = decodeNotification(message.params.response.payloadData);
    if (!action) {
      return;
    }
    if (action.name.startsWith("Action") ||
        ["NotifyGameEndResult", "NotifyGameTerminate"].includes(action.name)) {
      lastActivityAt = Date.now();
    }
    if (["NotifyAccountLogout", "NotifyAnotherLogin"].includes(action.name)) {
      friendly.roomPlayers = null;
      friendly.account = null;
      friendly.ready = false;
      positionVersion += 1;
      gameplay.expectedAction = null;
      friendlyRoom = false;
      publishLiveState({
        status: "logged_off", position: null, prediction: null, roomId: null,
      });
      output("logged_off", { reason: action.name });
      return;
    }
    if (action.name === "NotifyRoomPlayerUpdate") {
      friendly.updateRoomPlayers(action.data);
      publishLiveState({});
      return;
    }
    if (action.name === "NotifyRoomGameStart") {
      friendly.cancelVote = null;
      friendly.ready = false;
      friendlyRoom = true;
      publishLiveState({ status: "waiting_for_round" });
      output("autoplay_authorization", { friendlyRoom, autoplay });
      return;
    }
    if (action.name === "NotifyMatchGameStart") {
      friendlyRoom = false;
      pauseAutomation("Ranked play is not supported by the friendly vision.");
      return;
    }
    if (action.name === "NotifyEndGameVote") {
      controlVersion += 1;
      positionVersion += 1;
      if (friendly.cancelVote?.start_time !== action.data.start_time) {
        friendly.voteClickedAt = -Infinity;
      }
      friendly.cancelVote = action.data;
      publishLiveState({});
      output("end_game_vote", { action });
      return;
    }
    if (
      ["NotifyGameEndResult", "NotifyGameTerminate"].includes(action.name)
    ) {
      friendly.cancelVote = null;
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

async function click(x, y, manual = false, vote = false) {
  if (!manual && (vote ? !friendly.shouldVoteYes() : friendly.voteActive())) return false;
  if (automationBlock && !manual) throw new Error("Automation is paused; manual help required.");
  if (friendly.account) friendly.requireIdentity();
  await send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  await wait(8 + Math.floor(Math.random() * 5));
  if (!manual && (vote ? !friendly.shouldVoteYes() : friendly.voteActive())) return false;
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
  return true;
}

async function openBrowser() {
  friendly.cancelVote = null;
  friendly.roomPlayers = null;
  friendly.account = null;
  friendly.ready = false;
  publishLiveState({ status: "starting", stopReason: null });
  state = new MahjongSoulLiveState();
  gameRequests.clear();
  await browser.open({
    chromePath, profilePath, output, ranked: false,
    url: process.env.MAHJONG_SOUL_URL || "https://mahjongsoul.game.yo-star.com",
  });
  output("ready", { screenshotPath, autoplay });

  lastActivityAt = Date.now();
  publishLiveState({ status: "connected" });
}

async function hibernate(reason = null) {
  friendly.cancelVote = null;
  friendly.roomPlayers = null;
  friendly.account = null;
  friendly.ready = false;
  positionVersion += 1;
  gameplay.expectedAction = null;
  friendlyRoom = false;
  publishLiveState({
    status: "hibernating", position: null, prediction: null, roomId: null,
    stopReason: reason,
  });
  await io.captureChain;
  await browser.stop();
}

const controlServer = startLiveControl(controlCommand, () => liveState);
publishLiveState({
  status: "hibernating", controls: { joinRoom: true, screenshotIntervalMs: 250 },
});

const screenRefresh = setInterval(() => {
  captureLobbyScreen().catch((error) =>
    output("screen_capture_error", { message: error.message }),
  );
}, 250);
const heartbeat = setInterval(() => {
  publishLiveState({});
  if (automationBlock) return;
  if (gameplay.expectedAction && gameplay.expectedActionAt !== null && Date.now() - gameplay.expectedActionAt >= 15000) {
    pauseAutomation("The attempted move was not acknowledged by the server.");
    return;
  }
  if (!controlServer.busy && friendly.roomAction(liveState) === "reset") {
    controlServer.busy = true;
    output("empty_friendly_room_reset", {
      roomId: liveState.roomId, status: liveState.status,
      accountIds: friendly.roomPlayers,
    });
    hibernate("All humans left the friendly room").catch((error) =>
      output("hibernate_error", { message: error.message }),
    ).finally(() => { controlServer.busy = false; });
    return;
  }
  if (!controlServer.busy &&
      ["connected", "in_room", "finished", "logged_off"].includes(liveState.status) &&
      Date.now() - lastActivityAt >= idleTimeoutMs) {
    controlServer.busy = true;
    hibernate().catch((error) =>
      output("hibernate_error", { message: error.message }),
    ).finally(() => { controlServer.busy = false; });
    return;
  }
  watchRoom().catch((error) =>
    pauseAutomation(`Room controls failed: ${error.message}`),
  );
  watchFriendlyOverlays().catch((error) =>
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
  controlVersion += 1;
  lastActivityAt = Date.now();
  if (["autoqueue", "room_monitor", "ranked_scroll"].includes(command.action)) {
    throw new Error("Ranked and hosting commands are unavailable on the friendly vision.");
  }
  if (command.action === "resume") {
    if (!automationBlock) throw new Error("Automation is not paused.");
    if (liveState.status === "hibernating") throw new Error("Log in before resuming automation.");
    const enabled = automationBlock.settings.autoplay;
    automationBlock = null;
    afkClicked = false;
    readyClickedAt = null;
    friendly.voteClickedAt = -Infinity;
    postgame.reset();
    command = {action: "autoplay", enabled};
  } else if (automationBlock && (["summon", "join"].includes(command.action) ||
      (command.action === "autoplay" && command.enabled))) {
    throw new Error("Fix the issue, then explicitly resume automation.");
  }
  if (command.action === "summon") return friendly.summon(command.roomId, {
    status: () => liveState, open: openBrowser, wait,
    home: () => requireLobbyScreen("home", 30000), join: joinRoom,
    room: () => io.capture("room"),
    controls: () => io.capture("ready"), click, stop: hibernate,
  });
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
  if (command.action === "join") return joinRoom(command.roomId);
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
  if (command.action === "key") {
    if (command.value !== "Enter") {
      throw new TypeError("Unsupported manual key.");
    }
    await send("Input.dispatchKeyEvent", {
      type: "keyDown", key: "Enter", code: "Enter",
      windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13,
    });
    await send("Input.dispatchKeyEvent", {
      type: "keyUp", key: "Enter", code: "Enter",
      windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13,
    });
    await io.capture("preview");
    return {};
  }
  if (command.action === "screenshot") {
    return { control: await io.capture("lobby"), screenshotPath };
  }
  throw new TypeError("Unknown command");
}
