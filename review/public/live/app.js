import { renderCenterScores } from "../shared/center-scores.js";
import { renderHandResult, renderMatchResult } from "../shared/results.js";
import { observeBoard } from "../shared/board-layout.js";
import { kyokuNotice } from "../shared/kyoku-notice.js";
import { seatName, roundText } from "../shared/labels.js";
import { renderSeat, renderRiver, renderDora } from "../shared/board.js";
import { PredictionView } from "../shared/prediction.js";
import { accountRankText } from "../shared/ranks.js";
import { playbackState } from "../shared/replay.js";

observeBoard();
const showKyoku = kyokuNotice(document.querySelector(".board-viewport"), document, {closeOnDiscard: true});
const predictionView = new PredictionView();

const view = document.querySelector("#live-view");
let renderedUpdate;
let clicking = false;
let lobbyAvailable = false;
let screenshotUrl;
let screenshotGeneration = 0;
let screenshotIntervalMs = 1500;
const screenshot = document.querySelector("#live-screenshot");

document.querySelector("#live-login").addEventListener("click", async () => {
  const pin = document.querySelector("#live-control-key").value.trim();
  const status = document.querySelector("#live-login-status");
  if (!/^\d{4}$/.test(pin)) {
    status.textContent = "Enter the 4-digit control PIN.";
    return;
  }
  const button = document.querySelector("#live-login");
  button.disabled = true;
  status.textContent = "Opening browser…";
  try {
    const response = await fetch("../api/live/login", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${pin}` },
      body: "{}",
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error);
    status.textContent = "";
    await refresh();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

const autoqueueButton = document.querySelector("#live-autoqueue");
autoqueueButton.addEventListener("click", async () => {
  const pin = document.querySelector("#live-control-key").value.trim();
  const status = document.querySelector("#live-login-status");
  if (!/^\d{4}$/.test(pin)) {
    status.textContent = "Enter the 4-digit control PIN.";
    return;
  }
  autoqueueButton.disabled = true;
  try {
    const response = await fetch("../api/live/autoqueue", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${pin}` },
      body: JSON.stringify({ enabled: autoqueueButton.getAttribute("aria-pressed") !== "true",
        resume: autoqueueButton.dataset.blocked === "true" }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error);
    status.textContent = "";
    await refresh();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    autoqueueButton.disabled = false;
  }
});

screenshot.addEventListener("click", async (event) => {
  if (!lobbyAvailable || clicking || !screenshot.naturalWidth) return;
  const status = document.querySelector("#live-click-status");
  const pin = document.querySelector("#live-control-key").value.trim();
  if (!/^\d{4}$/.test(pin)) {
    status.textContent = "Enter the 4-digit control PIN above.";
    document.querySelector("#live-control-key").focus();
    return;
  }
  const bounds = screenshot.getBoundingClientRect();
  clicking = true;
  screenshotGeneration += 1;
  status.textContent = "Sending click…";
  try {
    const response = await fetch("../api/live/click", {
      method: "POST",
      headers: {
        "Content-Type": "application/json", Authorization: `Bearer ${pin}`,
      },
      body: JSON.stringify({
        x: Math.min(1279, Math.floor((event.clientX - bounds.left) * 1280 / bounds.width)),
        y: Math.min(719, Math.floor((event.clientY - bounds.top) * 720 / bounds.height)),
      }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error);
    status.textContent = "Click sent.";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    clicking = false;
    screenshotGeneration += 1;
  }
});

function renderStatus(snapshot) {
  const queueEnabled = snapshot.autoqueue?.enabled === true;
  autoqueueButton.setAttribute("aria-pressed", String(queueEnabled));
  autoqueueButton.dataset.blocked = String(Boolean(snapshot.automationBlock));
  autoqueueButton.textContent = snapshot.automationBlock
    ? "Resume automation" : `Autoqueue: ${queueEnabled ? "on" : "off"}`;
  const queue = snapshot.autoqueue;
  document.querySelector("#live-ranked-status").textContent = queue?.rank
    ? `Rank ${accountRankText(queue.rank)} · ${queue.copper.toLocaleString()} copper` +
      (queue.room ? ` · ${queue.room.name} ${queue.room.length}` : "") +
      (queue.blockedReason ? ` · ${queue.blockedReason}` :
        queue.pending ? " · matchmaking" : "") +
      (queue.schedule?.pauseUntil > Date.now()
        ? ` · break until ${new Date(queue.schedule.pauseUntil).toLocaleTimeString("en-GB", {timeZone: "Asia/Manila"})} Manila`
        : queue.schedule && !queue.schedule.open ? " · outside Manila queue hours" : "")
    : "Rank and copper awaiting login";
  const stale = snapshot.updatedAt &&
    Date.now() - new Date(snapshot.updatedAt).getTime() > 5000;
  document.querySelector("#live-login").hidden =
    !["logged_off", "hibernating", "offline", "starting"].includes(snapshot.status);
  screenshotIntervalMs = snapshot.controls?.screenshotIntervalMs || 1500;
  const active = snapshot.status === "playing" && !stale;
  lobbyAvailable = !stale &&
    (["connected", "in_room", "finished"].includes(snapshot.status) ||
      (snapshot.automationBlock && snapshot.status === "playing"));
  document.querySelector("#live-browser").hidden = !lobbyAvailable;
  if (["logged_off", "hibernating", "offline"].includes(snapshot.status)) {
    screenshot.removeAttribute("src");
    if (screenshotUrl) URL.revokeObjectURL(screenshotUrl);
    screenshotUrl = undefined;
  }
  screenshot.setAttribute("aria-disabled", String(!lobbyAvailable || clicking));
  document.querySelector("#live-status").textContent = stale
    ? "controller offline"
    : snapshot.automationBlock
    ? `Paused: ${snapshot.automationBlock.reason} · ` +
      (snapshot.discord?.error || "manual help requested")
    : snapshot.stopReason
    ? `${snapshot.stopReason} Log in when ready.`
    : snapshot.status === "logged_off"
    ? "Logged off"
    : `${snapshot.status.replaceAll("_", " ")} · autoplay ` +
      (snapshot.autoplay.enabled
        ? (snapshot.autoplay.ready ? "active" : "waiting for verified hand")
        : "off");
  document.querySelector("#live-indicator").classList.toggle("active", active);
  document.querySelector("#live-updated").textContent = snapshot.updatedAt
    ? `Updated ${new Date(snapshot.updatedAt).toLocaleTimeString()}`
    : "Controller is not running";
}

function render(snapshot) {
  renderStatus(snapshot);
  renderMatchResult(snapshot);
  renderHandResult(snapshot.lastHandResult);
  // Retain the table through settlement, lobby, and reconnect gaps.
  // Only a new round replaces the completed hand.
  if (!snapshot.position?.round || snapshot.position.selfSeat === null) return;
  const position = snapshot.position;
  showKyoku(JSON.stringify([position.round.prevailingWind, position.round.kyoku,
    position.round.honba]), roundText(position.round), position.round, position.selfSeat);
  const state = playbackState(position.round);
  for (let seat = 0; seat < 4; seat += 1) {
    renderSeat(seat, position, state);
    renderRiver(seat, position.selfSeat, state);
  }
  document.querySelector("#live-round").textContent = roundText(position.round);
  renderDora(state.doras);
  renderCenterScores(document.querySelector(".center"), state, position.round, position.selfSeat);
  predictionView.render({...snapshot, handOver: state.handOver,
    position: {...position, scores: state.scores}});
  view.hidden = false;
}

async function refresh() {
  try {
    const response = await fetch("../api/live", { cache: "no-store" });
    const snapshot = await response.json();
    if (snapshot.updatedAt !== renderedUpdate) {
      renderedUpdate = snapshot.updatedAt;
      render(snapshot);
    } else {
      renderStatus(snapshot);
    }
  } catch (error) {
    lobbyAvailable = false;
    document.querySelector("#live-browser").hidden = true;
    document.querySelector("#live-status").textContent = error.message;
    document.querySelector("#live-indicator").classList.remove("active");
  }
}

async function refreshScreenshot() {
  const started = performance.now();
  const generation = screenshotGeneration;
  try {
    if (lobbyAvailable && !clicking && !document.hidden) {
      const response = await fetch("../api/live/screenshot", { cache: "no-store" });
      if (response.ok) {
        const nextUrl = URL.createObjectURL(await response.blob());
        const frame = new Image();
        frame.src = nextUrl;
        try {
          await frame.decode();
        } catch (error) {
          URL.revokeObjectURL(nextUrl);
          throw error;
        }
        if (lobbyAvailable && !clicking && generation === screenshotGeneration) {
          screenshot.src = nextUrl;
          if (screenshotUrl) URL.revokeObjectURL(screenshotUrl);
          screenshotUrl = nextUrl;
        } else {
          URL.revokeObjectURL(nextUrl);
        }
      }
    }
  } catch (error) {
    document.querySelector("#live-click-status").textContent = error.message;
  } finally {
    setTimeout(refreshScreenshot, Math.max(0, screenshotIntervalMs - (performance.now() - started)));
  }
}

async function pollState() {
  await refresh();
  setTimeout(pollState, 150);
}

await pollState();
refreshScreenshot();
