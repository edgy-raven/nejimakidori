import { renderCenterScores } from "../shared/center-scores.js";
import { loadReview } from "./load.js";
import { parseReviewLink, seatFromLink } from "./link.js";
import { penaltyCategory } from "./penalty.js";
import { markMistake } from "./marks.js";
import { observeBoard } from "../shared/board-layout.js";
import { kyokuNotice } from "../shared/kyoku-notice.js";
import { seatName, winText, englishLabel, actionName, roundText } from "../shared/labels.js";
import { topModelAction } from "../shared/policy.js";
import { renderSeat as drawSeat, renderRiver as drawRiver, renderDora } from "../shared/board.js?v=tile-atlas-20260923";
import { PredictionView } from "../shared/prediction.js";
import { tileText } from "../shared/tiles.js";
import { playbackState, displayHand } from "../shared/replay.js";

observeBoard();
const showKyoku = kyokuNotice(document.querySelector(".board-viewport"));
const predictionView = new PredictionView();

const form = document.querySelector("#load-form");
const reviewError = document.querySelector("#review-error");
const replay = document.querySelector("#replay");
const roundSelect = document.querySelector("#round-select");
const povSelect = document.querySelector("#pov-select");
const cursor = document.querySelector("#cursor");
const progress = document.querySelector("#progress");
const playButton = document.querySelector("#play");
const predictionStatus = document.querySelector("#prediction-status");
const loading = document.querySelector("#review-loading");
const loadingProgress = document.querySelector("#review-progress");
const loadingLabel = document.querySelector("#review-loading-label");
let game = null;
let loadedLog = null;
let timer = null;
let loadRequest = null;
let loadTimer = null;
let pendingLog = null;
let pendingSeat = null;
let currentGrade = null;
let visibleMistakes = [];
let povSeat = 0;
let suppressKyokuNotice = false;
const hiddenHandSeats = new Set([0, 1, 2, 3]);

function updateReviewUrl(logUrl) {
  const url = new URL(window.location.href);
  if (povSelect.value === "auto") url.searchParams.delete("seat");
  else url.searchParams.set("seat", povSelect.value);
  if (logUrl) url.searchParams.set("url", logUrl);
  window.history.replaceState(null, "", url);
}

function selectPov() {
  const linkedSeat = seatFromLink(loadedLog, game.players);
  povSeat = povSelect.value === "auto" ? linkedSeat ?? 0 : Number(povSelect.value);
  hiddenHandSeats.clear();
  [0, 1, 2, 3].filter(seat => seat !== povSeat)
    .forEach(seat => hiddenHandSeats.add(seat));

}

function reviewPosition(eventCount = Number(cursor.value)) {
  const round = game.rounds[Number(roundSelect.value)];
  const events = round.events.slice(0, eventCount);
  const state = playbackState(round, events.length);
  const {concealed, drawn} = displayHand(state, povSeat);
  return {selfSeat: povSeat, players: game.players, scores: state.scores, step: events.length,
    action: events.at(-1)?.type || "Round start",
    round: {...round, events}, hand: {concealed, drawn}};
}

function renderPrediction(payload, eventCount) {
  const focus = [...payload.review.checks].sort((a, b) =>
    (a.played + 0.05) / a.best - (b.played + 0.05) / b.best)[0];
  const phase = payload.review.checks.some(check => check.head === "riichi_action")
    ? "RIICHI" : {discard_policy: "DISCARD", riichi_action: "RIICHI",
    response_action: "RESPONSE", kan_action: "KAN", chi_pattern: "RESPONSE",
    }[focus.head];
  const model = {...payload.predictions[0]};
  if (phase === "DISCARD" || phase === "RIICHI") {
    const probabilities = model.discard_policy_probabilities;
    const groups = new Map();
    for (const option of payload.discard_options) {
      if (!groups.has(option.tile)) groups.set(option.tile, []);
      groups.get(option.tile).push(option.action);
    }
    model.discard_policy_probabilities = probabilities.map(() => 0);
    for (const actions of groups.values()) {
      actions.sort((a, b) => probabilities[b] - probabilities[a]);
      model.discard_policy_probabilities[actions[0]] =
        actions.reduce((sum, action) => sum + probabilities[action], 0);
    }
  }
  predictionView.render({position: reviewPosition(eventCount), prediction: {status: 200, body: {
    ...payload, model, phase,
    action: topModelAction(phase, model, payload.discard_options),
    action_kind: "recommended",
  }}});
  if (phase === "RIICHI") {
    const reach = model.riichi_action_probabilities[1] > model.riichi_action_probabilities[0];
    const policy = payload.predictions[0].riichi_policy_probabilities.slice(reach ? 38 : 0, reach ? 76 : 38);
    const tiles = new Map();
    for (const option of payload.discard_options) {
      tiles.set(option.tile, (tiles.get(option.tile) || 0) + policy[option.action]);
    }
    const [tile, probability] = [...tiles].sort((a, b) => b[1] - a[1])[0];
    const recommendation = document.createElement("strong");
    recommendation.textContent =
      `${reach ? "Reach → " : ""}Discard ${tileText(tile)} (${(100 * probability).toFixed(1)}%)`;
    document.querySelector("#live-recommendation").replaceChildren(recommendation);
  }
}

function isPovDecision(event) {
  return event?.seat === povSeat &&
    ["discard", "call", "kan"].includes(event.type);
}

function renderGrade() {
  currentGrade = game.grades[povSeat];
  document.querySelector("#grade-letter").textContent = currentGrade.score === null
    ? "—" : [[90, "S"], [89.25, "S−"], [88.5, "A+"], [87.5, "A"],
      [86.5, "A−"], [85, "B+"], [83.5, "B"], [81.75, "B−"],
      [80, "C+"], [78, "C"], [76, "C−"], [0, "D"]].find(
      ([threshold]) => currentGrade.score >= threshold)[1];
  document.querySelector("#grade-letter").parentElement.dataset.tone = currentGrade.score === null
    ? "" : currentGrade.score >= 86.5 ? "green"
      : currentGrade.score >= 80 ? "yellow" : "red";
  document.querySelector("#grade-score").textContent = currentGrade.score === null
    ? "—" : `${currentGrade.score.toFixed(1)}%`;
  document.querySelector("#grade-match").textContent = currentGrade.graded
    ? `${(100 * currentGrade.matched / currentGrade.graded).toFixed(1)}%` : "—";
  document.querySelector("#grade-count").textContent = currentGrade.graded
    ? `${currentGrade.matched}/${currentGrade.graded}` : "—";
  const issues = currentGrade.mistakes.map(mistake => ({...mistake,
    category: penaltyCategory(100 * game.annotations[mistake.roundIndex][mistake.eventCount].review.penalty / currentGrade.weight),
  }));
  const selected = new Set();
  document.querySelectorAll("[data-penalty-filter]").forEach(input => {
    const category = input.dataset.penaltyFilter;
    document.querySelector(`[data-penalty-count="${category}"]`).textContent =
      issues.filter(issue => issue.category === category).length;
    if (input.checked) selected.add(category);
  });
  visibleMistakes = issues.filter(issue => selected.has(issue.category));
  document.querySelectorAll("#previous-mistake, #next-mistake").forEach(button => {
    button.disabled = !visibleMistakes.length;
  });
}

function updateHandVisibility() {
  document.querySelectorAll(".hand[data-hand-seat]").forEach((hand) => {
    const seat = Number(hand.dataset.handSeat);
    const visiblePov = seat === povSeat;
    hand.classList.toggle(
      "hand-hidden",
      !visiblePov && hand.dataset.revealed !== "true" && hiddenHandSeats.has(seat),
    );
    hand.classList.toggle("hand-active", visiblePov);
    hand.setAttribute("aria-label", visiblePov
      ? "Selected point-of-view hand"
      : hiddenHandSeats.has(seat)
        ? "Click to reveal this hand"
        : "Click to hide this hand");
  });
}

function renderSeat(seat, position, state) {
  drawSeat(seat, position, state);
  const relative = (seat - povSeat + 4) % 4;
  const hand = document.querySelector(`[data-live-seat="${relative}"] .hand`);
  hand.dataset.handSeat = seat;
  hand.addEventListener("click", () => {
    if (seat === povSeat) return;
    if (hiddenHandSeats.has(seat)) hiddenHandSeats.delete(seat);
    else hiddenHandSeats.add(seat);
    updateHandVisibility();
  });
}

function eventLabel(event, state) {
  if (!event) {
    return "Round start";
  }
  if (event.type === "draw") {
    return `${seatName(event.seat, povSeat, game.rounds[Number(roundSelect.value)].kyoku).split(" · ")[0]} draws ${tileText(event.tile)}`;
  }
  if (event.type === "discard") {
    return `${seatName(event.seat, povSeat, game.rounds[Number(roundSelect.value)].kyoku).split(" · ")[0]} ${event.riichi ? "Reach → Discard" : "Discard"} ${tileText(event.tile)}`;
  }
  if (event.type === "call" || event.type === "kan") {
    return `${seatName(event.seat, povSeat, game.rounds[Number(roundSelect.value)].kyoku).split(" · ")[0]} ${actionName(event.callType)}: ${event.tiles.map(tileText).join(", ")}`;
  }
  if (event.type === "agari") {
    return state.winners.map((winner) =>
      winText(winner, povSeat, game.rounds[Number(roundSelect.value)].kyoku)
    ).join(" · ");
  }
  return englishLabel(state.result || event.type);
}

function render() {
  if (!game) {
    return;
  }
  const round = game.rounds[Number(roundSelect.value)];
  const eventCount = Number(cursor.value);
  showKyoku(`${game.gameId}:${roundSelect.value}`, roundText(round), round, povSeat,
    {show: !suppressKyokuNotice});
  const state = playbackState(round, eventCount);
  const position = reviewPosition();
  predictionView.clear();
  [0, 1, 2, 3].forEach((seat) => {
    renderSeat(seat, position, state);
    drawRiver(seat, povSeat, state);
  });
  updateHandVisibility();
  document.querySelector("#live-round").textContent = roundText(round);
  renderDora(state.doras);
  renderCenterScores(document.querySelector(".center"), state, position.round, position.selfSeat);
  progress.value = `${eventCount} / ${round.events.length}`;
  const annotations = game.annotations[Number(roundSelect.value)];
  const annotatedEvent = annotations.findLastIndex((annotation, index) =>
    index <= eventCount && annotation?.actor === povSeat);
  const review = annotatedEvent === -1 ? null : annotations[annotatedEvent].review;
  if (review !== null) {
    renderPrediction(annotations[annotatedEvent], annotatedEvent);
    predictionStatus.replaceChildren();
    const check = [...review.checks].sort((a, b) =>
      (a.played + 0.05) / a.best - (b.played + 0.05) / b.best)[0];
    const played = document.createElement("strong");
    played.textContent = eventLabel(round.events[annotatedEvent], state)
      .replace(/^Self /, "") +
      ` (${(100 * check.played).toFixed(1)}%)`;
    document.querySelector("#player-choice").replaceChildren(played);
    const calculation = document.createElement("table");
    calculation.className = "review-calculation";
    calculation.setAttribute("aria-label", "Decision penalty calculation");
    const rows = document.createElement("tbody");
    for (const [label, value] of [
      ["Loss", `max(0, (${(100 * check.best).toFixed(1)}% − ${(100 * check.played).toFixed(1)}% − 5%) ÷ ${(100 * check.best).toFixed(1)}%) = ${(100 * (1 - review.credit)).toFixed(1)}%`],
      ["Weight", review.weight.toFixed(2)],
      ["Weighted penalty", `= ${review.penalty.toFixed(3)}`],
      ["Total game weight", `÷ ${game.grades[povSeat].weight.toFixed(2)}`],
      ["Grade penalty", `${review.mistake ? "−" : ""}${(100 * review.penalty / game.grades[povSeat].weight).toFixed(3)}%`],
    ]) {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = label;
      const result = document.createElement("td");
      result.textContent = value;
      row.append(name, result);
      rows.append(row);
    }
    calculation.append(rows);
    predictionStatus.append(calculation);
    predictionStatus.classList.toggle("review-difference", review.mistake);
    if (state.handOver) {
      predictionView.render({position, handOver: true, prediction: null});
    }
  } else {
    predictionView.render({position, handOver: state.handOver, prediction: null});
    predictionStatus.classList.remove("review-difference");
    predictionStatus.textContent = annotatedEvent === -1 ? "—"
      : round.events[annotatedEvent].type === "discard"
        ? "Forced discard · not graded" : "Forced action · not graded";
    document.querySelector("#player-choice").textContent = annotatedEvent === -1
      ? "—" : eventLabel(round.events[annotatedEvent], state).replace(/^Self /, "");
    document.querySelector("#live-recommendation").textContent =
      "—";
  }
  document.querySelector("#live-decision-label").textContent = "Grade breakdown";
  const category = review === null ? null : penaltyCategory(
    100 * review.penalty / game.grades[povSeat].weight);
  const badge = document.querySelector("#decision-category");
  badge.textContent = category || "";
  badge.dataset.penaltyCategory = category || "";
  predictionStatus.dataset.penaltyCategory = category || "";
  for (const mistake of visibleMistakes) {
    if (mistake.roundIndex === Number(roundSelect.value) && mistake.eventCount <= eventCount) {
      markMistake(round, eventCount, mistake.eventCount,
        mistake.category, povSeat);
    }
  }
}

function selectRound(index) {
  stopPlaying();
  roundSelect.value = String(index);
  cursor.max = game.rounds[index].events.length;
  cursor.value = "1";
  render();
}

function stopPlaying() {
  clearInterval(timer);
  timer = null;
  playButton.textContent = "Play";
}

function step(amount) {
  cursor.value = String(
    Math.max(0, Math.min(Number(cursor.max), Number(cursor.value) + amount)),
  );
  render();
  if (Number(cursor.value) === Number(cursor.max)) {
    stopPlaying();
  }
}

function stepRound(amount) {
  const index = Math.max(
    0,
    Math.min(game.rounds.length - 1, Number(roundSelect.value) + amount),
  );
  selectRound(index);
}

function stepChoice(amount) {
  const round = game.rounds[Number(roundSelect.value)];
  const current = Number(cursor.value);
  const choices = round.events
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => isPovDecision(event))
    .map(({ index }) => index);
  const target = amount > 0
    ? choices.find((index) => index > current)
    : choices.findLast((index) => index < current);
  if (target !== undefined) {
    cursor.value = String(target);
    render();
  }
}

function jumpToMistake(event) {
  const tile = event.target.closest("[data-mistake-decision]");
  if (!tile || (event.type === "keydown" && !["Enter", " "].includes(event.key))) return;
  event.preventDefault();
  event.stopPropagation();
  stopPlaying();
  cursor.value = tile.dataset.mistakeDecision;
  render();
}

function stepMistake(amount) {
  const offset = mistake => mistake.roundIndex - Number(roundSelect.value) ||
    mistake.eventCount - Number(cursor.value);
  const target = amount > 0
    ? visibleMistakes.find(mistake => offset(mistake) > 0) || visibleMistakes[0]
    : visibleMistakes.findLast(mistake => offset(mistake) < 0) || visibleMistakes.at(-1);
  suppressKyokuNotice = true;
  selectRound(target.roundIndex);
  cursor.value = String(target.eventCount);
  render();
  suppressKyokuNotice = false;
}

async function openReview(request) {
  reviewError.textContent = "";
  form.setAttribute("aria-busy", "true");
  try {
    const result = await loadReview(pendingLog.url, {
      seat: pendingSeat,
      signal: request.signal,
      onProgress: update => {
        if (request !== loadRequest) return;
        loadingLabel.textContent = update.total === undefined ? update.message
          : `${update.completed} / ${update.total} decisions`;
        if (update.total === undefined) loadingProgress.removeAttribute("value");
        else {
          loadingProgress.max = Math.max(1, update.total);
          loadingProgress.value = update.completed;
        }
      },
    });
    if (request !== loadRequest) return;
    game = result;
    loadedLog = pendingLog;
    selectPov();
    roundSelect.replaceChildren(
      ...game.rounds.map((round, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = roundText(round);
        return option;
      }),
    );
    loading.hidden = true;
    replay.hidden = false;
    updateReviewUrl(loadedLog.url);
    renderGrade();
    selectRound(0);
  } catch (error) {
    if (request === loadRequest && error.name !== "AbortError") {
      reviewError.textContent = error.message;
      loading.hidden = true;
    }
  } finally {
    if (request === loadRequest) {
      pendingLog = null;
      loadRequest = null;
      form.removeAttribute("aria-busy");
    }
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  let link;
  try {
    link = parseReviewLink(new FormData(form).get("url"));
  } catch (error) {
    reviewError.textContent = error.message;
    return;
  }
  document.querySelector("#log-url").value = link.url;
  const seat = povSelect.value === "auto" ? "auto" : Number(povSelect.value);
  if (pendingLog?.recordId === link.recordId && pendingSeat === seat && pendingLog.linkId === link.linkId) {
    pendingLog = link;
    return;
  }
  clearTimeout(loadTimer);
  loadRequest?.abort();
  loadRequest = null;
  pendingLog = null;
  form.removeAttribute("aria-busy");
  stopPlaying();
  if (game?.gameId === link.recordId && game.seat ===
      (seat === "auto" ? seatFromLink(link, game.players) ?? 0 : seat)) {
    loadedLog = link;
    selectPov();
    updateReviewUrl(link.url);
    renderGrade();
    render();
    return;
  }
  pendingLog = link;
  pendingSeat = seat;
  game = null;
  replay.hidden = true;
  loading.hidden = false;
  loadingLabel.textContent = "Loading…";
  loadingProgress.removeAttribute("value");
  roundSelect.replaceChildren();
  document.querySelectorAll("#previous-mistake, #next-mistake").forEach(button => { button.disabled = true; });
  loadRequest = new AbortController();
  reviewError.textContent = "";
  loadTimer = setTimeout(() => openReview(loadRequest), 350);
});

roundSelect.addEventListener("change", () => selectRound(Number(roundSelect.value)));
povSelect.addEventListener("change", () => {
  updateReviewUrl();
  if (document.querySelector("#log-url").value.trim()) form.requestSubmit();
});
cursor.addEventListener("input", render);
document.querySelector("#previous").addEventListener("click", () => step(-1));
document.querySelector("#next").addEventListener("click", () => step(1));
document.querySelector("#previous-round").addEventListener(
  "click", () => stepRound(-1),
);
document.querySelector("#next-round").addEventListener(
  "click", () => stepRound(1),
);
document.querySelector("#previous-choice").addEventListener(
  "click", () => stepChoice(-1),
);
document.querySelector("#next-choice").addEventListener(
  "click", () => stepChoice(1),
);
playButton.addEventListener("click", () => {
  if (timer) {
    stopPlaying();
    return;
  }
  if (Number(cursor.value) === Number(cursor.max)) {
    cursor.value = "0";
  }
  playButton.textContent = "Pause";
  timer = setInterval(() => step(1), 250);
});
window.addEventListener("keydown", (event) => {
  if (!game || event.target.matches("input, select, button")) {
    return;
  }
  if (event.key === "ArrowLeft") {
    step(-1);
  } else if (event.key === "ArrowRight") {
    step(1);
  } else if (event.key === " ") {
    event.preventDefault();
    playButton.click();
  }
});

replay.addEventListener("click", jumpToMistake, true);
replay.addEventListener("keydown", jumpToMistake, true);

document.querySelectorAll("[data-penalty-filter]").forEach(input => {
  input.addEventListener("change", () => {
    renderGrade();
    render();
  });
});
document.querySelector("#previous-mistake").addEventListener("click", () => stepMistake(-1));
document.querySelector("#next-mistake").addEventListener("click", () => stepMistake(1));

const sharedParams = new URLSearchParams(window.location.search);
const sharedSeat = sharedParams.get("seat") ?? "auto";
const sharedLogUrl = sharedParams.get("url");
if (sharedLogUrl) document.querySelector("#log-url").value = sharedLogUrl;
if (!/^(auto|[0-3])$/.test(sharedSeat)) {
  reviewError.textContent = "Seat must be auto, 0 (East), 1 (South), 2 (West), or 3 (North).";
} else {
  povSelect.value = sharedSeat;
  if (sharedLogUrl) form.requestSubmit();
}
