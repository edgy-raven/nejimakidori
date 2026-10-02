function pointsValue(value) {
  const number = document.createElement("span");
  number.className = "points-value";
  if (value === undefined) { number.textContent = "—"; return number; }
  number.textContent = `${value < 0 ? "−" : ""}${Math.floor(Math.abs(value) / 100)}`;
  const trailing = document.createElement("small");
  trailing.textContent = String(Math.abs(value) % 100).padStart(2, "0");
  number.append(trailing);
  return number;
}

function deltaEstimate(element, mean, deviation) {
  element.replaceChildren("Δ ", pointsValue(mean === undefined ? undefined : Math.round(mean / 100) * 100),
    " ± ", pointsValue(deviation === undefined ? undefined : Math.round(deviation / 100) * 100));
  element.title = "Expected settlement ± standard deviation, in points";
}

export function renderCenterScores(center, state, {events, kyoku = 1}, selfSeat) {
  center.classList.add("score-center");
  if (!center.querySelector(".center-details")) {
    const details = document.createElement("section");
    details.className = "center-details";
    details.append(...[...center.children].filter(node => node.id !== "final-match-result"));
    center.append(details);
  }
  let bank = center.querySelector(".center-bank");
  if (!bank) {
    bank = document.createElement("div");
    bank.className = "center-bank";
    bank.innerHTML = '<span class="bank-sticks"></span><span class="wall-counter"></span>';
    center.querySelector(".center-details").append(bank);
  }
  bank.querySelector(".bank-sticks").textContent = `${state.riichiSticks} ${state.riichiSticks === 1 ? "stick" : "sticks"}`;
  bank.querySelector(".wall-counter").textContent = `x${state.remaining ?? "—"}`;
  center.querySelectorAll(".center-player").forEach(node => node.remove());
  const event = events?.findLast(event => event.seat !== undefined);
  const acting = state.handOver || state.result ? null : event?.seat ?? (events ? kyoku - 1 : selfSeat);
  for (let seat = 0; seat < 4; seat++) {
    const relative = (seat - selfSeat + 4) % 4;
    const strip = document.createElement("section");
    strip.className = `center-player center-player-${relative}`;
    strip.classList.toggle("is-acting", seat === acting);
    strip.dataset.centerSeat = seat;
    const wind = ["East", "South", "West", "North"][(seat - kyoku + 5) % 4];
    strip.setAttribute("aria-label", `${wind}: ${state.scores[seat]} points${seat === acting ? ", acting player" : ""}`);
    const direction = document.createElement("b");
    direction.className = "center-direction";
    direction.textContent = wind;
    direction.title = wind;
    const score = document.createElement("strong");
    score.className = "center-score";
    score.append(pointsValue(state.scores[seat]));
    const ev = document.createElement("span");
    ev.className = "center-ev";
    deltaEstimate(ev);
    strip.append(direction, score, ev);
    center.append(strip);
  }
}

export function renderCenterEV(means, actor, deviations) {
  document.querySelectorAll("[data-center-seat]").forEach(strip => {
    const relative = (Number(strip.dataset.centerSeat) - actor + 4) % 4;
    const mean = means?.[relative];
    const deviation = deviations?.[relative];
    deltaEstimate(strip.querySelector(".center-ev"), mean, deviation);
  });
}
