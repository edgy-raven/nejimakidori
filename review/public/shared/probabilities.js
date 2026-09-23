export function probabilityBar(label, value) {
  const row = document.createElement("div");
  row.className = "probability";
  row.innerHTML =
    `<span>${label}</span><i><b style="width:${(value ?? 0) * 100}%"></b></i>` +
    `<strong>${value === undefined ? "—" : (value * 100).toFixed(1) + "%"}</strong>`;
  return row;
}

function distribution(labels, probabilities, className) {
  const section = document.createElement("div");
  section.className = className;
  const bar = document.createElement("div");
  bar.className = "speed-bar";
  const legend = document.createElement("div");
  legend.className = "speed-labels";
  labels.forEach((label, index) => {
    const value = probabilities?.[index];
    const segment = document.createElement("span");
    segment.style.width = `${(value ?? 0) * 100}%`;
    segment.title = `${label}: ${value === undefined ? "—" : (value * 100).toFixed(1) + "%"}`;
    bar.append(segment);
    legend.append(probabilityBar(label, value));
  });
  section.append(legend, bar);
  return section;
}

export function shantenDistribution(probabilities) {
  const section = distribution(["Ready", "1 away", "2 away", "3+ away"], probabilities, "seat-probabilities");
  if (probabilities) {
    const mode = probabilities.indexOf(Math.max(...probabilities));
    section.querySelector(".speed-labels").children[mode].classList.add("speed-mode");
  }
  return section;
}

export function placementBar(probabilities) {
  const bar = distribution(["1st", "2nd", "3rd", "4th"], probabilities, "placement-probabilities");
  bar.setAttribute("aria-label", "Final placement probabilities");
  return bar;
}

// Equivalent held/drawn copies are one discard choice; label the preferred copy.
export function renderDiscardChoices(hand, options, probabilities) {
  hand.querySelectorAll(".tile-choice-label").forEach(label => label.remove());
  if (!probabilities) return;
  const choices = new Map();
  for (const option of options) {
    const probability = probabilities[option.action];
    const choice = choices.get(option.tile);
    if (choice) {
      choice.total += probability;
      if (probability > choice.probability) Object.assign(choice, {option, probability});
    } else choices.set(option.tile, {option, probability, total: probability});
  }
  for (const {option, total} of choices.values()) {
    if (total <= .05) continue;
    const tiles = hand.querySelectorAll(option.tsumogiri ? ".drawn-slot .tile" : ":scope > .tile");
    const tile = [...tiles].find(tile => tile.dataset.tile === option.tile);
    // Retained sidebar predictions can refer to tiles no longer in the hand.
    if (!tile) continue;
    const label = document.createElement("span");
    label.className = "tile-choice-label";
    label.textContent = `${(total * 100).toFixed(1)}%`;
    label.setAttribute("aria-label", `Model discard choice: ${label.textContent}`);
    label.style.left = `${tile.offsetLeft + tile.offsetWidth / 2}px`;
    label.style.top = `${tile.offsetTop - 14}px`;
    hand.append(label);
  }
}
