import { tileElement } from "./tiles.js";
import { baseTiles } from "./tile_values.js";
import { shantenText } from "./labels.js";

function choiceRow(label, probability, tile = null, tag = "div") {
  const row = document.createElement(tag);
  row.className = "choice-probability choice-row";
  row.style.setProperty("--choice-percent", `${probability * 100}%`);
  const choice = document.createElement("span");
  choice.className = "choice-label";
  if (tile) choice.append(tileElement(tile));
  const text = document.createElement("span");
  text.textContent = label;
  choice.append(text);
  const percent = document.createElement("strong");
  percent.textContent = `${(probability * 100).toFixed(1)}%`;
  row.append(choice, percent);
  return row;
}

function discardActionTable(options, probabilities, minimumProbability) {
  const list = document.createElement("div");
  list.className = "discard-action-list";
  for (const option of [...options].sort((a, b) =>
    probabilities[b.action] - probabilities[a.action])) {
    if (probabilities[option.action] <= minimumProbability) continue;
    const row = choiceRow(shantenText(option.all_shanten),
      probabilities[option.action], option.tile);
    row.dataset.tile = option.tile;
    const counts = document.createElement("small");
    counts.textContent = `Acceptance: ${option.all_ukeire_count} · Upgrades: ${option.all_upgrade_count}`;
    row.querySelector(".choice-label > span:last-child").append(counts);
    list.append(row);
  }
  return list;
}

function reachActionTable(prediction, options, minimumProbability) {
  const section = document.createElement("div");
  section.className = "reach-action-list";
  const branches = [0, 37].map(offset => {
    const policy = prediction.riichi_policy_probabilities.slice(offset, offset + 37);
    return {policy, total: policy.reduce((sum, value) => sum + value, 0)};
  });
  const best = branches[1].total > branches[0].total ? 1 : 0;
  for (const [index, branch] of branches.entries()) {
    if (branch.total <= minimumProbability) continue;
    const group = document.createElement("details");
    group.className = "reach-choices";
    group.open = index === best;
    group.append(choiceRow(index ? "Reach → Discard" : "Dama → Discard",
      branch.total, null, "summary"),
      discardActionTable(options, branch.policy, minimumProbability));
    section.append(group);
  }
  return section;
}

export function topModelAction(phase, model, discardOptions) {
  if (phase === "DISCARD") {
    const top = [...discardOptions].sort((a, b) =>
      model.discard_policy_probabilities[b.action] - model.discard_policy_probabilities[a.action]
    )[0];
    return {type: "dahai", pai: top.tile, tsumogiri: top.tsumogiri};
  }
  if (phase === "KAN") {
    const probabilities = model.kan_action_probabilities;
    const choice = probabilities.indexOf(Math.max(...probabilities));
    return choice === 0 ? {type: "none"}
      : {type: "kan", pai: baseTiles[choice - 1]};
  }
  const [probabilities, names] = {
    RESPONSE: [model.response_action_probabilities, ["none", "chi", "pon", "daiminkan"]],
    RIICHI: [model.riichi_action_probabilities, ["none", "reach"]],
  }[phase];
  return {type: names[probabilities.indexOf(Math.max(...probabilities))]};
}

function responseActionTable(prediction, minimumProbability) {
  const section = document.createElement("div");
  const policy = prediction.response_policy_probabilities;
  const branches = [0, 1, 2, 3].map(index => {
    const discards = policy.slice(1 + index * 37, 38 + index * 37);
    return {discards, total: discards.reduce((sum, value) => sum + value, 0)};
  });
  const totals = [policy[0], ...branches.map(branch => branch.total), policy[149]];
  const best = totals.indexOf(Math.max(...totals));
  if (policy[0] > minimumProbability) {
    section.append(choiceRow("Pass", policy[0]));
  }
  for (const [index, label] of [
    "Chi (called tile low)", "Chi (called tile middle)",
    "Chi (called tile high)", "Pon",
  ].entries()) {
    const {discards, total} = branches[index];
    if (total <= minimumProbability) continue;
    const group = document.createElement("details");
    group.className = "reach-choices";
    group.open = index + 1 === best;
    const list = document.createElement("div");
    list.className = "discard-action-list";
    [...baseTiles, "0m", "0p", "0s"].map((tile, action) => ({
      tile, probability: discards[action],
    })).sort((a, b) => b.probability - a.probability).forEach(choice => {
      if (choice.probability > minimumProbability) {
        list.append(choiceRow("Discard", choice.probability, choice.tile));
      }
    });
    group.append(choiceRow(`${label} → Discard`, total, null, "summary"), list);
    section.append(group);
  }
  if (policy[149] > minimumProbability) {
    section.append(choiceRow("Open quad", policy[149]));
  }
  return section;
}

function appendActionPolicy(display, phase, prediction, minimumProbability) {
  const policies = {
    KAN: [
      ["Pass", ...baseTiles.map(() => "Quad")],
      prediction.kan_action_probabilities,
    ],
  };
  if (!policies[phase]) return;
  const [labels, probabilities] = policies[phase];
  labels.map((label, index) => ({label, index}))
    .sort((a, b) => probabilities[b.index] - probabilities[a.index])
    .forEach(({label, index}) => {
    if (probabilities[index] > minimumProbability) {
      display.append(choiceRow(label, probabilities[index],
        phase === "KAN" && index > 0 ? baseTiles[index - 1] : null));
    }
  });
}

export function actionDistribution(phase, prediction, discardOptions,
  {minimumProbability = 0.05} = {}) {
  const section = document.createElement("div");
  section.className = "action-distribution";
  if (phase === "RIICHI") {
    section.append(reachActionTable(prediction, discardOptions, minimumProbability));
    return section;
  }
  if (phase === "RESPONSE") {
    section.append(responseActionTable(prediction, minimumProbability));
    return section;
  }
  appendActionPolicy(section, phase, prediction, minimumProbability);
  if (phase === "DISCARD") {
    section.append(discardActionTable(
      discardOptions,
      prediction.discard_policy_probabilities,
      minimumProbability,
    ));
  }
  return section;
}
