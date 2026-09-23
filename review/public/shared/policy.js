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
  const groups = new Map();
  for (const option of options) {
    if (!groups.has(option.tile)) groups.set(option.tile, {options: [], probability: 0});
    const group = groups.get(option.tile);
    group.options.push(option);
    group.probability += probabilities[option.action];
  }
  for (const group of [...groups.values()].sort((a, b) => b.probability - a.probability)) {
    if (group.probability <= minimumProbability) continue;
    group.options.sort((a, b) => probabilities[b.action] - probabilities[a.action]);
    const option = group.options[0];
    const ambiguous = group.options.length > 1;
    const row = choiceRow(shantenText(option.all_shanten), group.probability,
      option.tile, ambiguous ? "summary" : "div");
    row.dataset.tile = option.tile;
    const counts = document.createElement("small");
    counts.textContent = `Acceptance: ${option.all_ukeire_count} · Upgrades: ${option.all_upgrade_count}`;
    row.querySelector(".choice-label > span:last-child").append(counts);
    if (ambiguous) {
      const dropdown = document.createElement("details");
      dropdown.className = "discard-variants";
      dropdown.append(row);
      for (const variant of group.options) {
        dropdown.append(choiceRow(variant.tsumogiri ? "Drawn tile" : "Held tile",
          probabilities[variant.action]));
      }
      list.append(dropdown);
    } else list.append(row);
  }
  return list;
}

function reachActionTable(prediction, options, minimumProbability) {
  const section = document.createElement("div");
  section.className = "reach-action-list";
  const branches = [0, 38].map(offset => {
    const policy = prediction.riichi_policy_probabilities.slice(offset, offset + 38);
    return {policy, total: policy.reduce((sum, value) => sum + value, 0)};
  });
  if (branches[1].total > 0) {
    const reach = document.createElement("details");
    reach.className = "reach-choices";
    const summary = choiceRow("Reach", branches[1].total, null, "summary");
    reach.append(summary, discardActionTable(options,
      branches[1].policy, minimumProbability * branches[1].total));
    section.append(reach);
  }
  if (branches[0].total > 0) {
    section.append(discardActionTable(options,
      branches[0].policy, minimumProbability * branches[0].total));
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

function appendActionPolicy(display, phase, prediction, minimumProbability) {
  const policies = {
    RESPONSE: [
      ["Pass", "Chi", "Pon", "Open quad"],
      prediction.response_action_probabilities,
    ],
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
