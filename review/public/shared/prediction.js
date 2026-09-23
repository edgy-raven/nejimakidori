import { renderCenterEV } from "./center-scores.js";
import { tileBase } from "./tiles.js";
import { baseTiles } from "./tile_values.js";
import { actionText, seatName } from "./labels.js";
import { probabilityBar, shantenDistribution, placementBar, renderDiscardChoices } from "./probabilities.js";
import { actionDistribution } from "./policy.js";
import { paymentEstimates } from "./outcomes.js";

function renderGameOutcome(body, position, estimates) {
  document.querySelectorAll("[data-outcome-rank]").forEach((cell) => {
    cell.replaceChildren();
    if (!estimates) {
      cell.textContent = "—";
      return;
    }
    const outcome = estimates.outcomes[Number(cell.dataset.outcomeRank)];
    const recipient = seatName((body.actor + outcome.recipient) % 4,
      position.selfSeat, position.round.kyoku).split(" · ")[0];
    const label = outcome.type === "Tsumo" ? `All → ${recipient}`
      : `${seatName((body.actor + outcome.payer) % 4,
        position.selfSeat, position.round.kyoku).split(" · ")[0]} → ${recipient}`;
    cell.append(probabilityBar(label, outcome.probability));

  });
  renderCenterEV(body?.model?.structured_outcome?.settlement_mean, body?.actor, estimates?.std);
  document.querySelector('.seat-0 .placement-probabilities').replaceWith(
    placementBar(body?.actor === position.selfSeat ? body?.model?.final_placement_probabilities : undefined));
}

// Each page owns its retained decision state; loading a module does not create it.
export class PredictionView {
  lastOpponentSpeed = null;
  lastDecision = null;

  clear() {
    this.lastOpponentSpeed = null;
    this.lastDecision = null;
  }

  render(snapshot) {
    if (this.lastDecision && (
      this.lastDecision.position.round.prevailingWind !== snapshot.position.round.prevailingWind ||
      this.lastDecision.position.round.kyoku !== snapshot.position.round.kyoku ||
      this.lastDecision.position.round.honba !== snapshot.position.round.honba ||
      this.lastDecision.position.step > snapshot.position.step
    )) {
      this.lastDecision = null;
      this.lastOpponentSpeed = null;
    }
    const forcedRiichi = snapshot.prediction?.status === 200 &&
      snapshot.prediction.body.action_kind === "forced" &&
      (snapshot.position.hand.riichi || snapshot.prediction.body.phase === "RIICHI");
    const prediction = forcedRiichi || snapshot.handOver ? null : snapshot.prediction;
    if (prediction?.status === 200) {
      const body = prediction.body;
      const estimates = body.model?.terminal_payment_probabilities && body.terminal_payment_values
        ? paymentEstimates(body.model, body.terminal_payment_values) : null;
      this.lastDecision = {position: snapshot.position, prediction, estimates};
    }
    const body = this.lastDecision?.prediction.body;
    const hand = document.querySelector(".seat-0 .hand");
    const current = prediction?.body;
    const probabilities = current?.phase === "RIICHI"
      ? current.model?.riichi_policy_probabilities?.slice(0, 38).map(
        (value, index) => value + current.model.riichi_policy_probabilities[index + 38])
      : current?.phase === "DISCARD" ? current.model?.discard_policy_probabilities : undefined;
    renderDiscardChoices(hand, current?.discard_options,
      current?.actor === snapshot.position.selfSeat &&
      Number(hand.dataset.step) === snapshot.position.step ? probabilities : undefined);
    const estimates = this.lastDecision?.estimates;
    renderGameOutcome(snapshot.handOver ? null : body,
      this.lastDecision?.position || snapshot.position,
      snapshot.handOver ? null : estimates);
    document.querySelector("#live-decision-label").textContent =
      snapshot.handOver ? "Last model decision" : "Model decision";
    const recommendation = document.querySelector("#live-recommendation");
    if (body) {
      const discardTile = body.action.type === "dahai"
        ? tileBase(body.action.pai) : null;
      const chosenProbability = body.model && (discardTile
        ? body.model.discard_policy_probabilities[
          body.action.tsumogiri ? 37 :
          body.action.pai.startsWith("0") || body.action.pai.endsWith("r")
            ? 34 + "mps".indexOf(discardTile[1])
            : "mpsz".indexOf(discardTile[1]) * 9 + Number(discardTile[0]) - 1]
        : body.phase === "KAN"
          ? body.model.kan_action_probabilities[body.action.type === "none" ? 0
            : 1 + baseTiles.indexOf(tileBase(body.action.pai || body.action.consumed[0]))]
        : ({
          RESPONSE: body.model.response_action_probabilities,
          RIICHI: body.model.riichi_action_probabilities,
        })[body.phase]?.[({
          RESPONSE: ["none", "chi", "pon", "daiminkan"],
          RIICHI: ["none", "reach"],
        })[body.phase]?.indexOf(body.action.type)]);
      recommendation.innerHTML =
        `<strong>${actionText(body.action)}` +
        (Number.isFinite(chosenProbability)
          ? ` (${(chosenProbability * 100).toFixed(1)}%)` : "") +
        (body.riichi_discard ? ` → ${actionText(body.riichi_discard)}` : "") +
        (body.call_discard ? ` → ${actionText(body.call_discard)}` : "") +
        `</strong>`;
    } else if (prediction) {
      recommendation.textContent = prediction.body?.detail ||
        prediction.body?.error || "Prediction failed.";
    } else {
      recommendation.textContent = "Waiting for the model.";
    }
    const actions = document.querySelector("#live-actions");
    const decisionKey = this.lastDecision && JSON.stringify([
      this.lastDecision.position.round.prevailingWind, this.lastDecision.position.round.kyoku,
      this.lastDecision.position.round.honba, this.lastDecision.position.step]);
    const expanded = [...actions.querySelectorAll('details')].map(node => node.open);
    const sameDecision = actions.dataset.decision === decisionKey;
    actions.dataset.decision = decisionKey;
    actions.replaceChildren();
    if (prediction?.status === 200 && body?.model) {
      this.lastOpponentSpeed = {
        actor: body.actor,
        probabilities: body.model.opponent_shanten_probabilities,
      };
    }
    document.querySelectorAll(".live-opponent-speed").forEach((element) => {
      element.replaceChildren(shantenDistribution());
    });
    if (this.lastOpponentSpeed && !snapshot.handOver) {
      this.lastOpponentSpeed.probabilities.forEach((probabilities, index) => {
        const seat = (this.lastOpponentSpeed.actor + index + 1) % 4;
        const display = document.querySelector(`[data-speed-seat="${seat}"]`);
        display.replaceChildren(shantenDistribution(probabilities));
      });

    }
    if (body?.model) {
      actions.append(actionDistribution(
        body.phase, body.predictions?.[0] || body.model, body.discard_options,
        {minimumProbability: 0.05},
      ));
    } else {
      actions.textContent = body?.action_kind === "forced"
        ? `${actionText(body.action)} · forced action`
        : "No policy prediction at this position.";
    }
    if (sameDecision) actions.querySelectorAll('details').forEach((node, index) => {
      node.open = expanded[index];
    });
    document.querySelectorAll("#live-latency").forEach(element => {
      element.textContent = Number.isFinite(body?.model_latency_ms)
        ? `${body.model_latency_ms.toFixed(1)} ms` : "—";
    });
    document.querySelectorAll("#live-difficulty").forEach(element => {
      element.textContent = Number.isFinite(body?.difficulty)
        ? `${(body.difficulty * 100).toFixed(1)}%` : "—";
    });
  }
}
