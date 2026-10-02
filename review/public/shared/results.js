import {englishLabel, winText} from "./labels.js";

export function renderHandResult(result) {
  const panel = document.querySelector("#live-hand-result");
  panel.replaceChildren();
  if (result?.winners.length) {
    result.winners.forEach(winner => {
      const line = document.createElement("div");
      line.className = "hand-winner-result";
      const name = document.createElement("strong");
      name.textContent = winText(winner, result.selfSeat, result.kyoku);
      const yaku = document.createElement("span");
      yaku.textContent = winner.yaku
        .filter(item => item.name !== "Ura Dora" || item.value > 0)
        .map(item => `${englishLabel(item.name)}${["Dora", "Red Five", "Ura Dora"].includes(item.name) ? ` ${item.value}` : ""}`).join(" · ");
      line.append(name, yaku);
      panel.append(line);
    });
  } else panel.textContent = englishLabel(result?.outcome || "");
  document.querySelectorAll("[data-last-seat]").forEach(cell => {
    const seat = result ? (result.selfSeat + Number(cell.dataset.lastSeat)) % 4 : null;
    cell.textContent = result ? result.deltas[seat].toLocaleString(undefined, {signDisplay: "exceptZero"}) : "—";
    cell.title = result ? `Score ${result.scores[seat].toLocaleString()}` : "";
  });
}

export function renderMatchResult(snapshot) {
  const finalResult = document.querySelector("#final-match-result");
  finalResult.hidden = snapshot.status !== "finished" || !snapshot.finalMatchResult;
  finalResult.closest(".center").classList.toggle("show-final-rankings", !finalResult.hidden);
  const rows = finalResult.querySelector("tbody");
  rows.replaceChildren();
  if (!finalResult.hidden) {
    snapshot.finalMatchResult.players.forEach((player) => {
      const row = document.createElement("tr");
      row.className = `final-place-${player.rank}`;
      for (const value of [player.rank,
        ["Self", "Right", "Across", "Left"][(player.seat - snapshot.finalMatchResult.selfSeat + 4) % 4],
        player.score.toLocaleString()]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      rows.append(row);
    });
  }
}
