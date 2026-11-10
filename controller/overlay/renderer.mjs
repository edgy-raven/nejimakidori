const element = id => document.getElementById(id);
const actions = {
  dahai: "Discard", reach: "Riichi", chi: "Chii", pon: "Pon",
  daiminkan: "Open kan", ankan: "Closed kan", kakan: "Added kan",
  hora: "Win", ryukyoku: "Draw", none: "Pass",
};
function actionText(action) {
  if (!action) return "—";
  return [actions[action.type] ?? action.type, action.pai,
    action.consumed?.length ? `(${action.consumed.join(" ")})` : "",
  ].filter(Boolean).join(" ");
}
function render(snapshot) {
  element("status").textContent = snapshot.status.replaceAll("_", " ");
  element("seat").textContent = snapshot.position?.selfSeat == null
    ? "" : `Seat ${snapshot.position.selfSeat + 1}`;
  element("game").textContent = snapshot.gameId ? `Game ${snapshot.gameId}` : "No game";
  element("decision").textContent = actionText(snapshot.prediction?.decision.action);
  const discard = snapshot.prediction?.decision.riichi_discard ??
    snapshot.prediction?.decision.call_discard;
  element("follow-up").textContent = discard ? `Then ${actionText(discard).toLowerCase()}` : "";
  element("error").textContent = snapshot.error ?? "";
  element("click-through").textContent = snapshot.clickThrough
    ? "Click-through on · Ctrl+Shift+O to unlock" : "Enable click-through";
}
async function run(action) {
  try { await action(); }
  catch (error) { element("error").textContent = error.message; }
}
window.overlay.subscribe(render);
await run(async () => {
  const snapshot = await window.overlay.ready();
  element("base").value = snapshot.base;
  render(snapshot);
});
element("settings").addEventListener("submit", event => {
  event.preventDefault();
  run(async () => {
    const result = await window.overlay.configure({
      base: element("base").value.trim(), token: element("token").value.trim(),
    });
    element("base").value = result.base;
    element("token").value = "";
    element("connection").textContent = "Connected";
    element("error").textContent = "";
    document.querySelector("details").open = false;
  });
});
element("start").addEventListener("click", () => run(() => window.overlay.start()));
element("click-through").addEventListener("click", () => run(() => window.overlay.toggleClickThrough()));
element("quit").addEventListener("click", () => run(() => window.overlay.quit()));
