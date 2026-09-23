import { tileText } from "./tiles.js";

export function shantenText(value) {
  if (value < 0) {
    return "Complete";
  }
  if (value === 0) {
    return "Ready";
  }
  return `${value} away`;
}

export function actionName(type) {
  return {
    none: "Pass", dahai: "Discard", chi: "Chi", chii: "Chi",
    pon: "Pon", daiminkan: "Open quad", ankan: "Closed quad",
    kan: "Quad", kakan: "Added quad", reach: "Reach", riichi: "Reach",
    hora: "Win", ron: "Win on discard", tsumo: "Self-draw win",
    ryukyoku: "Drawn hand", nukidora: "Extract North",
    DISCARD: "Discard", RESPONSE: "Respond to discard",
    KAN: "Quad decision", RIICHI: "Reach declaration",
  }[type] || "Game action";
}

export function actionText(action) {
  if (!action) return "—";
  if (action.type === "none") return "Pass";
  if (action.type === "pon") return "Pon";
  const tile = action.pai ? ` ${tileText(action.pai)}` : "";
  const consumed = action.consumed?.length
    ? ` (${action.consumed.map(tileText).join(", ")})`
    : "";
  return `${actionName(action.type)}${tile}${consumed}`;
}

export function roundText(round) {
  return `${["East", "South", "West", "North"][round.prevailingWind]} ${round.kyoku}` +
    (round.honba ? ` · ${round.honba} ${round.honba === 1 ? "repeat" : "repeats"}` : "");
}

export function seatName(seat, selfSeat, kyoku) {
  const names = ["Self", "Right", "Across", "Left"];
  const winds = ["East", "South", "West", "North"];
  return `${names[(seat - selfSeat + 4) % 4]} · ` +
    `[${winds[(seat - (kyoku - 1) + 4) % 4]}]`;
}

export function winText(winner, selfSeat, kyoku) {
  const source = winner.zimo ? "All"
    : seatName(winner.payer, selfSeat, kyoku).split(" · ")[0];
  return `${source} → ${seatName(winner.seat, selfSeat, kyoku).split(" · ")[0]}`;
}

export function englishLabel(name) {
  const special = {
    "Tsubame-gaeshi": "Swallow reversal", "Kanburi": "Win after a quad discard",
    "Shiiaruraotai": "Four exposed sets", "Uumensai": "Five gates",
    "Iipinmoyue": "Moon from the sea", "Chuupinraoyui": "Fish from the river",
    "Ishinouenimosannen": "Three years on a stone",
  };
  return (special[name] || name).replace(/riichi/gi, "Reach")
    .replace(/Ippatsu/g, "One shot").replace(/Ura Dora/g, "Hidden dora")
    .replace(/Kans?/g, "Quad").replace(/kans/g, "quads")
    .replace(/Kita/g, "North extraction")
    .replace(/Yakuless/g, "No-pattern").replace(/Triple ron/g, "Three discard winners");
}


// Hand value before repeat counters and reach deposits; settlement is below.
export function winPoints(winner, kyoku) {
  const yakuman = winner.yaku.reduce((total, yaku) => total + (yaku.yakuman ? yaku.value : 0), 0);
  const basic = yakuman ? 8000 * yakuman
    : winner.han >= 13 ? 8000
    : winner.han >= 11 ? 6000
    : winner.han >= 8 ? 4000
    : winner.han >= 6 ? 3000
    : Math.min(2000, winner.fu * 2 ** (winner.han + 2));
  const limit = yakuman ? (yakuman === 1 ? 'Yakuman' : `${yakuman}× Yakuman`)
    : winner.han >= 13 ? 'Counted Yakuman'
    : basic === 6000 ? 'Sanbaiman'
    : basic === 4000 ? 'Baiman'
    : basic === 3000 ? 'Haneman'
    : basic === 2000 ? 'Mangan' : '';
  const dealer = winner.seat === kyoku - 1;
  const payments = winner.zimo ? (dealer ? [basic * 2] : [basic, basic * 2])
    : [basic * (dealer ? 6 : 4)];
  return payments.map(points => Math.ceil(points / 100) * 100).join('/')
    + (winner.zimo && dealer ? ' all' : '') + (limit ? ` [${limit}]` : '');
}
