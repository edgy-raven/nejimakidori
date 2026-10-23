import { yakuNames } from "./mahjongsoul_yaku.mjs";

// Shared by recorded and live protocol adapters. Repeated protobuf fields may
// be omitted when empty; hand excludes the separately supplied winning tile.
export function terminalEvent(name, data) {
  if (name === "Hule") {
    return {
      type: "agari",
      hules: data.hules.map((hule) => ({
        seat: Number(hule.seat), tile: String(hule.hu_tile),
        hand: (hule.hand || []).map(String), zimo: Boolean(hule.zimo),
        title: hule.title || "Win", han: Number(hule.count), fu: Number(hule.fu),
      doraIndicators: hule.doras || [], uraIndicators: hule.li_doras || [],
        yaku: (hule.fans || []).map((fan) => ({
          name: yakuNames[fan.id].name, value: Number(fan.val),
          yakuman: yakuNames[fan.id].yakuman,
        })),
      })),
      scores: data.scores.map(Number),
      deltas: (data.delta_scores || []).map(Number),
    };
  }
  if (name === "NoTile") {
    const deltas = Array.from({length: 4}, (_, seat) =>
      data.scores.reduce((sum, payment) => sum +
        (payment.delta_scores.length ? Number(payment.delta_scores[seat]) : 0), 0));
    return {
      type: "ryuukyoku", label: data.liujumanguan ? "Nagashi mangan" : "Exhaustive draw",
      deltas,
      scores: data.scores.length ? data.scores[0].old_scores.map((score, seat) =>
        Number(score) + deltas[seat]) : [],
      readySeats: data.players.flatMap((player, seat) => player.tingpai ? [seat] : []),
      tenpaiHands: data.players.flatMap((player, seat) =>
        player.tingpai && player.hand.length
          ? [{seat, tiles: player.hand.map(String)}] : []),
    };
  }
  if (name === "LiuJu") {
    return {
      type: "ryuukyoku",
      label: ({1: "Nine terminals and honors", 2: "Four winds",
        3: "Four kans", 4: "Four riichi", 5: "Triple ron"})[data.type] || "Abortive draw",
      scores: (data.gameend?.scores || []).map(Number),
      deltas: [0, 0, 0, 0],
      tenpaiHands: [], readySeats: [],
      liqi: data.liqi ? {seat: Number(data.liqi.seat),
        score: Number(data.liqi.score), sticks: Number(data.liqi.liqibang)} : null,
    };
  }
  return null;
}
