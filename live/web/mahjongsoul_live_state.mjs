import { terminalEvent } from "./mahjongsoul_terminal.mjs";
import { playbackState } from "../../review/public/shared/replay.js";

const callTypes = ["chii", "pon", "daiminkan"];

function strings(value) {
  return Array.isArray(value) ? value.map(String) : [];
}

function numbers(value) {
  return Array.isArray(value) ? value.map(Number) : [];
}

function operationSeat(data) {
  return data.operation && Number.isInteger(Number(data.operation.seat))
    ? Number(data.operation.seat)
    : null;
}

function acceptedRiichi(data) {
  return data.liqi
    ? {
        seat: Number(data.liqi.seat),
        score: Number(data.liqi.score),
        sticks: Number(data.liqi.liqibang),
      }
    : null;
}

function doras(data) {
  const values = strings(data.doras);
  return values.length ? values : data.dora ? [String(data.dora)] : [];
}

function removeTile(tiles, tile) {
  const index = tiles.indexOf(tile);
  if (index === -1) {
    throw new Error(`live hand does not contain ${tile}`);
  }
  tiles.splice(index, 1);
}

function consumedKanTiles(concealedTiles, drawnTile, tile, count) {
  const alternatives = tile[0] === "0"
    ? [tile, `5${tile[1]}`]
    : (tile[0] === "5" && "mps".includes(tile[1])
      ? [tile, `0${tile[1]}`]
      : [tile]);
  const consumed = [...concealedTiles, ...(drawnTile ? [drawnTile] : [])]
    .filter((value) => alternatives.includes(value))
    .slice(0, count);
  if (consumed.length !== count) {
    throw new Error(`live hand does not contain ${count} copies of ${tile}`);
  }
  return consumed;
}

function event(name, data) {
  if (name === "ActionDealTile") {
    return {
      type: "draw",
      seat: Number(data.seat),
      tile: String(data.tile || "?"),
      remaining: Number(data.left_tile_count),
      doras: doras(data),
      liqi: acceptedRiichi(data),
    };
  }
  if (name === "ActionDiscardTile") {
    return {
      type: "discard",
      seat: Number(data.seat),
      tile: String(data.tile),
      tsumogiri: Boolean(data.moqie),
      riichi: Boolean(data.is_liqi || data.is_wliqi),
      doras: doras(data),
    };
  }
  if (name === "ActionChiPengGang") {
    return {
      type: "call",
      seat: Number(data.seat),
      callType: callTypes[Number(data.type)] || "call",
      tiles: strings(data.tiles),
      froms: numbers(data.froms),
      liqi: acceptedRiichi(data),
    };
  }
  if (name === "ActionAnGangAddGang") {
    const callType = Number(data.type) === 3 ? "ankan" : "kakan";
    const tile = String(data.tiles);
    return {
      type: "kan",
      seat: Number(data.seat),
      callType,
      tiles: callType === "ankan" ? [tile, tile, tile, tile] : [tile],
      doras: doras(data),
    };
  }
  return terminalEvent(name.replace(/^Action/, ""), data);
}

export class MahjongSoulLiveState {
  constructor() {
    this.selfSeat = null;
    this.players = [];
    this.accountRanks = [null, null, null, null];
    this.round = null;
    this.lastHandResult = null;
    this.finalMatchResult = null;
    this.eastOnly = false;
  }

  apply(action) {
    const { name, data } = action;
    if (name === "NotifyGameEndResult") {
      this.finalMatchResult = {
        selfSeat: this.selfSeat,
        players: [...data.result.players].sort((a, b) => b.total_point - a.total_point)
          .map((player, index) => ({
            seat: Number(player.seat), rank: index + 1,
            score: Number(player.part_point_1),
          })),
      };
      return null;
    }
    let normalized = null;
    if (name === "ActionNewRound") {
      const dealer = Number(data.ju);
      const privateTiles = strings(data.tiles);
      if (privateTiles.length === 14) {
        this.selfSeat = dealer;
      } else if (operationSeat(data) !== null) {
        this.selfSeat = operationSeat(data);
      }
      this.round = {
        prevailingWind: Number(data.chang),
        eastOnly: this.eastOnly,
        kyoku: dealer + 1,
        honba: Number(data.ben),
        riichiSticks: Number(data.liqibang),
        scores: numbers(data.scores),
        doraIndicators: doras(data),
        hands: Array.from({ length: 4 }, () => Array(13).fill("?")),
        events: [],
      };
      this.privateTiles = privateTiles.slice(0, 13);
      this.concealedTiles = [...this.privateTiles];
      this.drawnTile = privateTiles.length === 14 ? privateTiles[13] : null;
      this.meldCount = 0;
      this.riichiSeats = new Set();
      this.round.events.push({
        type: "draw",
        seat: dealer,
        tile: privateTiles.length === 14 ? privateTiles[13] : "?",
        remaining: Number(data.left_tile_count),
        doras: doras(data),
        liqi: null,
      });
    } else if (this.round) {
      normalized = event(name, data);
      if (normalized) {
        if (
          name === "ActionAnGangAddGang" &&
          Number(data.seat) === this.selfSeat
        ) {
          normalized.tiles = consumedKanTiles(
            this.concealedTiles,
            this.drawnTile,
            String(data.tiles),
            Number(data.type) === 3 ? 4 : 1,
          );
        }
        this.round.events.push(normalized);
      }
    }

    if (!this.round) {
      return {
        action: name,
        step: action.step,
        selfSeat: this.selfSeat,
        accountRanks: this.accountRanks,
      players: this.players,
        phase: null,
        operation: data.operation,
        round: null,
        hand: null,
      };
    }

    if (name === "ActionDealTile" && data.tile) {
      this.selfSeat = Number(data.seat);
      this.drawnTile = String(data.tile);
    }
    if (operationSeat(data) !== null) {
      this.selfSeat = operationSeat(data);
    }
    if (
      name === "ActionDiscardTile" &&
      (data.is_liqi || data.is_wliqi)
    ) {
      this.riichiSeats.add(Number(data.seat));
    }
    if (
      name === "ActionDiscardTile" &&
      Number(data.seat) === this.selfSeat
    ) {
      const tile = String(data.tile);
      if (
        data.moqie ||
        (this.drawnTile === tile && !this.concealedTiles.includes(tile))
      ) {
        if (this.drawnTile !== tile) {
          throw new Error("tsumogiri does not match the tracked draw");
        }
      } else {
        removeTile(this.concealedTiles, tile);
        if (this.drawnTile) {
          this.concealedTiles.push(this.drawnTile);
        }
      }
      this.drawnTile = null;
    }
    if (
      name === "ActionChiPengGang" &&
      Number(data.seat) === this.selfSeat
    ) {
      data.tiles.forEach((tile, index) => {
        if (Number(data.froms[index]) === this.selfSeat) {
          removeTile(this.concealedTiles, String(tile));
        }
      });
      this.meldCount += 1;
    }
    if (
      name === "ActionAnGangAddGang" &&
      Number(data.seat) === this.selfSeat
    ) {
      for (const tile of normalized.tiles) {
        if (this.drawnTile === tile) {
          this.drawnTile = null;
        } else {
          removeTile(this.concealedTiles, tile);
        }
      }
      if (this.drawnTile) {
        this.concealedTiles.push(this.drawnTile);
        this.drawnTile = null;
      }
      if (Number(data.type) === 3) {
        this.meldCount += 1;
      }
    }
    if (this.round && this.selfSeat !== null) {
      this.round.hands[this.selfSeat] = this.privateTiles;
    }

    if (["ActionHule", "ActionNoTile", "ActionLiuJu"].includes(name)) {
      const completed = playbackState(this.round);
      this.lastHandResult = {
        prevailingWind: this.round.prevailingWind,
        kyoku: this.round.kyoku,
        honba: this.round.honba,
        selfSeat: this.selfSeat,
        outcome: name === "ActionHule" ? "Win" : normalized.label,
        winners: completed.winners.map((winner) => ({
          ...winner, tsumo: winner.zimo,
        })),
        tenpaiHands: completed.tenpaiHands,
        scores: completed.scores,
        deltas: completed.deltas,
      };
    }

    let phase = null;
    if (
      this.selfSeat !== null &&
      ((name === "ActionNewRound" && data.tiles.length === 14) ||
        (name === "ActionDealTile" && Number(data.seat) === this.selfSeat) ||
        (name === "ActionChiPengGang" &&
          Number(data.seat) === this.selfSeat &&
          Number(data.type) !== 2))
    ) {
      phase = "DISCARD";
    } else if (
      this.selfSeat !== null &&
      operationSeat(data) === this.selfSeat &&
      name !== "ActionNewRound"
    ) {
      phase = "RESPONSE";
    }
    const handOver = ["agari", "ryuukyoku"].includes(
      this.round.events.at(-1)?.type);
    return {
      action: name,
      step: action.step,
      selfSeat: this.selfSeat,
      accountRanks: this.accountRanks,
      players: this.players,
      phase: handOver ? null : phase,
      operation: data.operation,
      round: this.round,
      hand: {
        concealed: [...this.concealedTiles],
        drawn: this.drawnTile,
        meldCount: this.meldCount,
        riichi: this.riichiSeats.has(this.selfSeat),
      },
    };
  }
}
