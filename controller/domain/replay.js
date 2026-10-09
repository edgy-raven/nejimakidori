import { tileBase } from "./tile_values.js";

function removeTile(hand, tile) {
  let index = hand.indexOf(tile);
  if (index < 0 && /^[05][mps]$/.test(tile)) {
    index = hand.indexOf(`${tile[0] === "0" ? "5" : "0"}${tile[1]}`);
  }
  if (index < 0) index = hand.indexOf("?");
  if (index >= 0) {
    hand.splice(index, 1);
  }
}

function applyRiichi(state, liqi) {
  if (!liqi) return;
  state.riichi[liqi.seat] = true;
  state.scores[liqi.seat] = liqi.score;
  state.riichiSticks = liqi.sticks;
}

function markCalledRiver(state, event) {
  const calledIndex = event.froms.findIndex((from) => from !== event.seat);
  if (calledIndex < 0) return;
  const source = event.froms[calledIndex];
  const tile = event.tiles[calledIndex];
  for (let index = state.rivers[source].length - 1; index >= 0; index -= 1) {
    if (tileBase(state.rivers[source][index].tile) === tileBase(tile)) {
      state.rivers[source][index].called = true;
      return;
    }
  }
}

function applyEvent(state, event, previousEvent) {
  if (event.doras?.length) state.doras = event.doras;
  applyRiichi(state, event.liqi);
  if (event.type === "draw") {
    state.hands[event.seat].push(event.tile);
    state.drawn[event.seat] = event.tile;
    if (Number.isFinite(event.remaining)) state.remaining = event.remaining;
  } else if (event.type === "discard") {
    removeTile(state.hands[event.seat], event.tile);
    state.drawn[event.seat] = null;
    state.rivers[event.seat].push({
      tile: event.tile,
      riichi: event.riichi,
      tsumogiri: event.tsumogiri,
    });
    if (event.riichi) state.riichi[event.seat] = true;
  } else if (event.type === "call") {
    state.drawn[event.seat] = null;
    event.tiles.forEach((tile, index) => {
      if (event.froms[index] === event.seat) {
        removeTile(state.hands[event.seat], tile);
      }
    });
    markCalledRiver(state, event);
    state.melds[event.seat].push({ ...event, tiles: [...event.tiles] });
  } else if (event.type === "kan") {
    state.drawn[event.seat] = null;
    if (event.callType === "kakan") {
      removeTile(state.hands[event.seat], event.tiles[0]);
      const meld = state.melds[event.seat].find(
        (value) => value.callType === "pon" &&
          tileBase(value.tiles[0]) === tileBase(event.tiles[0])
      );
      if (meld) {
        meld.tiles.push(event.tiles[0]);
        meld.callType = "kakan";
      } else {
        state.melds[event.seat].push({ ...event, tiles: [...event.tiles] });
      }
    } else {
      event.tiles.forEach((tile) => removeTile(state.hands[event.seat], tile));
      state.melds[event.seat].push({ ...event, tiles: [...event.tiles] });
    }
  } else if (event.type === "agari") {
    state.handOver = true;
    state.winners = event.hules.map((winner) => ({
      ...winner, payer: winner.zimo ? winner.seat : previousEvent.seat,
    }));
    for (const winner of state.winners) {
      if (winner.hand.length) {
        state.hands[winner.seat] = [...winner.hand];
        state.revealed[winner.seat] = true;
      } else if (winner.zimo) {
        // A record without revealed tiles still has its last drawn tile.
        state.hands[winner.seat].pop();
      }
      state.drawn[winner.seat] = winner.zimo ? winner.tile : null;
    }
    state.riichiSticks = 0;
    if (event.hules.some((hule) => !hule.zimo) &&
        previousEvent?.type === "discard") {
      state.rivers[previousEvent.seat].at(-1).dealIn = true;
    }
    if (event.scores.length === 4) state.scores = [...event.scores];
    state.result = "Hand won";
  } else if (event.type === "ryuukyoku") {
    // Four riichi ends immediately after the final declaration: there is no
    // next draw to carry its acceptance, and the abort may omit liqi entirely.
    if (event.label === "Four riichi" && !event.liqi) {
      state.scores[previousEvent.seat] -= 1000;
      state.riichiSticks += 1;
    }
    state.handOver = true;
    state.tenpaiHands = event.tenpaiHands;
    state.readySeats = event.readySeats;
    for (const hand of state.tenpaiHands) {
      state.hands[hand.seat] = [...hand.tiles];
      state.drawn[hand.seat] = null;
      state.revealed[hand.seat] = true;
    }
    if (event.scores.length === 4) state.scores = [...event.scores];
    state.result = event.label;
  }
}

export function playbackState(round, eventCount = round.events.length) {
  const state = {
    hands: round.hands.map((hand) => [...hand]),
    rivers: [[], [], [], []],
    melds: [[], [], [], []],
    scores: [...round.scores],
    riichi: [false, false, false, false],
    riichiSticks: round.riichiSticks,
    doras: [...round.doraIndicators],
    remaining: null,
    result: "",
    winners: [],
    drawn: [null, null, null, null],
    handOver: false,
    tenpaiHands: [],
    readySeats: [],
    revealed: [false, false, false, false],
  };
  round.events.slice(0, eventCount).forEach((event, index) =>
    applyEvent(state, event, round.events[index - 1]));
  state.deltas = state.scores.map((score, seat) => score - round.scores[seat]);
  return state;
}

// A winner's published hand excludes the winning tile. During play, the
// tracked draw is still in hands and must be separated exactly once.
export function displayHand(state, seat) {
  const winner = state.winners.find((value) => value.seat === seat);
  const concealed = [...state.hands[seat]];
  const drawn = winner ? (winner.zimo ? winner.tile : null) : state.drawn[seat];
  if (!winner && drawn) {
    const index = concealed.lastIndexOf(drawn);
    if (index < 0) throw new Error("Drawn tile is missing from tracked hand");
    concealed.splice(index, 1);
  }
  return {concealed, drawn};
}
