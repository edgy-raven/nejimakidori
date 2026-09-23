import { shantenDistribution, placementBar } from "./probabilities.js";
import { accountRankText } from "./ranks.js";
import { displayHand } from "./replay.js";
import { tileElement, tilesElement, sortTiles } from "./tiles.js";
import { actionName, seatName } from "./labels.js";
import { fitRack } from "./board-layout.js";

export function renderDora(indicators) {
  const wall = document.querySelector("#live-dora");
  wall.className = "dora-wall";
  wall.setAttribute("aria-label", "Dora indicators on the dead wall");
  wall.replaceChildren(...Array.from({length: 5}, (_, index) => {
    const stack = document.createElement("span");
    stack.className = "dora-wall-stack";
    const indicator = indicators[index];
    const tile = tileElement(indicator || "?");
    if (!indicator) {
      tile.textContent = "";
      tile.classList.add("tile-back");
    }
    stack.append(tile);
    return stack;
  }));
}

export function renderSeat(seat, position, state) {
  const relativeSeat = (seat - position.selfSeat + 4) % 4;
  const element = document.querySelector(`[data-live-seat="${relativeSeat}"]`);
  const heading = document.createElement("div");
  heading.className = "seat-heading";
  element.replaceChildren();
  const rank = state.scores.filter((score, otherSeat) =>
    score > state.scores[seat] || (score === state.scores[seat] && otherSeat < seat)
  ).length + 1;
  const name = document.createElement("strong");
  const username = document.createElement("span");
  username.className = "player-name";
  username.textContent = position.players?.find(player => player.seat === seat)?.nickname || seatName(seat, position.selfSeat, position.round.kyoku).split(" · ")[0];
  const rankRow = document.createElement("span");
  rankRow.className = "player-rank-row";
  const standing = document.createElement("small");
  standing.className = `player-standing placement-${rank}`;
  standing.textContent = ["1st", "2nd", "3rd", "4th"][rank - 1];
  rankRow.append(standing);
  const accountRank = accountRankText(position.accountRanks?.[seat]);
  if (accountRank) {
    const label = document.createElement("small");
    label.className = "account-rank";
    label.textContent = accountRank;
    rankRow.append(label);
  }
  name.append(username, rankRow);
  heading.append(name);
  const wonHand = state.winners?.find((winner) => winner.seat === seat);
  if (wonHand) {
    heading.classList.add("hand-winner");
  }
  heading.dataset.cardSeat = relativeSeat;
  heading.classList.add("opponent-card", `opponent-card-${relativeSeat}`);
  const {concealed, drawn} = displayHand(state, seat);
  const hand = tilesElement(sortTiles(concealed), "hand");
  hand.dataset.step = position.step;
  const drawnSlot = document.createElement("span");
  drawnSlot.className = "drawn-slot";
  if (drawn) {
    const tile = tileElement(drawn);
    tile.classList.add("live-drawn-tile");
    if (wonHand) tile.classList.add("winning-tsumo");
    drawnSlot.append(tile);
  }
  hand.append(drawnSlot);
  hand.dataset.revealed = String(state.revealed?.[seat] === true);
  if (seat !== position.selfSeat && !state.revealed?.[seat]) {
    hand.classList.add("hand-hidden");
  }
  const rack = document.createElement("div");
  rack.className = "seat-rack";
  const rackContent = document.createElement("div");
  rackContent.className = "seat-rack-content";
  rackContent.append(hand);
  const melds = document.createElement("div");
  melds.className = "melds";
  [...state.melds[seat]].reverse().forEach((meld) => {
    const group = document.createElement("div");
    group.className = "meld";
    group.dataset.type = meld.callType;
    group.setAttribute("aria-label", actionName(meld.callType));
    const calledIndex = meld.froms?.findIndex((from) => from !== seat) ?? -1;
    const tiles = meld.tiles.map((tile, index) => ({tile, called: index === calledIndex}));
    if (calledIndex >= 0) {
      const [called] = tiles.splice(calledIndex, 1);
      const source = (meld.froms[calledIndex] - seat + 4) % 4;
      tiles.splice(source === 3 ? 0 : source === 2 ? 1 : tiles.length, 0, called);
    }
    tiles.forEach(({tile, called}, index) => {
      const element = tileElement(tile);
      if (meld.callType === "ankan" && (index === 0 || index === 3)) {
        element.classList.add("tile-back");
        element.textContent = "";
      }
      if (called) {
        const slot = document.createElement("span");
        slot.className = "meld-called-tile";
        slot.append(element);
        group.append(slot);
      } else {
        group.append(element);
      }
    });
    melds.append(group);
  });
  rackContent.append(melds);
  rack.append(rackContent, heading);
  if (seat === position.selfSeat) rack.append(placementBar());
  else {
    const speed = document.createElement("div");
    speed.className = "live-opponent-speed";
    speed.dataset.speedSeat = seat;
    speed.setAttribute("aria-label", "Distance from ready");
    speed.append(shantenDistribution());
    rack.append(speed);
  }

  element.append(rack);
  fitRack(rack);

}

export function renderRiver(seat, selfSeat, state) {
  const relativeSeat = (seat - selfSeat + 4) % 4;
  const element = document.querySelector(
    `[data-live-river="${relativeSeat}"]`
  );
  element.replaceChildren();
  let discardsInRow = 0;
  state.rivers[seat].forEach((discard) => {
    if (!element.lastElementChild ||
      (element.children.length < 3 && discardsInRow === 6 && !discard.called)) {
      discardsInRow = 0;
      const row = document.createElement("div");
      row.className = "river-row";
      element.append(row);
    }
    const slot = document.createElement("span");
    slot.className = discard.riichi ? "river-slot river-riichi-slot" : "river-slot";
    slot.append(tileElement(discard.tile, discard));
    element.lastElementChild.append(slot);
    if (!discard.called) discardsInRow += 1;
  });
}
