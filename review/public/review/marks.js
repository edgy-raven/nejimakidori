import {tileBase, tileText} from "../shared/tiles.js";

// Follow the recorded move from the hand into the river or meld.
export function markMistake(round, eventCount, decision, category, povSeat) {
  const event = round.events[decision];
  const mark = element => {
    element.classList.add("review-mistake");
    element.dataset.mistakeDecision = decision;
    element.setAttribute("role", "button");
    element.tabIndex = 0;
    element.dataset.penaltyCategory = category;
    element.setAttribute("aria-label",
      `${element.getAttribute("aria-label")} · Go to mistake`);
  };
  if (eventCount === decision) {
    const hand = [...document.querySelectorAll('[data-live-seat="0"] .hand > .tile')];
    if (event.type === "discard") {
      mark(hand.find(tile => tile.getAttribute("aria-label") === tileText(event.tile) &&
        tile.classList.contains("live-drawn-tile") === event.tsumogiri));
    } else {
      const consumed = event.type === "call" ? event.tiles.filter((_, index) =>
        event.froms[index] === povSeat) : event.tiles;
      for (const value of consumed) {
        const index = hand.findIndex(tile =>
          tile.getAttribute("aria-label") === tileText(value));
        mark(hand.splice(index, 1)[0]);
      }
      if (event.type === "call") {
        const source = event.froms.find(seat => seat !== povSeat);
        mark([...document.querySelectorAll(
          `[data-live-river="${(source - povSeat + 4) % 4}"] .tile`,
        )].at(-1));
      }
    }
  } else if (event.type === "discard") {
    const index = round.events.slice(0, decision).filter(value =>
      value.type === "discard" && value.seat === povSeat).length;
    mark(document.querySelectorAll('[data-live-river="0"] .tile')[index]);
  } else {
    const meldEvents = round.events.slice(0, eventCount).filter(value =>
      value.seat === povSeat && ["call", "kan"].includes(value.type) &&
      value.callType !== "kakan");
    const index = event.callType === "kakan" ? meldEvents.findIndex(value =>
      value.callType === "pon" && tileBase(value.tiles[0]) === tileBase(event.tiles[0]))
      : meldEvents.indexOf(event);
    const meld = document.querySelectorAll('[data-live-seat="0"] .meld')[
      meldEvents.length - 1 - index];
    meld.querySelectorAll(".tile").forEach(mark);
  }
}
