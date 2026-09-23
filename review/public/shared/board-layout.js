import { scaleTileSprite } from "./tiles.js";

export function observeBoard() {
  const boardViewport = document.querySelector(".board-viewport");
  const mainHand = document.querySelector(".main-hand");
  function fitBoard() {
    const board = boardViewport.querySelector(".board");
    const mainHandStyle = mainHand && getComputedStyle(mainHand);
    // The self hand is outside the table. Reserve its rendered height once,
    // rather than treating it as a second bottom rack in the table geometry.
    const selfHandHeight = mainHand
      ? mainHand.getBoundingClientRect().height + parseFloat(mainHandStyle.marginTop)
      : 0;
    const availableHeight = innerHeight - Math.max(0, boardViewport.getBoundingClientRect().top)
      - selfHandHeight - 12;
    const style = getComputedStyle(board);
    const tileWidth = parseFloat(style.getPropertyValue("--tile-width"));
    const tileHeight = parseFloat(style.getPropertyValue("--tile-height"));
    // Seven center widths, three discard rows on each side, and only the
    // across player's rack. The self rack lives below the table.
    // Each river has two 1px gaps; racks have 37px clearance; edges use 14px.
    const size = 7 * tileWidth + 2 * (3 * tileHeight + 2 + 14) + tileHeight + 37;
    const scale = Math.min(boardViewport.clientWidth / size, availableHeight / (size + 14));
    if (scale <= 0) return;
    board.style.setProperty("--board-scale", scale);
    const width = boardViewport.clientWidth / scale;
    board.style.width = `${width}px`;
    board.style.height = `${availableHeight / scale}px`;
    board.style.marginLeft = `${-width / 2}px`;
    board.style.transform = `scale(${scale})`;
    boardViewport.style.height = `${availableHeight}px`;
    board.querySelectorAll(".seat-rack").forEach(fitRack);
    fitNameCards(board);

  }
  const observer = new ResizeObserver(fitBoard);
  observer.observe(boardViewport);
  if (mainHand) observer.observe(mainHand);
  window.addEventListener("resize", fitBoard);

}

export function fitRack(rack) {
  const seat = rack.parentElement;
  const side = ["1", "3"].includes(seat.dataset.liveSeat);
  const field = seat.parentElement.querySelector(".table-field");
  const available = seat.parentElement.clientWidth - 4;
  if (available <= 0) return; // The initial board is hidden until a hand loads.
  if (side) {
    rack.style.top = `${100 * (field.offsetTop - seat.offsetTop + field.clientHeight / 2) / seat.clientHeight}%`;
  }
  const tileWidth = parseFloat(getComputedStyle(rack).getPropertyValue("--tile-width"));
  rack.querySelectorAll(".tile").forEach(tile => scaleTileSprite(tile, tileWidth / 25));
  const handWidth = rack.querySelector(".hand").scrollWidth;
  const meldWidth = rack.querySelector(".melds").scrollWidth;
  rack.style.width = `${Math.min(available, Math.max(14 * tileWidth + 18, handWidth + meldWidth + 12))}px`;
  rack.querySelector(".opponent-card").style.left = `${Math.max(-36, (rack.clientWidth - available) / 2 + 12)}px`;
  fitNameCards(seat.parentElement);
}

function fitNameCards(board) {
  board.querySelectorAll(".player-name").forEach(name => {
    name.style.fontSize = "24px";
    name.style.fontSize = `${24 * Math.min(1, name.clientWidth / name.scrollWidth)}px`;
  });
}
