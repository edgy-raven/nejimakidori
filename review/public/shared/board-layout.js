// Live and review use the same seat structure and one scene scale.
export function observeBoard() {
  const viewport = document.querySelector(".board-viewport");
  const board = viewport.querySelector(".board");
  const mainHand = document.querySelector(".main-hand");
  document.querySelectorAll("[data-live-seat]").forEach(seat => {
    seat.innerHTML = '<div class="seat-rack"><div class="seat-info"><div class="seat-bar"></div></div><div class="seat-rack-content"></div></div>';
    const card = document.createElement("div");
    card.className = "player-card";
    card.dataset.cardSeat = seat.dataset.liveSeat;
    seat.querySelector(".seat-info").prepend(card);
  });
  mainHand.append(mainHand.querySelector(".seat-info"));
  function fitBoard() {
    const style = getComputedStyle(board);
    const width = viewport.clientWidth;
    if (width === 0) return; // Both pages initially hide the board.
    const height = board.offsetHeight;
    const handHeight = parseFloat(style.getPropertyValue("--self-height"));
    const handGap = parseFloat(style.getPropertyValue("--hand-gap"));
    const availableHeight = innerHeight - Math.max(0, viewport.getBoundingClientRect().top) - 12;
    const scale = Math.min(width / parseFloat(style.getPropertyValue("--board-min-width")), availableHeight / (height + handHeight + handGap));
    if (scale <= 0) return;
    viewport.parentElement.style.setProperty("--board-scale", scale);
    board.style.width = `${width / scale}px`;
    viewport.style.height = `${height * scale}px`;
    mainHand.style.width = `${width}px`;
    mainHand.style.height = `${handHeight * scale}px`;
    fitPlayerNames(viewport.parentElement);
  }
  new ResizeObserver(fitBoard).observe(viewport);
  window.addEventListener("resize", fitBoard);
  document.fonts.ready.then(fitBoard);
}

export function fitPlayerNames(element) {
  element.querySelectorAll(".player-name").forEach(name => {
    name.style.fontSize = "20px";
    if (name.clientWidth === 0) return; // Hidden boards fit when shown.
    name.style.fontSize = `${20 * Math.min(1, (name.clientWidth - 1) / name.scrollWidth)}px`;
  });
}
