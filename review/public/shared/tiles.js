import { tileBase } from "./tile_values.js";
export { tileBase } from "./tile_values.js";

const tileRects = await fetch("../assets/tiles_rects.json").then((response) =>
  response.json()
);

// Recolor only dark face markings, preserving the atlas's face and bevel.
const tileFilters = document.createElementNS("http://www.w3.org/2000/svg", "svg");
tileFilters.setAttribute("width", "0");
tileFilters.setAttribute("height", "0");
tileFilters.style.position = "absolute";
tileFilters.innerHTML = `<defs><filter id="red-five-ink" color-interpolation-filters="sRGB">
  <feFlood x="5%" y="12%" width="90%" height="82%" flood-color="#b82025"/>
  <feComposite in2="SourceAlpha" operator="in" result="ink"/>
  <feColorMatrix in="SourceGraphic" type="matrix"
    values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  .2126 .7152 .0722 0 0"/>
  <feComponentTransfer><feFuncA type="discrete" tableValues="1 1 1 0 0 0 0 0"/></feComponentTransfer>
  <feComposite in="ink" operator="in"/>
  <feMerge><feMergeNode in="SourceGraphic"/><feMergeNode/></feMerge>
</filter></defs>`;
document.body.append(tileFilters);


export function tileText(tile) {
  if (!tile || tile === "?") return "Unknown tile";
  const base = tileBase(tile);
  if (base[1] === "z") {
    return ["East", "South", "West", "North", "White dragon",
      "Green dragon", "Red dragon"][Number(base[0]) - 1];
  }
  return (tile.startsWith("0") || tile.endsWith("r") ? "Red " : "") +
    `${base[0]} ${{m: "characters", s: "bamboos", p: "circles"}[base[1]]}`;
}

export function sortTiles(tiles) {
  const suits = { m: 0, p: 1, s: 2, z: 3 };
  return [...tiles].sort((left, right) => {
    const leftRank = left[0] === "0" ? 5 : Number(left[0]);
    const rightRank = right[0] === "0" ? 5 : Number(right[0]);
    return suits[left[1]] * 10 + leftRank -
      (suits[right[1]] * 10 + rightRank);
  });
}

export function scaleTileSprite(element, scale = 0.3125) {
  const base = tileBase(element.dataset.tile);
  const suit = "mpsz".indexOf(base?.[1]);
  const tileId = suit < 3
    ? suit * 9 + Number(base?.[0])
    : 27 + Number(base?.[0]);
  const rect = tileRects[tileId - 1];
  if (!rect) return;
  element.style.backgroundSize = `${1464 * scale}px ${531 * scale}px`;
  element.style.backgroundPosition = `${-rect.x * scale}px ${-rect.y * scale}px`;
}

export function tileElement(tile, options = {}) {
  const element = document.createElement("span");
  element.dataset.tile = tile;
  element.className = `tile suit-${tile?.[1] || "x"}`;
  if (tile?.startsWith("0") || tile?.endsWith("r")) {
    element.classList.add("red-five");
  }
  if (options.called) {
    element.classList.add("called", "tile-back");
  }
  if (options.riichi) element.classList.add("riichi-tile");
  if (options.tsumogiri) element.classList.add("tsumogiri");
  if (options.dealIn) element.classList.add("deal-in");
  if (options.draw) element.classList.add("current-draw");
  const base = tileBase(tile);
  const suit = "mpsz".indexOf(base?.[1]);
  const tileId = suit < 3
    ? suit * 9 + Number(base?.[0])
    : 27 + Number(base?.[0]);
  const rect = tileRects[tileId - 1];
  if (rect) {
    element.style.backgroundImage = 'url("../assets/tiles.png")';
    scaleTileSprite(element);
  } else {
    element.textContent = tile || "?";
  }
  if (options.called) element.textContent = "";
  element.setAttribute("aria-label", tileText(tile));
  return element;
}

export function tilesElement(tiles, className, options = {}) {
  const element = document.createElement("div");
  element.className = className;
  tiles.forEach((tile, index) =>
    element.append(
      tileElement(tile, Array.isArray(options) ? options[index] || {} : options)
    )
  );
  return element;
}
