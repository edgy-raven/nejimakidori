import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";

const layout = fs.readFileSync(new URL("./public/shared/board-layout.js", import.meta.url), "utf8");
const styles = fs.readFileSync(new URL("./public/shared/board.css", import.meta.url), "utf8");
const tiles = fs.readFileSync(new URL("./public/shared/tiles.js", import.meta.url), "utf8");

test("board fitting reserves the external self hand once", () => {
  assert.match(layout, /availableHeight \/ \(height \+ handHeight \+ handGap\)/);
  assert.match(layout, /mainHand.style.height = `\$\{handHeight \* scale\}px`/);
  assert.match(layout, /viewport.style.height = `\$\{height \* scale\}px`/);
});

test("the external hand scales atlas tiles and is outside the three-sided table", () => {
  assert.match(tiles, /export function scaleTileSprite/);
  assert.match(styles, /--seat-clearance: 12px/);
  assert.match(styles, /transform-origin: top left/);
  assert.match(styles, /width: calc\(14 \* var\(--tile-width\) \+ 18px\)/);
});
