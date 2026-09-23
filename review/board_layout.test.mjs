import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";

const layout = fs.readFileSync(new URL("./public/shared/board-layout.js", import.meta.url), "utf8");
const styles = fs.readFileSync(new URL("./public/shared/styles.css", import.meta.url), "utf8");
const tiles = fs.readFileSync(new URL("./public/shared/tiles.js", import.meta.url), "utf8");

test("board fitting reserves the external self hand once", () => {
  assert.match(layout, /const selfHandHeight = mainHand/);
  assert.match(layout, /- selfHandHeight - 12/);
  assert.match(layout, /2 \* \(3 \* tileHeight \+ 2 \+ 14\) \+ tileHeight \+ 37/);
  assert.doesNotMatch(layout, /2 \* \(3 \* tileHeight \+ 2 \+ tileHeight \+ 37 \+ 14\)/);
});

test("the external hand scales atlas tiles and is outside the three-sided table", () => {
  assert.match(tiles, /export function scaleTileSprite/);
  assert.match(layout, /scaleTileSprite\(tile, tileWidth \/ 25\)/);
  assert.match(styles, /grid-template-areas: "left top right" "left center right"/);
  assert.doesNotMatch(styles, /"left bottom right"/);
  assert.match(styles, /border-bottom: 0/);
});
