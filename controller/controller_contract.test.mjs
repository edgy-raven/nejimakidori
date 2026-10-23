import assert from "node:assert/strict";
import test from "node:test";
import { createGameplay } from "./mahjongsoul_gameplay.mjs";

test("a mismatched server action clears the expectation and pauses autoplay", () => {
  const events = [];
  let autoplay = true;
  const gameplay = createGameplay({
    buttonCapturePath: "/tmp/unused", casual: false,
    handleActionMismatch: () => {autoplay = false;},
    output: (type, value) => events.push({type, value}),
  });
  gameplay.expectedAction = {actor: 0, type: "dahai", pai: "9s"};
  gameplay.verifyExpectedAction({
    name: "ActionDiscardTile",
    data: {seat: 0, tile: "7s", is_liqi: false},
  });
  assert.equal(gameplay.expectedAction, null);
  assert.equal(autoplay, false);
  assert.deepEqual(events[0], {
    type: "action_mismatch",
    value: {
      expected: {actor: 0, type: "dahai", pai: "9s"},
      actual: {tile: "7s", riichi: false},
    },
  });
});
