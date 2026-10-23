import assert from "node:assert/strict";
import test from "node:test";
import {createGameplay} from "./mahjongsoul_gameplay.mjs";

test("another accepted call or ron clears a pending call without reporting a mismatch", () => {
  const events = [];
  const game = createGameplay({output: (name, value) => events.push({name, value}),
    handleActionMismatch: () => assert.fail("Valid call priority is not a mismatch")});
  game.expectedAction = {actor: 3, type: "chi", pai: "7s", consumed: ["5s", "6s"]};
  game.verifyExpectedAction({name: "ActionChiPengGang", data: {
    seat: 1, type: 1, tiles: ["7s", "7s", "7s"], froms: [1,1,2],
  }});
  assert.equal(game.expectedAction, null);
  assert.equal(game.expectedActionAt, null);
  game.verifyExpectedAction({name: "ActionDealTile", data: {seat: 2}});
  assert.deepEqual(events.map(event => event.name), ["call_superseded"]);
  game.expectedAction = {actor: 3, type: "pon", pai: "7s", consumed: ["7s", "7s"]};
  game.verifyExpectedAction({name: "ActionHule", data: {hules: [{seat: 0}]}});
  assert.equal(game.expectedAction, null);
  assert.equal(events[1].name, "call_superseded");
});
