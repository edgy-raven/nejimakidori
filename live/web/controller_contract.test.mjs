import assert from "node:assert/strict";
import test from "node:test";
import { FriendlySession } from "./mahjongsoul_friendly.mjs";
import { createGameplay } from "./mahjongsoul_gameplay.mjs";
import { Autoqueue } from "./mahjongsoul_ui_controls.mjs";

test("a mismatched server action clears the expectation and stops requeue", () => {
  const events = [];
  const queue = new Autoqueue();
  queue.enabled = true;
  const gameplay = createGameplay({
    buttonCapturePath: "/tmp/unused", casual: false,
    handleActionMismatch: verification => queue.stop(
      `Action mismatch: ${verification.actual.tile}`,
    ),
    output: (type, value) => events.push({type, value}),
  });
  gameplay.expectedAction = {actor: 0, type: "dahai", pai: "9s"};
  gameplay.verifyExpectedAction({
    name: "ActionDiscardTile",
    data: {seat: 0, tile: "7s", is_liqi: false},
  });
  assert.equal(gameplay.expectedAction, null);
  assert.equal(queue.enabled, false);
  assert.deepEqual(events[0], {
    type: "action_mismatch",
    value: {
      expected: {actor: 0, type: "dahai", pai: "9s"},
      actual: {tile: "7s", riichi: false},
    },
  });
});

test("ranked queue rejects unsafe room state instead of clicking", () => {
  const queue = new Autoqueue(() => 0);
  queue.enabled = true;
  queue.updateAccount({
    nickname: "Account", copper: 100,
    rank: {id: 10101, score: 0},
  });
  const queueTime = Date.UTC(2026, 8, 22, 0, 0, 15);
  assert.equal(queue.next("connected", [], queueTime), null);
  assert.throws(() => queue.next("connected", [], queueTime + 15000),
    /No verified control/);
  assert.equal(queue.next("connected", [], queueTime,
    {screen: "ranked modes", room: "Bronze"}), null);
});

test("a dedicated friendly account cannot play until its room and identity agree", () => {
  const friendly = new FriendlySession({
    MAHJONG_SOUL_INSTANCE_ID: "friendly_1",
    MAHJONG_SOUL_FRIENDLY_ONLY: "on",
    MAHJONG_SOUL_EXPECTED_ACCOUNT_ID: "42",
    MAHJONG_SOUL_EXPECTED_NICKNAME: "Guest",
    MAHJONG_SOUL_PROFILE: "/tmp/profile",
    MAHJONG_SOUL_SCREENSHOT: "/tmp/screenshot",
    MAHJONG_SOUL_LIVE_STATE: "/tmp/state",
    MAHJONG_SOUL_CONTROL_URL: "http://127.0.0.1:8765",
    MAHJONG_SOUL_CONTROL_PIN: "pin",
    MAHJONG_SOUL_BUTTON_CAPTURES: "/tmp/buttons",
  });
  assert.equal(friendly.canPlay(true), false);
  friendly.account = {accountId: 42, nickname: "Guest"};
  assert.equal(friendly.canPlay(false), false);
  assert.equal(friendly.canPlay(true), true);
});
