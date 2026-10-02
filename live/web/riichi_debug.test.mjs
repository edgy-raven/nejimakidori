import assert from "node:assert/strict";
import test from "node:test";
import {riichiOnlyDecision} from "./riichi_debug.mjs";

test("debug players force legal riichi, discard normally and decline calls and wins", () => {
  const discard = {actor: 0, type: "dahai", pai: "5m", tsumogiri: true};
  const position = {selfSeat: 0, hand: {drawn: "5m"},
    operation: {operation_list: [{type: 7, combination: ["5m"]}]}};
  const body = {action: discard, possible_actions: [discard], riichi_discard: null};
  const result = riichiOnlyDecision(position, body);
  assert.equal(result.action.type, "reach");
  assert.deepEqual(result.riichi_discard, discard);
  body.riichi_discard = {...discard, pai: "9p"};
  assert.deepEqual(riichiOnlyDecision(position, body).riichi_discard, discard);
  position.operation.operation_list = [{type: 7, combination: ["0p"]}];
  assert.equal(riichiOnlyDecision(position, body).riichi_discard.pai, "0p");
  position.operation.operation_list = [];
  assert.deepEqual(riichiOnlyDecision(position, body).action, discard);
  for (const type of ["chi", "pon", "ankan", "kakan", "daiminkan", "hora", "ryukyoku"]) {
    const offered = {...body, action: {actor: 0, type}};
    assert.equal(riichiOnlyDecision(position, offered).action.type, "dahai");
    offered.possible_actions = [{actor: 0, type}, {actor: 0, type: "none"}];
    assert.equal(riichiOnlyDecision(position, offered).action.type, "none");
  }
  position.hand.riichi = true;
  for (const type of [4, 8, 9]) {
    position.operation.operation_list = [{type}];
    assert.equal(riichiOnlyDecision(position, body).action.type, "none");
  }
});

test("only debug mode permits an all-bot friendly room", async () => {
  const {FriendlySession} = await import("./mahjongsoul_friendly.mjs");
  const session = new FriendlySession({MAHJONG_SOUL_EXPECTED_ACCOUNT_ID: "1",
    MAHJONG_SOUL_EXPECTED_NICKNAME: "debug", MAHJONG_SOUL_BOT_ACCOUNT_IDS: "1,2,3,4",
    MAHJONG_SOUL_RIICHI_ONLY: "on"});
  session.friendlyOnly = true;
  session.account = {accountId: 1, nickname: "debug"};
  session.roomPlayers = [1,2,3,4];
  assert.equal(session.roomAction({roomId: 12345, status: "in_room"}), "ready");
  session.riichiOnly = false;
  assert.equal(session.roomAction({roomId: 12345, status: "in_room"}), "reset");
});
