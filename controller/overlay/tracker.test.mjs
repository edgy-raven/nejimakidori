import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import protobuf from "protobufjs";
import { GameTracker } from "./tracker.mjs";

const root = protobuf.Root.fromJSON(JSON.parse(fs.readFileSync(
  new URL("../mahjongsoul_liqi.json", import.meta.url), "utf8")));

function message(kind, name, type, value) {
  return {
    method: kind === 2 ? "Network.webSocketFrameSent"
      : "Network.webSocketFrameReceived",
    params: {requestId: "game", response: {opcode: 2,
      payloadData: Buffer.concat([
        Buffer.from([kind, 1, 0]),
        root.lookupType("Wrapper").encode({name,
          data: root.lookupType(type).encode(value).finish()}).finish(),
      ]).toString("base64")}},
  };
}

test("authenticate and restore a game, discard stale inference, then track the next game", async () => {
  let snapshot;
  const pending = [];
  const tracker = new GameTracker({
    publish: value => {snapshot = value;},
    predict: (body, signal) => new Promise(resolve => {
      pending.push({body, signal, resolve});
    }),
  });
  function authenticate(gameId) {
    tracker.handle(message(2, ".lq.FastTest.authGame", "ReqAuthGame",
      {account_id: 42, game_uuid: gameId}));
    tracker.handle(message(3, "", "ResAuthGame",
      {seat_list: [42, 10, 20, 30], players: [],
        game_config: {mode: {mode: 1}}}));
  }
  authenticate("first-game");
  assert.equal(snapshot.gameId, "first-game");
  tracker.handle(message(2, ".lq.FastTest.syncGame", "ReqSyncGame", {}));
  tracker.handle(message(3, "", "ResSyncGame", {game_restore: {actions: [{
    name: "ActionNewRound", step: 0,
    data: root.lookupType("ActionNewRound").encode({
      ju: 0, scores: [25000, 25000, 25000, 25000],
      tiles: ["1m", "2m", "3m", "4m", "5m", "6m", "7m",
        "8m", "9m", "1p", "2p", "3p", "4p", "4p"],
      left_tile_count: 69,
    }).finish(),
  }]}}));
  assert.equal(snapshot.status, "playing");
  assert.equal(snapshot.position.phase, "DISCARD");
  assert.equal(pending[0].body.actor, 0);
  assert.equal(pending[0].body.round.hands[0].length, 13);
  const frozen = JSON.stringify(pending[0].body);
  tracker.state.round.events.push({type: "test"});
  assert.equal(JSON.stringify(pending[0].body), frozen);
  authenticate("second-game");
  assert.equal(pending[0].signal.aborted, true);
  pending[0].resolve({decision: {action: {type: "dahai", pai: "4p"}}});
  await Promise.resolve();
  assert.equal(snapshot.gameId, "second-game");
  assert.equal(snapshot.prediction, null);
  tracker.update({phase: "DISCARD", round: {}, selfSeat: 0});
  const prediction = {decision: {action: {type: "dahai", pai: "4p"}}};
  pending[1].resolve(prediction);
  await Promise.resolve();
  assert.deepEqual(snapshot.prediction, prediction);
  tracker.handle({method: "Network.webSocketClosed",
    params: {requestId: "game"}});
  assert.equal(snapshot.status, "disconnected");
  assert.equal(snapshot.position, null);
});

test("current API failures are visible and invalid history clears recommendations", async () => {
  let snapshot;
  const tracker = new GameTracker({
    publish: value => {snapshot = value;},
    predict: async () => {throw new Error("Unauthorized");},
  });
  tracker.update({phase: "DISCARD", round: {}, selfSeat: 0});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(snapshot.status, "api_error");
  assert.equal(snapshot.error, "Unauthorized");
  tracker.handle({method: "Network.webSocketFrameReceived", params: {
    requestId: "game", response: {opcode: 2, payloadData: "Ag=="},
  }});
  assert.equal(snapshot.status, "tracking_error");
  assert.equal(snapshot.prediction, null);
  assert.equal(snapshot.position, null);
});
