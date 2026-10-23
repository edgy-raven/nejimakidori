import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import protobuf from "protobufjs";

import { decodeGameResponse, decodeNotification } from "./mahjongsoul_protocol.mjs";
import { MahjongSoulLiveState } from "./mahjongsoul_live_state.mjs";

const root = protobuf.Root.fromJSON(JSON.parse(readFileSync(
  new URL("./mahjongsoul_liqi.json", import.meta.url), "utf8")));

function frame(kind, name, type, value) {
  return Buffer.concat([
    Buffer.from([kind, 1, 0]),
    root.lookupType("Wrapper").encode({
      name, data: root.lookupType(type).encode(value).finish(),
    }).finish(),
  ]);
}

test("room joins and reconnects correlate replies and restore private rounds", () => {
  const requests = new Map();
  for (const [method, requestType, responseType, response] of [
    ["Lobby.joinRoom", "ReqJoinRoom", "ResJoinRoom", { room: { room_id: 59794 } }],
    ["FastTest.authGame", "ReqAuthGame", "ResAuthGame", {
      seat_list: [10, 0, 42, 30],
      players: [
        {account_id: 42, nickname: "Nejimakidori", level: {id: 10403, score: 2300}, level3: {id: 20101, score: 0}},
        {account_id: 30, level: {id: 10501, score: 3500}},
        {account_id: 10, level: {id: 10202, score: 500}},
      ],
      game_config: { meta: { room_id: 59794 }, mode: {mode: 1} },
    }],
  ]) {
    assert.equal(decodeGameResponse(
      frame(2, `.lq.${method}`, requestType, {account_id: 42, game_uuid: "completed-game-id"}), requests, "socket"), null);
    assert.equal(decodeGameResponse(
      frame(3, "", responseType, response), requests, "other"), null);
    const decoded = decodeGameResponse(
      frame(3, "", responseType, response), requests, "socket");
    assert.equal(decoded.roomId, 59794);
    if (method === "FastTest.authGame") {
      assert.equal(decoded.gameId, "completed-game-id");
      assert.equal(decoded.eastOnly, true);
      assert.equal(decoded.selfSeat, 2);
      assert.deepEqual(decoded.accountRanks, [
        {id: 10202, score: 500}, null,
        {id: 10403, score: 2300}, {id: 10501, score: 3500},
      ]);
      assert.equal(decoded.players[2].nickname, "Nejimakidori");
      assert.equal(decoded.players[2].seat, 2);
      const live = new MahjongSoulLiveState();
      live.players = decoded.players;
      live.accountRanks = decoded.accountRanks;
      for (const ju of [0, 1]) {
        const position = live.apply({name: "ActionNewRound", data: {
          ju, scores: [25000, 25000, 25000, 25000], doras: [],
          tiles: Array(14).fill("1m"), left_tile_count: 69,
        }});
        assert.deepEqual(position.accountRanks, decoded.accountRanks);
        assert.deepEqual(position.players, decoded.players);
      }
    }
  }
  decodeGameResponse(frame(2, ".lq.FastTest.syncGame", "ReqSyncGame", {}), requests, "socket");
  const result = decodeGameResponse(frame(3, "", "ResSyncGame", {
    game_restore: { actions: [{
      name: "ActionNewRound", step: 0,
      data: root.lookupType("ActionNewRound").encode({
        ju: 0, scores: [25000, 25000, 25000, 25000],
        tiles: ["1m", "2m", "3m", "4m", "5m", "6m", "2p",
          "3p", "4p", "6s", "7s", "8s", "1z", "2z"],
        doras: ["1p"], left_tile_count: 69,
      }).finish(),
    }] },
  }), requests, "socket");
  const state = new MahjongSoulLiveState();
  state.eastOnly = true;
  const position = state.apply(result.actions[0]);
  assert.equal(position.round.eastOnly, true);
  assert.equal(position.round.prevailingWind, 0);
  assert.equal(position.selfSeat, 0);
  assert.equal(position.round.hands[0].length, 13);
  assert.equal(position.hand.drawn, "2z");
  assert.equal(requests.size, 0);
});

test("hand results retain settlement scores through the next round", () => {
  const state = new MahjongSoulLiveState();
  const round = {name: "ActionNewRound", step: 0, data: {
    ju: 0, chang: 0, ben: 1, liqibang: 0,
    scores: [25000, 25000, 25000, 25000], doras: ["1p"],
    tiles: Array(14).fill("1m"), left_tile_count: 69,
  }};
  state.apply(round);
  state.apply({name: "ActionHule", data: {
    scores: [33000, 17000, 25000, 25000],
    hules: [{seat: 0, hu_tile: "1m", zimo: false, count: 4, fu: 40, fans: [{id: 2, val: 1}, {id: 31, val: 3}]}],
  }});
  assert.deepEqual(state.lastHandResult.deltas, [8000, -8000, 0, 0]);
  assert.equal(state.lastHandResult.winners[0].tile, "1m");
  assert.deepEqual(state.lastHandResult.winners[0].yaku, [
    {name: "Riichi", value: 1, yakuman: false},
    {name: "Dora", value: 3, yakuman: false},
  ]);
  const result = state.lastHandResult;
  state.apply(round);
  assert.equal(state.lastHandResult, result);
  state.apply({name: "ActionNoTile", data: {players: [
    {tingpai: true, hand: ["0p", "7s", "8s", "9s"]},
    {tingpai: false, hand: []}, {tingpai: false, hand: []},
    {tingpai: false, hand: []},
  ], scores: [{
    old_scores: [25000, 25000, 25000, 25000],
    delta_scores: [3000, -1000, -1000, -1000],
  }]}});
  assert.equal(state.lastHandResult.outcome, "Exhaustive draw");
  assert.deepEqual(state.lastHandResult.tenpaiHands,
    [{seat: 0, tiles: ["0p", "7s", "8s", "9s"]}]);
  const drawResult = state.lastHandResult;
  state.apply(round);
  assert.equal(state.lastHandResult, drawResult);
  assert.deepEqual(state.lastHandResult.scores, [28000, 24000, 24000, 24000]);
  state.apply(round);
  state.apply({name: "ActionLiuJu", data: {}});
  assert.equal(state.lastHandResult.outcome, "Abortive draw");
  assert.deepEqual(state.lastHandResult.deltas, [0, 0, 0, 0]);
});

test("session replies track logout and login without exposing credentials", () => {
  const requests = new Map();
  for (const [method, requestType, responseType, errorCode] of [
    ["logout", "ReqLogout", "ResLogout", 0],
    ["oauth2Login", "ReqOauth2Login", "ResLogin", 1001],
    ["oauth2Login", "ReqOauth2Login", "ResLogin", 0],
    ["login", "ReqLogin", "ResLogin", 0],
    ["emailLogin", "ReqEmailLogin", "ResLogin", 0],
  ]) {
    decodeGameResponse(frame(2, `.lq.Lobby.${method}`, requestType, {}), requests, "socket");
    assert.deepEqual(decodeGameResponse(frame(3, "", responseType, {
      error: { code: errorCode }, access_token: "must-not-be-returned",
      account_id: 42, account: {nickname: "FriendlyOne", gold: 7000,
        level: {id: 10301, score: 300}},
    }), requests, "socket"), { method, errorCode,
      ...(method !== "logout" && !errorCode
        ? {account: {accountId: 42, nickname: "FriendlyOne", copper: 7000,
          rank: {id: 10301, score: 300}}} : {}),
    });
  }
  assert.equal(requests.size, 0);
});

test("final rankings use awarded match points and retain the player viewpoint", () => {
  const state = new MahjongSoulLiveState();
  state.selfSeat = 2;
  state.apply({name: "NotifyGameEndResult", data: {result: {players: [
    {seat: 0, part_point_1: 25000, total_point: -5000},
    {seat: 1, part_point_1: 18000, total_point: -22000},
    {seat: 2, part_point_1: 32000, total_point: 22000},
    {seat: 3, part_point_1: 25000, total_point: 5000},
  ]}}});
  assert.deepEqual(state.finalMatchResult, {selfSeat: 2, players: [
    {seat: 2, rank: 1, score: 32000},
    {seat: 3, rank: 2, score: 25000},
    {seat: 0, rank: 3, score: 25000},
    {seat: 1, rank: 4, score: 18000},
  ]});
});

test("authenticated seat shows opening discards before our first draw", () => {
  const state = new MahjongSoulLiveState();
  state.selfSeat = 3;
  for (const dealer of [0, 1]) {
    const position = state.apply({name: "ActionNewRound", step: 0, data: {
      ju: dealer, chang: 0, ben: 0, liqibang: 0,
      scores: [25000, 25000, 25000, 25000], doras: ["1p"],
      tiles: Array(13).fill("1m"), left_tile_count: 69,
    }});
    assert.equal(position.selfSeat, 3);
    assert.deepEqual(position.round.hands[3], Array(13).fill("1m"));
    const discarded = state.apply({name: "ActionDiscardTile", step: 1,
      data: {seat: dealer, tile: "9p", moqie: false}});
    assert.equal(discarded.selfSeat, 3);
    assert.equal(discarded.round.events.at(-1).tile, "9p");
    assert.equal(discarded.round.events.at(-1).seat, dealer);
  }
});

test("a concealed quad retains an unrelated draw before the replacement draw", () => {
  const state = new MahjongSoulLiveState();
  state.apply({name: "ActionNewRound", step: 0, data: {
    ju: 0, chang: 0, ben: 0, liqibang: 0,
    scores: [25000, 25000, 25000, 25000], doras: ["1p"],
    tiles: ["4p", "4p", "4p", "4p", "1m", "2m", "3m",
      "5m", "6m", "7m", "1s", "2s", "3s", "4m"], left_tile_count: 69,
  }});
  state.apply({name: "ActionAnGangAddGang", step: 1,
    data: {seat: 0, type: 3, tiles: "4p"}});
  assert.equal(state.concealedTiles.length, 10);
  assert.ok(state.concealedTiles.includes("4m"));
  assert.equal(state.drawnTile, null);
  state.apply({name: "ActionDealTile", step: 2,
    data: {seat: 0, tile: "1z", left_tile_count: 68}});
  const position = state.apply({name: "ActionDiscardTile", step: 3,
    data: {seat: 0, tile: "4m", moqie: false}});
  assert.equal(position.hand.concealed.length, 10);
  assert.ok(position.hand.concealed.includes("1z"));
  assert.ok(!position.hand.concealed.includes("4m"));
});


test("Ready replies confirm the requested state without issuing protocol actions", () => {
  const requests = new Map();
  for (const ready of [true, false]) {
    decodeGameResponse(frame(2, ".lq.Lobby.readyPlay", "ReqRoomReady", {ready}), requests, "socket");
    assert.deepEqual(decodeGameResponse(frame(3, "", "ResCommon", {}), requests, "socket"),
      {method: "readyPlay", ready, errorCode: 0});
  }
});


test("leaving a room reports success or rejection to the controller", () => {
  const requests = new Map();
  for (const errorCode of [0, 1100]) {
    decodeGameResponse(frame(2, ".lq.Lobby.leaveRoom", "ReqCommon", {}), requests, "socket");
    assert.deepEqual(decodeGameResponse(
      frame(3, "", "ResCommon", {error: {code: errorCode}}), requests, "socket"),
      {method: "leaveRoom", errorCode});
  }
});

test("account refresh and matchmaking failures expose only queue-relevant data", () => {
  const requests = new Map();
  decodeGameResponse(frame(2, ".lq.Lobby.fetchAccountInfo", "ReqAccountInfo", {}), requests, "lobby");
  assert.deepEqual(decodeGameResponse(frame(3, "", "ResAccountInfo", {
    account: {account_id: 42, nickname: "Account", gold: 2499,
      diamond: 20000, level: {id: 10201, score: 400},
      level3: {id: 20501, score: 0}},
  }), requests, "lobby"), {method: "fetchAccountInfo", errorCode: 0,
    account: {accountId: 42, nickname: "Account", copper: 2499,
      rank: {id: 10201, score: 400}}});
  for (const code of [0, 1304]) {
    decodeGameResponse(frame(2, ".lq.Lobby.matchGame", "ReqJoinMatchQueue", {
      match_mode: 5,
    }), requests, "lobby");
    assert.deepEqual(decodeGameResponse(frame(3, "", "ResCommon", {
      error: {code},
    }), requests, "lobby"), {method: "matchGame", errorCode: code});
  }
});

test("current game-finish rewards decode rank, chest and character updates", () => {
  const reward = root.lookupType("NotifyGameFinishRewardV2");
  const frame = Buffer.concat([Buffer.from([1]), root.lookupType("Wrapper").encode({
    name: ".lq.NotifyGameFinishRewardV2",
    data: reward.encode({mode_id: 2,
      level_change: {origin: {id: 10101, score: 0}, final: {id: 10101, score: 30}},
      match_chest: {chest_id: 1, origin: 0, final: 30, is_graded: false, rewards: []},
      main_character: {level: 1, exp: 30, add: 30},
    }).finish(),
  }).finish()]);
  const decoded = decodeNotification(frame);
  assert.equal(decoded.name, "NotifyGameFinishRewardV2");
  assert.equal(decoded.data.mode_id, 2);
  assert.equal(decoded.data.level_change.final.score, 30);
  assert.equal(decoded.data.match_chest.final, 30);
  assert.equal(decoded.data.main_character.add, 30);
  assert.deepEqual(decoded.data.badges, []);
});
