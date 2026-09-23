import assert from "node:assert/strict";
import test from "node:test";
import { FriendlySession } from "./mahjongsoul_friendly.mjs";

function fixture() {
  const env = {
    MAHJONG_SOUL_INSTANCE_ID: "friendly-1", MAHJONG_SOUL_FRIENDLY_ONLY: "on",
    MAHJONG_SOUL_EXPECTED_ACCOUNT_ID: "42", MAHJONG_SOUL_EXPECTED_NICKNAME: "Guest",
  };
  for (const name of ["PROFILE", "SCREENSHOT", "LIVE_STATE", "CONTROL_URL",
    "CONTROL_PIN", "BUTTON_CAPTURES"]) env[`MAHJONG_SOUL_${name}`] = "isolated";
  const session = new FriendlySession(env);
  let status = {status: "hibernating"};
  const clicks = [];
  const io = {
    status: () => status,
    open: async () => {
      session.account = {accountId: 42, nickname: "Guest"};
      status = {status: "connected"};
    },
    wait: async () => {}, home: async () => {},
    join: async roomId => { status = {status: "in_room", roomId}; },
    room: async () => ({in_room: true}),
    controls: async () => ({targets: [{control: "ready", center: [750, 650]}]}),
    click: async (...center) => {clicks.push(center); session.ready = true;},
    stop: async () => {status = {status: "hibernating"};},
  };
  return {session, io, clicks};
}

test("dedicated guest wakes, authenticates, joins and readies; repeat stays idempotent", async () => {
  const {session, io, clicks} = fixture();
  assert.equal(session.canPlay(true), false);
  assert.deepEqual(await session.summon(12345, io), {roomId: 12345, ready: true});
  assert.deepEqual(await session.summon(12345, io), {roomId: 12345, ready: true});
  assert.deepEqual(clicks, [[750, 650]]);
  assert.equal(session.canPlay(true), true);
  assert.equal(session.canPlay(false), false);
  await assert.rejects(session.summon(99999, io), /another room/);
  assert.equal(new FriendlySession({}).canPlay(false), true);
  await assert.rejects(new FriendlySession({}).summon(12345, io), /dedicated/);
});

test("missing or mismatched identity and active games refuse every click", async () => {
  for (const setup of [
    session => {session.expectedNickname = null;},
    session => {session.expectedAccountId = 43;},
    session => {session.expectedNickname = "Wrong";},
  ]) {
    const {session, io, clicks} = fixture();
    setup(session);
    await assert.rejects(session.summon(12345, io), /expected account/);
    assert.deepEqual(clicks, []);
  }
  const {session, io, clicks} = fixture();
  await io.open();
  io.status = () => ({status: "playing", roomId: 12345});
  await assert.rejects(session.summon(12345, io), /already playing/);
  assert.deepEqual(clicks, []);
});

test("unknown Ready never clicks and game errors return the bot to login", async () => {
  const {session, io, clicks} = fixture();
  io.controls = async () => ({targets: []});
  assert.match((await session.summon(12345, io)).needs, /not visually verified/);
  assert.deepEqual(clicks, []);
  io.controls = async () => ({targets: [], error: {message: "Game error"}});
  await assert.rejects(session.summon(12345, io), /Game error/);
  assert.equal(io.status().status, "hibernating");
  assert.deepEqual(clicks, []);
});

test("room resets require a known roster containing only configured bots", () => {
  const {session} = fixture();
  session.botAccountIds = new Set([42, 43, 44]);
  assert.equal(session.roomHasNoHumans, false);
  session.updateRoomPlayers({player_list: [{account_id: 42}, {account_id: 99}], update_list: [], remove_list: []});
  assert.equal(session.roomHasNoHumans, false);
  session.updateRoomPlayers({player_list: [], update_list: [{account_id: 43}], remove_list: [99]});
  assert.equal(session.roomHasNoHumans, true);
  session.updateRoomPlayers({player_list: [], update_list: [{account_id: 100}], remove_list: []});
  assert.equal(session.roomHasNoHumans, false);
});

test("rematches ready while humans remain; departures disable ready and allow reset", async () => {
  const {session, io} = fixture();
  session.botAccountIds = new Set([42, 43, 44]);
  await io.open();
  assert.equal(session.roomAction({status: "in_room", roomId: 12345}), null);
  session.updateRoomPlayers({
    player_list: [{account_id: 42}, {account_id: 43}, {account_id: 99}],
    update_list: [], remove_list: [],
  });
  assert.equal(session.roomAction({status: "in_room", roomId: 12345}), "ready");
  await session.summon(12345, io);
  assert.equal(session.roomAction({status: "in_room", roomId: 12345}), null);
  // Game start consumes Ready; the next visible room can ready immediately.
  session.ready = false;
  assert.equal(session.roomAction({status: "in_room", roomId: 12345}), "ready");
  // A connectivity change isn't a departure from the room roster.
  session.updateRoomPlayers({
    player_list: [], update_list: [{account_id: 99}], remove_list: [],
  });
  assert.equal(session.roomHasNoHumans, false);
  session.updateRoomPlayers({
    player_list: [], update_list: [], remove_list: [99],
  });
  for (const status of ["in_room", "playing", "finished", "reconnecting"]) {
    assert.equal(session.roomAction({status, roomId: 12345}), "reset");
  }
  assert.equal(session.roomAction({status: "connected", roomId: null}), null);
  session.account = null;
  assert.equal(session.roomAction({status: "playing", roomId: 12345}), null);
});

test("friendly cancellation votes retry until confirmed and stop at expiry", async () => {
  const {session, io} = fixture();
  await io.open();
  assert.equal(session.shouldVoteYes(100000), false);
  session.cancelVote = {
    start_time: 100, duration_time: 60,
    results: [{account_id: 99, yes: true}],
  };
  assert.equal(session.shouldVoteYes(100000), true);
  session.voteClickedAt = 100000;
  assert.equal(session.shouldVoteYes(101500), false);
  assert.equal(session.shouldVoteYes(105000), true);
  session.cancelVote.results.push({account_id: 42, yes: true});
  assert.equal(session.shouldVoteYes(106000), false);
  session.cancelVote = {start_time: 200, duration_time: 60, results: []};
  assert.equal(session.shouldVoteYes(200000), true);
  assert.equal(session.shouldVoteYes(260000), false);
  session.friendlyOnly = false;
  assert.equal(session.shouldVoteYes(200000), false);
  session.friendlyOnly = true;
  session.account = null;
  assert.equal(session.shouldVoteYes(200000), false);
});
