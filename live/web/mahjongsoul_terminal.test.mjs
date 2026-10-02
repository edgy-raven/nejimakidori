import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { MahjongSoulLiveState } from "./mahjongsoul_live_state.mjs";
import { terminalEvent } from "./mahjongsoul_terminal.mjs";
import { playbackState, displayHand } from "../../review/public/shared/replay.js";

const start = {name: "ActionNewRound", step: 0, data: {
  ju: 0, chang: 0, ben: 0, liqibang: 1,
  scores: [25000, 24000, 25000, 25000], doras: ["1p"],
  tiles: Array(14).fill("1m"), left_tile_count: 69,
}};
const hand = ["1m", "2m", "3m", "1p", "2p", "3p", "4p", "6p",
  "7s", "8s", "9s", "5z", "5z"];

test("completed live hand reveals one tsumo tile, settles and survives until next hand", () => {
  const live = new MahjongSoulLiveState();
  live.apply(start);
  live.apply({name: "ActionDiscardTile", data: {seat: 0, tile: "1m", moqie: true}});
  live.apply({name: "ActionDealTile", data: {seat: 1, left_tile_count: 68}});
  const terminal = {name: "ActionHule", data: {
    hules: [{seat: 1, hand, hu_tile: "0p", zimo: true, count: 3, fu: 30, fans: []}],
    scores: [21000, 33000, 23000, 23000],
  }};
  const position = live.apply(terminal);
  const completed = playbackState(position.round);
  assert.equal(position.phase, null);
  assert.equal(completed.handOver, true);
  assert.deepEqual(completed.hands[1], hand);
  assert.equal(completed.drawn[1], "0p");
  assert.equal(displayHand(completed, 1).concealed.length, 13);
  assert.equal(displayHand(completed, 1).drawn, "0p");
  assert.equal(completed.revealed[1], true);
  assert.equal(completed.revealed[2], false);
  assert.equal(completed.riichiSticks, 0);
  assert.deepEqual(completed.deltas, [-4000, 9000, -2000, -2000]);
  assert.deepEqual(position.round.events.at(-1), terminalEvent("Hule", terminal.data));
  const last = live.lastHandResult;
  live.apply(start);
  assert.equal(live.lastHandResult, last);
  assert.equal(playbackState(live.round).handOver, false);
  assert.equal(playbackState(live.round).revealed.some(Boolean), false);
});

test("double ron identifies the same payer without inventing tsumo tiles", () => {
  const live = new MahjongSoulLiveState();
  live.apply(start);
  live.apply({name: "ActionDiscardTile", data: {seat: 0, tile: "1m", moqie: true}});
  live.apply({name: "ActionHule", data: {
    hules: [1, 2].map((seat) => ({seat, hand, hu_tile: "1m", zimo: false,
      count: 3, fu: 30, fans: []})),
    scores: [17000, 29000, 29000, 25000],
  }});
  const completed = playbackState(live.round);
  assert.deepEqual(completed.winners.map((winner) => winner.payer), [0, 0]);
  assert.equal(completed.rivers[0].at(-1).dealIn, true);
  assert.equal(completed.drawn[1], null);
  assert.equal(completed.drawn[2], null);
  for (const seat of [1, 2]) {
    assert.equal(displayHand(completed, seat).concealed.length, 13);
    assert.equal(displayHand(completed, seat).drawn, null);
  }
  assert.deepEqual(live.lastHandResult.deltas, [-8000, 5000, 4000, 0]);
});

test("draw settlements sum payments, reveal ready hands and retain sticks", () => {
  const live = new MahjongSoulLiveState();
  live.apply(start);
  live.apply({name: "ActionNoTile", data: {
    players: [0, 1, 2, 3].map((seat) => ({tingpai: seat === 1,
      hand: seat === 1 ? hand : []})),
    scores: [{old_scores: start.data.scores, delta_scores: [-1000, 3000, -1000, -1000]}],
  }});
  const completed = playbackState(live.round);
  assert.equal(completed.handOver, true);
  assert.deepEqual(completed.hands[1], hand);
  assert.deepEqual(completed.readySeats, [1]);
  assert.equal(completed.drawn[1], null);
  assert.equal(completed.riichiSticks, 1);
  assert.deepEqual(completed.scores, [24000, 27000, 24000, 24000]);
  assert.equal(terminalEvent("NoTile", {players: [], scores: [
    {old_scores: [25000, 25000, 25000, 25000], delta_scores: [12000, -4000, -4000, -4000]},
    {old_scores: [25000, 25000, 25000, 25000], delta_scores: [-4000, 12000, -4000, -4000]},
  ], liujumanguan: true}).scores[0], 33000);
  live.apply(start);
  live.apply({name: "ActionLiuJu", data: {type: 4,
    liqi: {seat: 0, score: 24000, liqibang: 2}}});
  assert.equal(live.lastHandResult.outcome, "Four riichi");
  assert.deepEqual(live.lastHandResult.deltas, [-1000, 0, 0, 0]);
  assert.equal(playbackState(live.round).riichiSticks, 2);
});

test("captured live four-riichi abort settles the missing acceptance before the next hand", () => {
  const record = JSON.parse(fs.readFileSync(new URL(
    "./fixtures/suucha-riichi-live.json", import.meta.url)));
  const live = new MahjongSoulLiveState();
  live.selfSeat = record.selfSeat;
  for (const action of record.actions.slice(0, -1)) live.apply(action);
  assert.equal(record.actions.at(-2).data.liqi, null);
  assert.equal(live.lastHandResult.outcome, "Four riichi");
  assert.deepEqual(live.lastHandResult.deltas, [-1000, -1000, -1000, -1000]);
  assert.deepEqual(live.lastHandResult.scores, [16500, 22500, 21500, 21500]);
  assert.equal(playbackState(live.round).riichiSticks, 18);
  live.apply(record.actions.at(-1));
  assert.deepEqual(live.lastHandResult.scores, live.round.scores);
  assert.equal(live.round.riichiSticks, 18);
  assert.equal(live.round.honba, 6);
});


test("draw separation retains every tile through calls and an open-hand ron", () => {
  const live = new MahjongSoulLiveState();
  live.apply(start);
  let board = playbackState(live.round);
  assert.equal(displayHand(board, 0).concealed.length, 13);
  assert.equal(displayHand(board, 0).drawn, "1m");
  live.apply({name: "ActionDiscardTile", data: {seat: 0, tile: "1m", moqie: true}});
  live.apply({name: "ActionChiPengGang", data: {seat: 1, type: 1,
    tiles: ["1m", "1m", "1m"], froms: [1, 1, 0]}});
  live.apply({name: "ActionDiscardTile", data: {seat: 1, tile: "7s", moqie: false}});
  live.apply({name: "ActionDealTile", data: {seat: 2, left_tile_count: 67}});
  board = playbackState(live.round);
  assert.equal(displayHand(board, 2).concealed.length, 13);
  assert.equal(displayHand(board, 2).drawn, "?");
  live.apply({name: "ActionDiscardTile", data: {seat: 2, tile: "0p", moqie: true}});
  live.apply({name: "ActionHule", data: {
    hules: [{seat: 1, hand: hand.slice(3), hu_tile: "0p", zimo: false,
      count: 3, fu: 30, fans: []}], scores: [25000, 33000, 17000, 25000],
  }});
  board = playbackState(live.round);
  assert.equal(displayHand(board, 1).concealed.length, 10);
  assert.equal(displayHand(board, 1).drawn, null);
  assert.equal(board.melds[1][0].tiles.length, 3);
  assert.equal(board.rivers[2].at(-1).dealIn, true);
});
