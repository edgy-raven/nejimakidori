import assert from "node:assert/strict";
import test from "node:test";
import {discardTarget, offeredButtons} from "./mahjongsoul_targeting.mjs";

test("riichi screen retains the simultaneous kan offer from the game", () => {
  assert.deepEqual(offeredButtons({operation: {operation_list: [
    {type: 1}, {type: 4, combination: ["1p|1p|1p|1p"]},
    {type: 7, combination: ["5m", "5m"]},
  ]}}), ["discard", "kan", "reach", "skip"]);
});

test("a separated draw cannot be reassigned by a better visual layout score", () => {
  const position = {hand: {concealed: ["9s", "1m"], drawn: "7s"}};
  const control = {hand: {tiles: [
    {box: [100, 620, 60, 90], drawn: false, scores: {"1m": .99, "9s": .7}},
    {box: [160, 620, 60, 90], drawn: false, scores: {"9s": .80, "7s": .90}},
    {box: [240, 620, 60, 90], drawn: true, scores: {"9s": .90, "7s": .80}},
  ]}};
  // The sorted-all interpretation wins on image scores but contradicts the gap.
  assert.equal(discardTarget({pai: "9s"}, position, control).x, 190);
  assert.equal(discardTarget({pai: "7s"}, position, control).x, 270);
  // Without a separated draw, the sorted-all visual arrangement is valid.
  control.hand.tiles[2].drawn = false;
  assert.equal(discardTarget({pai: "9s"}, position, control).x, 270);
});

test("opening dealer hand uses recognized identities instead of assuming a separate draw", () => {
  const position = {action: "ActionNewRound", hand: {concealed: ["9s", "1m"], drawn: "3s"}};
  const control = {hand: {tiles: [
    {box: [100, 620, 60, 90], label: "1m", trusted: true, scores: {"1m": .99}},
    {box: [160, 620, 60, 90], label: "3s", trusted: true, scores: {"3s": .85, "9s": .88}},
    {box: [240, 620, 60, 90], label: "9s", trusted: true, drawn: true, scores: {"3s": .80, "9s": .89}},
  ]}};
  assert.equal(discardTarget({pai: "3s"}, position, control).x, 190);
  control.hand.tiles[1].label = "7s";
  assert.throws(() => discardTarget({pai: "3s"}, position, control), /no layout agrees/);
});

test("dora follows newly revealed indicators and glare cannot veto its whole hand", async () => {
  const {doraTiles} = await import('./mahjongsoul_targeting.mjs');
  assert.deepEqual(doraTiles({round: {doraIndicators: ['9m', '4z', '7z'], events: []}}),
    ['1m', '1z', '5z']);
  const position = {hand: {concealed: ['9p', '5s'], drawn: null},
    round: {doraIndicators: ['7s'], events: [{doras: ['7s', '4s']}]}};
  const control = {hand: {tiles: [
    {box: [100, 614, 62, 101], label: '9p', trusted: true, scores: {'9p': .9}},
    {box: [163, 614, 62, 101], label: 'white', trusted: true, glare: true,
      scores: {white: .85, '5s': .78}},
  ]}};
  assert.deepEqual(doraTiles(position), ['8s', '5s']);
  assert.equal(discardTarget({pai: '9p'}, position, control).x, 131);
  assert.equal(discardTarget({pai: '5s'}, position, control).x, 194);
  control.hand.tiles[1].glare = false;
  assert.throws(() => discardTarget({pai: '9p'}, position, control), /no layout/);
});

test("tsumogiri and same-tile tedashi target their own physical copies", () => {
  const position = {hand: {concealed: ["5m", "1p"], drawn: "5m"}};
  const control = {hand: {tiles: [
    {box: [100, 620, 60, 90], drawn: false, scores: {"5m": .99}},
    {box: [160, 620, 60, 90], drawn: false, scores: {"1p": .99}},
    {box: [240, 620, 60, 90], drawn: true, scores: {"5m": .99}},
  ]}};
  assert.equal(discardTarget({type: "dahai", pai: "5m", tsumogiri: true},
    position, control).x, 270);
  assert.equal(discardTarget({type: "dahai", pai: "5m", tsumogiri: false},
    position, control).x, 130);
  position.hand.drawn = "0m";
  assert.equal(discardTarget({type: "dahai", pai: "5mr", tsumogiri: true},
    position, control).x, 270);
  assert.equal(discardTarget({type: "dahai", pai: "5m", tsumogiri: false},
    position, control).x, 130);
  assert.throws(() => discardTarget(
    {type: "dahai", pai: "5mr", tsumogiri: false}, position, control),
  /requested copy/);
});
