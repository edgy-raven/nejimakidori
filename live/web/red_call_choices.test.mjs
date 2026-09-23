import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {callComposition, createGameplay} from './mahjongsoul_gameplay.mjs';

test('red-call selection retains all server choices and follows their exact order', () => {
  for (const [type, operationType, red, ordinary] of [
    ['chi', 2, ['5mr', '6m'], ['5m', '6m']],
    ['pon', 3, ['5pr', '5p'], ['5p', '5p']],
  ]) {
    const action = {type, consumed:red};
    const soul = tiles => tiles.map(tile => tile.replace('5mr', '0m').replace('5pr', '0p')).join('|');
    for (const combinations of [[soul(ordinary), soul(red)], [soul(red), soul(ordinary)]]) {
      const operation = {operation_list:[{type:operationType, combination:combinations}]};
      const index = combinations.indexOf(soul(red));
      const target = callComposition(action, operation);
      assert.equal(target.x, index === 0 ? 561 : 694);
      assert.equal(target.y, 510);
    }
    const single = {operation_list:[{type:operationType, combination:[soul(red)]}]};
    assert.equal(callComposition(action, single), null);
    const missing = {operation_list:[{type:operationType, combination:[soul(ordinary)]}]};
    assert.throws(() => callComposition(action, missing), /not offered/);
  }
});


test('gameplay clicks the red option once and pauses before any click if it disappears', async t => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'red-call-chooser-'));
  t.after(() => fs.rm(directory, {recursive:true, force:true}));
  const action = {actor:1, type:'pon', pai:'5p', consumed:['5pr', '5p']};
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({
    decision:{action, possible_actions:[action], difficulty:0},
  }), {status:200}));
  const clicks = [];
  const ctx = {
    autoplay:true, casual:true, positionVersion:1, liveState:{status:'playing'},
    modelUrl:'http://model.test', buttonCapturePath:directory,
    canPlay:() => true, send:async () => {}, wait:async () => {},
    click:async (x, y) => {clicks.push([x, y]);}, output:() => {},
    publishLiveState:() => {},
    pauseAutomation:reason => {ctx.automationBlock = reason;},
    captureControl:async () => ({
      control_ready:true, screen:'table', hand:{meld_count:0, decision:false},
      actions:{pon:{center:[627, 567], box:[590, 530, 80, 65]}},
      image:await fs.readFile(new URL('../../vision/static/action_buttons/button_anchor.png', import.meta.url)),
    }),
  };
  const game = createGameplay(ctx);
  const position = {phase:'RESPONSE', selfSeat:1, hand:{meldCount:0, riichi:false},
    receivedAt:Date.now(), round:{},
    operation:{operation_list:[{type:3, combination:['5p|5p', '0p|5p']}]}};
  await game.prediction(position, 1);
  assert.equal(clicks.length, 2);
  assert.deepEqual(clicks[1], [694, 510]);
  assert.equal(ctx.automationBlock, undefined);
  clicks.length = 0;
  position.operation.operation_list[0].combination = ['5p|5p'];
  await game.prediction(position, 1);
  assert.match(ctx.automationBlock, /not offered/);
  assert.deepEqual(clicks, []);
  await game.prediction(position, 1);
  assert.deepEqual(clicks, []);
});
