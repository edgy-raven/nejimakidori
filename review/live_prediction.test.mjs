import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import fs from 'node:fs/promises';
import http from 'node:http';
import test from 'node:test';
import {BrowserTransport} from '../live/web/browser_transport.mjs';
import {publicFiles} from './public_files.mjs';
import {livePredictionBody} from '../live/web/live_prediction.mjs';
import {MahjongSoulLiveState} from '../live/web/mahjongsoul_live_state.mjs';

test('live riichi survives forced draws, long rivers and the next hand', async t => {
  const model = {
    riichi_policy_probabilities: Array(76).fill(0),
    riichi_action_probabilities: [.2, .8],
    discard_policy_probabilities: Array(38).fill(0),
    opponent_shanten_probabilities: Array(3).fill([.1, .5, .3, .1]),
    final_placement_probabilities: [.4, .3, .2, .1],
  };
  model.riichi_policy_probabilities[31] = .2;
  model.riichi_policy_probabilities[75] = .8;
  model.discard_policy_probabilities[37] = 1;
  const decision = {actor: 0, phase: 'RIICHI', action_kind: 'recommended',
    action: {actor: 0, type: 'reach'},
    riichi_discard: {actor: 0, type: 'dahai', pai: 'P', tsumogiri: true},
    discard_options: [31, 37].map(action => ({action, tile: '5z', tsumogiri: action === 37,
      all_shanten: 0, all_ukeire_count: 4, all_upgrade_count: 0}))};
  const body = livePredictionBody({decision, inference: {latency_ms: 12,
    metadata: {payment_values: [0, 1000]}, predictions: [model]}});
  assert.deepEqual(body.model, model);
  const live = new MahjongSoulLiveState();
  const opening = {name: 'ActionNewRound', step: 0, data: {
    ju: 0, chang: 0, ben: 0, liqibang: 0, scores: Array(4).fill(25000), doras: ['1p'],
    tiles: ['1m','2m','3m','4m','5m','6m','1p','2p','3p','7s','8s','9s','5z','5z'],
    left_tile_count: 69,
  }};
  const fixture = {status: 'playing', updatedAt: new Date().toISOString(),
    autoplay: {enabled: false, ready: false}, autoqueue: {enabled: true, rank: {id: 10301, score: 300}, copper: 6999,
      room: {name: "Gold", length: "South"},
      blockedReason: "Insufficient copper: Gold South needs 7,000."},
    position: live.apply(opening), prediction: {status: 200, body}};
  const files = publicFiles(new URL('./public/', import.meta.url).pathname);
  for (const name of ['tiles.png', 'tiles_rects.json']) {
    files.set(`/assets/${name}`, [new URL(`../vision/static/${name}`, import.meta.url).pathname,
      name.endsWith('png') ? 'image/png' : 'application/json']);
  }
  const server = http.createServer(async (request, response) => {
    const file = files.get(new URL(request.url, 'http://localhost').pathname);
    if (!file) return response.writeHead(404).end();
    response.writeHead(200, {'Content-Type': file[1]});
    response.end(await fs.readFile(file[0]));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => server.close());
  const profile = await fs.mkdtemp('/tmp/live-riichi-browser-');
  const chrome = spawn('/usr/bin/google-chrome', ['--headless=new', '--no-sandbox',
    '--disable-dev-shm-usage', '--password-store=basic', '--remote-debugging-pipe',
    `--user-data-dir=${profile}`, 'about:blank'],
  {stdio: ['ignore', 'ignore', 'ignore', 'pipe', 'pipe']});
  const errors = [];
  const browser = new BrowserTransport(message => {
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params);
  });
  browser.attach(chrome);
  t.after(async () => {
    browser.close();
    const exited = once(chrome, 'exit');
    chrome.kill('SIGTERM');
    await exited;
    await fs.rm(profile, {recursive: true, force: true, maxRetries: 5});
  });
  const {targetId} = await browser.send('Target.createTarget', {url: 'about:blank'});
  const {sessionId} = await browser.send('Target.attachToTarget', {targetId, flatten: true});
  const send = (method, params = {}) => browser.send(method, params, sessionId);
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    assert.equal(result.exceptionDetails, undefined, JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Page.addScriptToEvaluateOnNewDocument', {source: `
    window.fixture = ${JSON.stringify(fixture)};
    const realFetch = fetch;
    window.fetch = (url, options) => String(url).includes('/api/')
      ? Promise.resolve(new Response(JSON.stringify(window.fixture))) : realFetch(url, options);
  `});
  await send('Page.navigate', {url: `http://127.0.0.1:${server.address().port}/live/`});
  await evaluate(`new Promise(resolve => {
    const timer = setInterval(() => {
      if (document.querySelector('#live-actions')?.textContent) {clearInterval(timer); resolve();}
    }, 20);
    setTimeout(() => {clearInterval(timer); resolve();}, 3000);
  })`);
  assert.match(await evaluate("document.querySelector('#live-ranked-status').textContent"),
    /Rank E1:300.*6,999 copper.*Gold South.*Insufficient copper/);
  // Check actual page rendering; its polling catch otherwise hides JS errors.
  assert.match(await evaluate("document.querySelector('#live-actions').textContent"), /Reach80\.0%/,
    await evaluate("document.querySelector('#live-status').textContent"));
  const recommended = await evaluate("document.querySelector('#live-recommendation').textContent");
  assert.match(recommended, /Reach.*80\.0%.*Discard/);
  await evaluate(`(async () => {
    window.view = new (await import('/shared/prediction.js')).PredictionView();
    window.view.render(window.fixture);
  })()`);
  const forced = livePredictionBody({decision: {...decision, phase: 'DISCARD',
    action_kind: 'forced', action: decision.riichi_discard}, inference: null});
  assert.equal(forced.model, null);
  await evaluate(`window.fixture.position.hand.riichi = true;
    window.fixture.prediction = ${JSON.stringify({status: 200, body: forced})};
    window.view.render(window.fixture);
    window.fixture.prediction = null;
    window.view.render(window.fixture);`);
  assert.equal(await evaluate("document.querySelector('#live-recommendation').textContent"), recommended);
  const rivers = await evaluate(`(async () => {
    const {renderRiver} = await import('/shared/board.js');
    const state = {rivers: Array.from({length: 4}, () =>
      Array.from({length: 24}, (_, index) => ({tile: '1m', riichi: index === 6}))) };
    for (let seat = 0; seat < 4; seat++) renderRiver(seat, 0, state);
    return [...document.querySelectorAll('[data-live-river]')].map(river => ({
      rows: [...river.children].map(row => row.children.length),
      sideways: river.querySelectorAll('.river-riichi-slot').length,
    }));
  })()`);
  assert.deepEqual(rivers, Array(4).fill({rows: [6, 6, 12], sideways: 1}));
  await evaluate(`window.fixture = structuredClone(window.fixture);
    window.fixture.position.round.kyoku++;
    window.view.render(window.fixture);`);
  assert.equal(await evaluate("document.querySelector('#live-recommendation').textContent"), 'Waiting for the model.');
  await evaluate(`window.fixture.automationBlock = {reason: 'Unexpected dialog'};
    window.fixture.updatedAt = new Date(Date.now() + 1).toISOString();
    new Promise(resolve => setTimeout(resolve, 400));`);
  assert.equal(await evaluate("document.querySelector('#live-autoqueue').textContent"),
    'Resume automation');
  assert.match(await evaluate("document.querySelector('#live-status').textContent"),
    /Paused: Unexpected dialog/);
  assert.equal(await evaluate("document.querySelector('#live-browser').hidden"), false);
  assert.deepEqual(errors, []);
});
