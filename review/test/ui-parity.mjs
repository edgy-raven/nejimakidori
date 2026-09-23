import assert from "node:assert/strict";
import fs from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { fileURLToPath } from "node:url";
import { setTimeout as wait } from "node:timers/promises";
import { publicFiles } from "../public_files.mjs";
import { BrowserTransport } from "../../live/web/browser_transport.mjs";
import { MahjongSoulLiveState } from "../../live/web/mahjongsoul_live_state.mjs";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const baseline = path.resolve(process.argv[2]);
const artifactPath = path.resolve(web, "../artifacts/ui-parity");
await fs.mkdir(artifactPath, {recursive: true});
const profile = await fs.mkdtemp("/tmp/nejimakidori-ui-parity-");
const errors = [];
const browser = new BrowserTransport((message) => {
  if (message.method === "Runtime.exceptionThrown") errors.push(message.params);
});
const servers = [];
const live = new MahjongSoulLiveState();
const start = {name: "ActionNewRound", step: 0, data: {
  ju: 0, chang: 0, ben: 0, liqibang: 1,
  scores: [25000, 24000, 25000, 25000], doras: ["0p"],
  tiles: ["1m", "2m", "3m", "4m", "5mr", "6m", "1p", "2p", "3p", "7s", "8s", "9s", "5z", "5z"],
  left_tile_count: 69,
}};
const body = {
  actor: 0, phase: "DISCARD", action: {type: "dahai", pai: "5z"},
  action_kind: "recommended", model_latency_ms: 12.3, difficulty: 0.4,
  discard_options: [{tile: "5z", action: 31, all_shanten: 1, all_ukeire_count: 16, all_upgrade_count: 8}],
  terminal_payment_values: [0, 1000],
  model: {
    discard_policy_probabilities: Array.from({length: 37}, (_, i) => i === 31 ? 1 : 0),
    opponent_shanten_probabilities: Array(3).fill([0.1, 0.5, 0.3, 0.1]),
    terminal_payment_probabilities: Array.from({length: 4}, () =>
      Array.from({length: 4}, () => Array(4).fill([0.8, 0.2]))),
    structured_outcome: {types: ["ron", "tsumo", "draw", "bank"], settlement_mean: [100, -100, 0, 0, 0]},
    final_placement_probabilities: [0.4, 0.3, 0.2, 0.1],
  },
};
function snapshot(position, prediction = null) {
  return structuredClone({status: "playing", updatedAt: "2026-09-10T12:00:00Z",
    autoplay: {enabled: false, ready: false}, autoqueue: {enabled: false},
    position, prediction, lastHandResult: live.lastHandResult,
    finalMatchResult: live.finalMatchResult});
}
const opening = snapshot(live.apply(start), {status: 200, body});
live.apply({name: "ActionDiscardTile", data: {seat: 0, tile: "5z", moqie: true}});
live.apply({name: "ActionChiPengGang", data: {seat: 1, type: 1,
  tiles: ["5z", "5z", "5z"], froms: [1, 1, 0]}});
const called = snapshot(live.apply({name: "ActionDiscardTile", data: {seat: 1, tile: "0p"}}));
const won = snapshot(live.apply({name: "ActionHule", data: {
  hules: [{seat: 2, hand: start.data.tiles.slice(0, 13), hu_tile: "0p", zimo: false,
    count: 3, fu: 30, fans: []}], scores: [25000, 16000, 33000, 25000],
}}));
const finished = {...won, status: "finished", finalMatchResult: {selfSeat: 0,
  players: [2, 0, 3, 1].map((seat, i) => ({seat, rank: i + 1, score: won.lastHandResult.scores[seat]}))}};
const lobby = {...finished, status: "connected", position: null};
const newHand = snapshot(live.apply(start));
const selfDraw = snapshot(live.apply({name: "ActionHule", data: {
  hules: [{seat: 1, hand: start.data.tiles.slice(0, 13), hu_tile: "0p", zimo: true,
    count: 3, fu: 30, fans: []}], scores: [21000, 33000, 23000, 23000],
}}));
live.apply(start);
const forcedReach = {...opening, position: {...opening.position,
  hand: {...opening.position.hand, riichi: true}},
  prediction: {status: 200, body: {...body, action_kind: "forced"}}};

const drawn = snapshot(live.apply({name: "ActionNoTile", data: {
  players: [0, 1, 2, 3].map((seat) => ({tingpai: seat === 1,
    hand: seat === 1 ? start.data.tiles.slice(0, 13) : []})),
  scores: [{old_scores: start.data.scores, delta_scores: [-1000, 3000, -1000, -1000]}],
}}));

const chrome = spawn(process.env.CHROME_PATH || "/usr/bin/google-chrome", [
  "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
  "--disable-background-networking", "--disable-background-timer-throttling",
  "--disable-renderer-backgrounding", "--remote-debugging-pipe",
  `--user-data-dir=${profile}`, "about:blank",
], {stdio: ["ignore", "ignore", "ignore", "pipe", "pipe"]});
browser.attach(chrome);

try {
  const pages = [];
  for (const directory of [baseline, web]) {
    const files = publicFiles(path.join(directory, "public"));
    for (const name of ["tiles.png", "tiles_rects.json"]) {
      files.set(`/assets/${name}`, [path.resolve(directory, "../../vision/static", name),
        name.endsWith("png") ? "image/png" : "application/json"]);
    }
    const server = http.createServer(async (request, response) => {
      const file = files.get(request.url);
      if (!file) {response.writeHead(404).end(); return;}
      response.writeHead(200, {"Content-Type": file[1]});
      response.end(await fs.readFile(file[0]));
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    servers.push(server);
    const {targetId} = await browser.send("Target.createTarget", {url: "about:blank"});
    const {sessionId} = await browser.send("Target.attachToTarget", {targetId, flatten: true});
    const send = (method, params = {}) => browser.send(method, params, sessionId);
    const evaluate = async (expression) => {
      const result = await send("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
      assert.equal(result.exceptionDetails, undefined, JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    await send("Runtime.enable");
    await send("Page.enable");
    await send("Emulation.setDeviceMetricsOverride", {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
    await send("Page.addScriptToEvaluateOnNewDocument", {source: `
      window.fixture = ${JSON.stringify(opening)};
      const realFetch = fetch;
      window.fetch = (url, options) => String(url).includes('/api/')
        ? Promise.resolve(new Response(JSON.stringify(window.fixture))) : realFetch(url, options);
    `});
    await send("Page.navigate", {url: `http://127.0.0.1:${server.address().port}/live/`});
    await wait(700);
    await evaluate("document.fonts.ready.then(() => true)");
    pages.push({send, evaluate, origin: `http://127.0.0.1:${server.address().port}`});
  }
  const results = [];
  for (const [name, fixture] of Object.entries({opening, forcedReach, called, won, finished, lobby, newHand, selfDraw, drawn})) {
    for (const page of pages) await page.evaluate(`window.fixture = ${JSON.stringify(fixture)}`);
    await wait(400);
    const states = await Promise.all(pages.map(page => page.evaluate(`({
      table: document.querySelector('.board').outerHTML,
      choices: document.querySelector('#live-actions').outerHTML,
      result: document.querySelector('#live-hand-result').outerHTML,
      recommendation: document.querySelector('#live-recommendation').outerHTML,
      forecasts: [...document.querySelectorAll('[data-outcome-rank]')].map(x => x.outerHTML)
    })`)));
    assert.deepEqual(states[1], states[0], name);
    const screenshots = await Promise.all(pages.map(page => page.send("Page.captureScreenshot", {format: "png"})));
    for (const [i, screenshot] of screenshots.entries()) {
      await fs.writeFile(path.join(artifactPath, `${name}-${i === 0 ? "baseline" : "staged"}.png`), Buffer.from(screenshot.data, "base64"));
    }
    assert.equal(screenshots[1].data === screenshots[0].data, true, `${name} screenshot differs`);
    results.push(`${name}: identical DOM and screenshot`);
  }
  for (const [width, height] of [[1280,720], [1024,768], [768,1024], [390,844]]) {
    for (const page of pages) await page.send("Emulation.setDeviceMetricsOverride", {width, height, deviceScaleFactor: 1, mobile: false});
    await wait(300);
    const shots = await Promise.all(pages.map(page => page.send("Page.captureScreenshot", {format: "png"})));
    assert.equal(shots[0].data === shots[1].data, true, `${width}x${height} screenshot differs`);
    results.push(`${width}x${height}: identical screenshot`);
  }
  // Exercise the replay page adapter and controls.
  const replay = {gameId: "parity", rounds: [opening.position.round]};
  for (const page of pages) {
    await page.send("Emulation.setDeviceMetricsOverride", {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
    await page.send("Page.navigate", {url: `${page.origin}/review/`});
    await wait(300);
    await page.evaluate("document.fonts.ready.then(() => true)");
    await page.evaluate(`window.fixture = ${JSON.stringify(replay)};
      document.querySelector('[name=url]').value = 'https://mahjongsoul.game.yo-star.com/?paipu=parity';
      document.querySelector('#load-form').requestSubmit();`);
  }
  await wait(400);
  for (const page of pages) {
    await page.send("Page.bringToFront");
    await wait(250);
  }
  const tables = await Promise.all(pages.map(page => page.evaluate("document.querySelector('.board').outerHTML")));
  assert.deepEqual(tables[1], tables[0], "review");
  for (const page of pages) assert.ok(await page.evaluate("document.querySelectorAll('.seat-rack').length === 4"), "review rendered all seats");
  const shots = await Promise.all(pages.map(page => page.send("Page.captureScreenshot", {format: "png"})));
  assert.equal(shots[0].data === shots[1].data, true, "review screenshot differs");
  results.push("review: identical DOM and screenshot");
  const page = pages[1];
  await page.send("Page.navigate", {url: `${page.origin}/live/`});
  await page.send("Page.bringToFront");
  await wait(400);
  await page.evaluate(`window.fixture = ${JSON.stringify({...opening, updatedAt: "2026-09-10T12:00:01Z",
    position: {...opening.position, accountRanks: [
      {id: 10403, score: 2300}, {id: 10712, score: 1234},
      {id: 10501, score: 3500}, null]}})}`);
  await wait(400);
  const labels = await page.evaluate("[...document.querySelectorAll('.account-rank')].map(x => x.textContent).sort()");
  assert.deepEqual(labels, ["C12:12.3", "M3:2300", "S1:3500"]);
  for (const [width, height] of [[1440,1000], [1280,720], [390,844]]) {
    await page.send("Emulation.setDeviceMetricsOverride", {width, height, deviceScaleFactor: 1, mobile: false});
    await wait(200);
    const overflow = await page.evaluate(`(() => {
      return [...document.querySelectorAll('.seat-heading')].some(heading => {
        const box = heading.getBoundingClientRect();
        return [...heading.querySelectorAll('.account-rank, .live-opponent-ev')].some(child => {
          const bounds = child.getBoundingClientRect();
          return bounds.bottom > box.bottom + 1 || bounds.right > box.right + 1 || bounds.left < box.left - 1;
        });
      });
    })()`);
    assert.equal(overflow, false, `account ranks fit player bars at ${width}x${height}`);
  }
  results.push("account ranks: correct labels, omitted bot rank, no player-bar overflow");
  assert.deepEqual(errors, []);
  await fs.writeFile(path.join(artifactPath, "results.json"), JSON.stringify(results, null, 2));
  console.log(results.join("\n"));
} finally {
  browser.close();
  const exited = once(chrome, "exit");
  chrome.kill("SIGTERM");
  await exited;
  for (const server of servers) server.close();
  await fs.rm(profile, {recursive: true, force: true});
}
