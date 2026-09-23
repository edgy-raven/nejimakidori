import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {ReviewCache, readCache, writeCache} from "./review_cache.mjs";
import {ReviewerAccounts} from "./reviewer_accounts.mjs";

const source = {recordId: "sample", region: "en"};
const game = {gameId: "sample", players: [{accountId: 42, seat: 0}],
  rounds: [{events: [{type: "draw", seat: 0}, {type: "discard", seat: 0}]}]};
const prediction = {actor: 0, expert_decisions: {discard_policy: 1},
  discard_options: [{action: 0, tile: "1m", all_shanten: 2},
    {action: 1, tile: "2m", all_shanten: 2}],
  predictions: [{discard_policy_probabilities: [0.6, 0.4],
    opponent_shanten_probabilities: Array.from({length: 3}, () => [0, 0, 0, 1])}]};

test("duplicate loads share work, survive a closed tab and restart, and regrade cached predictions", async t => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "review-cache-test-"));
  t.after(() => fs.rm(directory, {recursive: true, force: true}));
  let model = "epoch-3";
  let downloads = 0;
  let predictions = 0;
  const started = Promise.withResolvers();
  const finish = Promise.withResolvers();
  const options = {directory, modelKey: async () => model,
    download: async () => {
      downloads++;
      return {...game, rounds: [{events: [...game.rounds[0].events, {type: "discard", seat: 1}]}]};
    },
    predict: async (round, eventCount) => {
      predictions++;
      started.resolve();
      await finish.promise;
      return {...prediction, actor: round.events[eventCount].seat};
    }};
  const cache = new ReviewCache(options);
  const closedTab = new AbortController();
  const progress = [];
  const first = cache.load(source, 0, () => {}, closedTab.signal);
  const second = cache.load(source, 0, message => progress.push(message), new AbortController().signal);
  await started.promise;
  closedTab.abort();
  finish.resolve();
  const [review, other] = await Promise.all([first, second]);
  assert.equal(other, review);
  assert.equal(downloads, 1);
  assert.equal(predictions, 1);
  assert.ok(progress.some(update => update.completed === 1 && update.total === 1));
  assert.equal(review.players[0].accountId, 42);
  assert.ok(Math.abs(review.grades[0].score - 75) < 1e-9);
  assert.equal((await cache.load(source, "auto", () => {}, new AbortController().signal)).seat, 0);
  assert.equal(await cache.load(source, 0, () => {}, new AbortController().signal), review);
  assert.equal(predictions, 1);

  const filename = path.join(directory, "reviews/sample-seat-0.json.gz");
  const saved = await readCache(filename);
  saved.game.grades[0].score = -999;
  saved.game.grades[0].matched = -999;
  saved.game.annotations[0][1].review.weight = -999;
  await writeCache(filename, saved);
  const modified = (await fs.stat(filename)).mtimeMs;
  const restarted = new ReviewCache(options);
  assert.deepEqual(await restarted.load(source, 0, () => {}, new AbortController().signal), review);
  assert.equal(predictions, 1);
  assert.equal(downloads, 1);
  const otherSeat = await restarted.load(source, 1, () => {}, new AbortController().signal);
  assert.equal(otherSeat.seat, 1);
  assert.equal(otherSeat.annotations[0][1], null);
  assert.equal(otherSeat.annotations[0][2].actor, 1);
  assert.equal(review.annotations[0][2], null);
  assert.equal(predictions, 2);
  assert.equal(downloads, 1);
  assert.equal((await fs.stat(filename)).mtimeMs, modified);
  model = "epoch-4";
  await restarted.load(source, 0, () => {}, new AbortController().signal);
  assert.equal(predictions, 3);
  assert.equal(downloads, 1);
  assert.equal((await readCache(filename)).model, "epoch-4");
});

test("failed or mixed-model analysis keeps the game and can be retried without another download", async t => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "review-cache-test-"));
  t.after(() => fs.rm(directory, {recursive: true, force: true}));
  let model = "epoch-3";
  let downloads = 0;
  let outcome = "failed";
  const cache = new ReviewCache({directory, modelKey: async () => model,
    download: async () => { downloads++; return game; },
    predict: async () => {
      if (outcome === "failed") throw new Error("Model unavailable");
      if (outcome === "changed") model = "epoch-4";
      return prediction;
    }});
  await assert.rejects(cache.load(source, 0, () => {}, new AbortController().signal), /Model unavailable/);
  assert.equal(await readCache(path.join(directory, "reviews/sample-seat-0.json.gz")), null);
  outcome = "changed";
  await assert.rejects(cache.load(source, 0, () => {}, new AbortController().signal), /model changed/i);
  outcome = "ready";
  assert.ok((await cache.load(source, 0, () => {}, new AbortController().signal)).grades[0].score > 0);
  assert.equal(downloads, 1);
});

test("reviewers rotate serially, remember cooldown across restarts, and recover without retrying bad logs", async t => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "review-accounts-test-"));
  t.after(() => fs.rm(directory, {recursive: true, force: true}));
  const accounts = [];
  for (const name of ["one", "two"]) {
    const envFile = path.join(directory, name + ".env");
    await fs.writeFile(envFile, `MAJSOUL_EN_ACCESS_TOKEN=${name}\n`, {mode: 0o600});
    accounts.push({name, region: "en", envFile});
  }
  let now = 0;
  let blocked = true;
  let active = 0;
  let maximum = 0;
  const visits = [];
  const options = {accounts, stateFile: path.join(directory, "accounts.json.gz"), now: () => now,
    fetchRecord: async (id, region, credentials) => {
      active++;
      maximum = Math.max(maximum, active);
      const name = credentials.MAJSOUL_EN_ACCESS_TOKEN;
      visits.push(name);
      try {
        assert.equal(region, "en");
        await new Promise(resolve => setImmediate(resolve));
        if (id === "bad") throw new Error("record_not_found");
        if (blocked && name === "one") throw {error: {code: 540}};
        return {id, name};
      } finally {active--;}
    }};
  const pool = new ReviewerAccounts(options);
  assert.equal((await pool.fetch(source)).name, "two");
  const restarted = new ReviewerAccounts(options);
  await Promise.all([restarted.fetch(source), restarted.fetch({...source, recordId: "other"})]);
  assert.deepEqual(visits, ["one", "two", "two", "two"]);
  assert.equal(maximum, 1);
  now += 30 * 60 * 1000;
  blocked = false;
  assert.equal((await restarted.fetch(source)).name, "one");
  assert.equal((await restarted.fetch(source)).name, "two");
  await assert.rejects(restarted.fetch({...source, recordId: "bad"}), /record_not_found/);
  assert.equal(visits.at(-1), "one");
  assert.equal(visits.length, 7);

  const allBlocked = new ReviewerAccounts({...options,
    fetchRecord: async () => { throw Object.assign(new Error("session_fetchGameRecord_error_540"), {code: 540}); }});
  await assert.rejects(allBlocked.fetch(source), /cooling down/);
  const noRetry = new ReviewerAccounts({...options, fetchRecord: async () => assert.fail("cooldown ignored")});
  await assert.rejects(noRetry.fetch(source), /cooling down/);
  now += 30 * 60 * 1000;
  const expired = new ReviewerAccounts({...options,
    fetchRecord: async (id, region, credentials) => {
      if (credentials.MAJSOUL_EN_ACCESS_TOKEN === "one") {
        throw new Error("yostar__user_quick-login_401");
      }
      return {id, name: "two"};
    }});
  assert.equal((await expired.fetch(source)).name, "two");
});
