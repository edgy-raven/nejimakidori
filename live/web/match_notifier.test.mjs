import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { MatchNotifier } from "./match_notifier.mjs";

const result = {selfSeat: 2, players: [
  {seat: 0, rank: 1, score: 32000}, {seat: 2, rank: 2, score: 27000},
  {seat: 1, rank: 3, score: 23000}, {seat: 3, rank: 4, score: 18000},
]};

function settings(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "match-report-"));
  t.after(() => fs.rmSync(directory, {recursive: true}));
  const value = {tokenPath: path.join(directory, "token"),
    statePath: path.join(directory, "reports.json"), channelId: "123"};
  fs.writeFileSync(value.tokenPath, "fake-token");
  return value;
}

test("completed match reports settled account values once", async t => {
  const calls = [];
  const notifier = new MatchNotifier(settings(t), async (url, options) => {
    calls.push({url, body: JSON.parse(options.body)});
    return Response.json({id: "sent-1"});
  });
  const account = {accountId: 42, nickname: "Account", copper: 7000,
    rank: {id: 10203, score: 800}};
  notifier.finish("game-1", account, result, 0);
  account.copper = 9200;
  account.rank = {id: 10301, score: 300};
  notifier.updateAccount("game-1", account, 1000);
  await notifier.flush(4000);
  assert.equal(calls.length, 1);
  assert.match(calls[0].body.content, /Placement: 2 · Score: 27,000/);
  assert.match(calls[0].body.content, /Copper: 9,200 \(\+2,200\)/);
  notifier.finish("game-1", account, result, 5000);
  await notifier.flush(10000);
  assert.equal(calls.length, 1);
  assert.equal(notifier.snapshot.pending, 0);
});

test("delivery failures remain visible and retain the report", async t => {
  const responses = [new TypeError("network unavailable"),
    new Response("forbidden", {status: 403})];
  const notifier = new MatchNotifier(settings(t), async () => {
    const response = responses.shift();
    if (response instanceof Error) throw response;
    return response;
  });
  const account = {nickname: "Account", copper: 2500,
    rank: {id: 10201, score: 100}};
  notifier.finish("game-2", account, result, 0);
  await notifier.flush(3000);
  assert.match(notifier.snapshot.error, /connection failed/);
  await notifier.flush(65000);
  assert.match(notifier.snapshot.error, /403/);
  assert.equal(notifier.snapshot.pending, 1);
});
