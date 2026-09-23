import assert from "node:assert/strict";
import { PassThrough } from "node:stream";
import test from "node:test";
import { BrowserTransport } from "./browser_transport.mjs";

test("browser commands correlate out-of-order replies across fragmented pipe frames", async () => {
  const events = [];
  const input = new PassThrough();
  const output = new PassThrough();
  const transport = new BrowserTransport((message) => events.push(message));
  transport.attach({stdio: [null, null, null, input, output]});
  const first = transport.send("Page.enable", {}, "game");
  const second = transport.send("Network.enable", {}, "game");
  const commands = input.read().toString().split("\0").filter(Boolean).map(JSON.parse);
  assert.deepEqual(commands.map(({method, sessionId}) => [method, sessionId]), [
    ["Page.enable", "game"], ["Network.enable", "game"],
  ]);
  const notification = {method: "Network.webSocketFrameReceived", params: {payload: "tile"}};
  const frames = Buffer.from([
    {id: commands[1].id, result: {enabled: true}}, notification,
    {id: commands[0].id, error: {message: "closed"}},
    {id: 999, result: {}}, // A late response must not become a game notification.
  ].map(JSON.stringify).join("\0") + "\0");
  output.write(frames.subarray(0, 13));
  assert.equal(events.length, 0);
  output.write(frames.subarray(13));
  assert.deepEqual(await second, {enabled: true});
  await assert.rejects(first, /closed/);
  assert.deepEqual(events, [notification]);
  transport.close();
});
