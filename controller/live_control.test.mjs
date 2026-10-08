import assert from "node:assert/strict";
import { once } from "node:events";
import test from "node:test";

process.env.MAHJONG_SOUL_CONTROL_PIN = "1234";
process.env.MAHJONG_SOUL_CONTROL_URL = "http://127.0.0.1:0";
const { startLiveControl } = await import("./live_control.mjs");

test("authenticated commands complete once and exclude concurrent input", async (t) => {
  let finish;
  let entered;
  const started = new Promise((resolve) => { entered = resolve; });
  const server = startLiveControl(async (command) => {
    if (command.action === "join") {
      entered();
      await new Promise((resolve) => { finish = resolve; });
      return { roomId: command.roomId };
    }
    throw new TypeError("Unknown command");
  }, () => ({ status: "connected", autoplay: { enabled: true } }));
  t.after(() => server.close());
  await once(server, "listening");
  const url = `http://127.0.0.1:${server.address().port}`;
  const headers = { Authorization: "Bearer 1234" };
  assert.equal((await fetch(`${url}/status`)).status, 401);
  assert.deepEqual(await (await fetch(`${url}/status`, { headers })).json(), {
    status: "connected", autoplay: { enabled: true }, busy: false,
  });
  assert.equal((await fetch(`${url}/command`, {
    method: "POST", headers, body: "{",
  })).status, 400);
  assert.equal((await fetch(`${url}/command`, {
    method: "POST", headers, body: "null",
  })).status, 400);
  const joining = fetch(`${url}/join`, {
    method: "POST", headers, body: JSON.stringify({ roomId: 12345 }),
  });
  await started;
  assert.equal((await (await fetch(`${url}/status`, { headers })).json()).busy, true);
  assert.equal((await fetch(`${url}/command`, {
    method: "POST", headers, body: JSON.stringify({ action: "click" }),
  })).status, 409);
  finish();
  const response = await joining;
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.roomId, 12345);
  assert.equal(result.action, "join");
  assert.ok(result.elapsedMs >= 0);
  assert.equal((await fetch(`${url}/command`, {
    method: "POST", headers, body: JSON.stringify({ action: "unknown" }),
  })).status, 400);
  assert.equal((await (await fetch(`${url}/status`, { headers })).json()).busy, false);
  // Background room monitoring holds the same lock as HTTP commands.
  server.busy = true;
  assert.equal((await fetch(`${url}/command`, {
    method: "POST", headers, body: JSON.stringify({ action: "click" }),
  })).status, 409);
  assert.equal((await (await fetch(`${url}/status`, { headers })).json()).busy, true);
  server.busy = false;
});
