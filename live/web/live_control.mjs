import { timingSafeEqual } from "node:crypto";
import fs from "node:fs";
import http from "node:http";

export const controlPin = (
  process.env.MAHJONG_SOUL_CONTROL_PIN ||
  fs.readFileSync(new URL(".live-control-pin", import.meta.url), "utf8")
).trim();
if (!/^\d{4}$/.test(controlPin)) {
  throw new Error("Invalid live control PIN; expected four digits");
}
export const liveControlUrl = process.env.MAHJONG_SOUL_CONTROL_URL ||
  "http://127.0.0.1:8795";

export function authorizedLiveControl(request) {
  const value = Buffer.from(request.headers.authorization || "");
  const expected = Buffer.from(`Bearer ${controlPin}`);
  return value.length === expected.length &&
    timingSafeEqual(value, expected);
}

export function startLiveControl(command, status) {
  const server = http.createServer(async (request, response) => {
    const reply = (status, body) => {
      response.writeHead(status, {
        "Content-Type": "application/json", "Cache-Control": "no-store",
      });
      response.end(JSON.stringify(body));
    };
    if (!authorizedLiveControl(request)) {
      reply(401, { error: "Control PIN required." });
      return;
    }
    if (request.method === "GET" && request.url === "/status") {
      reply(200, { ...status(), busy: server.busy });
      return;
    }
    if (request.method !== "POST" ||
        !["/join", "/command"].includes(request.url)) {
      reply(404, { error: "Unknown controller command." });
      return;
    }
    let body = "";
    for await (const chunk of request) {
      body += chunk;
      if (body.length > 1024) {
        reply(413, { error: "Request too large." });
        return;
      }
    }
    let payload;
    try {
      payload = JSON.parse(body);
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      reply(400, { error: "Invalid JSON." });
      return;
    }
    if (request.url === "/join") payload = {
      action: "join", roomId: payload?.roomId,
    };
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      reply(400, { error: "Expected a command object." });
      return;
    }
    if (server.busy) {
      reply(409, { error: "A browser command is already in progress." });
      return;
    }
    server.busy = true;
    const started = performance.now();
    try {
      const result = await command(payload);
      reply(200, {
        ...result, action: payload.action,
        elapsedMs: Math.round(performance.now() - started),
      });
    } catch (error) {
      reply(error instanceof TypeError ? 400 : 409, { error: error.message });
    } finally {
      server.busy = false;
    }
  });
  server.busy = false;
  const url = new URL(liveControlUrl);
  if (url.hostname !== "127.0.0.1") {
    throw new Error("Live control must bind to 127.0.0.1");
  }
  server.listen(Number(url.port), url.hostname);
  return server;
}
