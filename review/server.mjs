import {parseReviewLink} from "./public/review/link.js";
import {decodeGame} from "./replay_record.mjs";
import {ReviewCache} from "./review_cache.mjs";
import {ReviewerAccounts} from "./reviewer_accounts.mjs";
import { publicFiles } from "./public_files.mjs";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";


import {
  authorizedLiveControl,
  liveControlUrl,
} from "../live/web/live_control.mjs";
import {
  configuredRegions,
  fetchSessionRecord,
  hasSession,
} from "./session_record.mjs";

const projectPath = path.dirname(fileURLToPath(import.meta.url));
const publicPath = path.join(projectPath, "public");
const controllerTilesPath = path.resolve(
  projectPath,
  "../vision/static",
);
const host = process.env.HOST || "127.0.0.1";
const port = Number(process.env.PORT || 8793);
const modelUrl = process.env.MODEL_URL || "http://127.0.0.1:8792";
const liveStatePath =
  process.env.MAHJONG_SOUL_LIVE_STATE || "/tmp/nejimakidori-live.json";
const liveScreenshotPath = process.env.MAHJONG_SOUL_SCREENSHOT ||
  "/tmp/mahjongsoul-nejimakidori.png";
const maximumRecordBytes = 8 * 1024 * 1024;
const maximumRequestBytes = 16 * 1024;
const staticFiles = publicFiles(publicPath);
staticFiles.set("/assets/tiles.png", [
  path.join(controllerTilesPath, "tiles.png"), "image/png",
]);
staticFiles.set("/assets/tiles_rects.json", [
  path.join(controllerTilesPath, "tiles_rects.json"), "application/json",
]);
const reviewDirectory = process.env.MAHJONG_SOUL_REVIEW_CACHE ||
  path.join(projectPath, ".review-cache");
const reviewerAccounts = process.env.MAHJONG_SOUL_REVIEW_ACCOUNTS
  ? new ReviewerAccounts({
    accounts: JSON.parse(fs.readFileSync(process.env.MAHJONG_SOUL_REVIEW_ACCOUNTS, "utf8")),
    stateFile: path.join(reviewDirectory, "accounts.json.gz"),
  }) : null;
const reviews = new ReviewCache({
  directory: reviewDirectory,
  modelKey: async () => {
    const response = await fetch(modelUrl + "/health", {signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error("Could not identify the review model.");
    const model = await response.json();
    return JSON.stringify([model.model, model.model_contract,
      process.env.MAHJONG_SOUL_REVIEW_MODEL_REVISION || ""]);
  },
  download: async source => {
    const record = reviewerAccounts?.accounts.some(account => account.region === source.region)
      ? await reviewerAccounts.fetch(source)
      : hasSession(source.region) ? await fetchSessionRecord(source.recordId, source.region)
        : await fetchRecord(source.archiveId);
    return {...decodeGame(source.recordId, record.data), players: record.players};
  },
  predict: async (round, eventCount, signal) => {
    const response = await fetch(modelUrl + "/predict-replay", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({round, event_count: eventCount}),
      signal: AbortSignal.any([signal, AbortSignal.timeout(20000)]),
    });
    if (!response.ok) {
      throw new Error("Could not annotate " + round.label + ", event " + eventCount +
        " (model HTTP " + response.status + ").");
    }
    const result = await response.json();
    return {...result.position, predictions: result.inference.predictions,
      model_latency_ms: result.inference.latency_ms,
      terminal_payment_values: result.inference.metadata.payment_values};
  },
});
const redirects = new Map([
  ["/", "review/"],
  ["/review", "review/"],
  ["/live", "live/"],
]);

function send(response, status, body, contentType, cacheControl = "no-store") {
  const value = Buffer.isBuffer(body) ? body : Buffer.from(body);
  response.writeHead(status, {
    "Content-Type": contentType,
    "Content-Length": value.length,
    "Cache-Control": cacheControl,
    "X-Content-Type-Options": "nosniff",
  });
  response.end(value);
}

function sendJson(response, status, body) {
  send(response, status, JSON.stringify(body), "application/json");
}

function readBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > maximumRequestBytes) {
        reject(new Error("request_too_large"));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    request.on("error", reject);
  });
}

async function fetchRecord(recordId) {
  const url =
    "https://record-v2.maj-soul.com:5333/majsoul/game_record/" +
    encodeURIComponent(recordId);
  const response = await fetch(url, {
    headers: { "User-Agent": "nejimakidori-local-replay" },
    signal: AbortSignal.timeout(20000),
  });
  if (!response.ok) {
    throw new Error(`record_archive_http_${response.status}`);
  }
  const contentLength = Number(response.headers.get("content-length") || 0);
  if (contentLength > maximumRecordBytes) {
    throw new Error("record_too_large");
  }
  const data = Buffer.from(await response.arrayBuffer());
  if (!data.length || data.length > maximumRecordBytes) {
    throw new Error(data.length ? "record_too_large" : "record_not_found");
  }
  if (data.subarray(0, 16).toString("utf8").includes("<?xml")) {
    throw new Error("record_not_found");
  }
  return {data, players: []};
}

function errorMessage(error) {
  const message = error instanceof Error
    ? error.message
    : error?.error?.message ||
      `mahjong_soul_error_${error?.error?.code ?? "unknown"}`;
  if (/record_archive_http_|record_not_found/.test(message)) {
    return "This log is unavailable to the review account or public archive.";
  }
  if (error.name === "TimeoutError") {
    return "Mahjong Soul's record archive timed out.";
  }
  return message.replaceAll("_", " ");
}

async function replay(request, response) {
  const controller = new AbortController();
  response.once("close", () => vision.abort());
  response.writeHead(200, {
    "Content-Type": "application/x-ndjson; charset=utf-8",
    "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
    "X-Accel-Buffering": "no",
  });
  const write = message => {
    if (!response.destroyed) response.write(`${JSON.stringify(message)}\n`);
  };
  try {
    const payload = JSON.parse(await readBody(request));
    const source = parseReviewLink(payload.url);
    const game = await reviews.load(source, payload.seat,
      progress => write({type: "progress", ...progress}), vision.signal);
    write({type: "complete", game});
  } catch (error) {
    if (!vision.signal.aborted) {
      write({type: "error", message: errorMessage(error)});
    }
  } finally {
    response.end();
  }
}

function staticFile(response, pathname) {
  const file = staticFiles.get(pathname);
  if (!file) {
    sendJson(response, 404, { error: "not_found" });
    return;
  }
  send(
    response,
    200,
    fs.readFileSync(file[0]),
    file[1],
    ["text/html; charset=utf-8", "text/javascript; charset=utf-8",
      "text/css; charset=utf-8", "application/json"].includes(file[1])
      ? "no-store"
      : "public, max-age=3600",
  );
}

function sendLiveState(response) {
  try {
    const snapshot = JSON.parse(fs.readFileSync(liveStatePath, "utf8"));
    if (Date.now() - Date.parse(snapshot.updatedAt) > 5000) {
      snapshot.status = "offline";
      snapshot.controls = {};
      snapshot.autoplay = { enabled: false, friendlyGame: false, ready: false };
    }
    sendJson(response, 200, snapshot);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    sendJson(response, 200, {
      status: "offline",
      controls: {},
      autoplay: { enabled: false, friendlyGame: false, ready: false },
      updatedAt: null,
      position: null,
      prediction: null,
    });
  }
}

async function liveCommand(request, response, action) {
  if (!authorizedLiveControl(request)) {
    sendJson(response, 401, { error: "Control PIN required." });
    return;
  }
  try {
    const payload = JSON.parse(await readBody(request));
    const controllerResponse = await fetch(`${liveControlUrl}/command`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: request.headers.authorization,
      },
      body: JSON.stringify({ ...payload, action }),
      signal: AbortSignal.timeout(90000),
    });
    sendJson(response, controllerResponse.status, await controllerResponse.json());
  } catch (error) {
    sendJson(response, 503, { error: `Controller unavailable: ${errorMessage(error)}` });
  }
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url || "/", "http://localhost");
  if (request.method === "GET" && redirects.has(url.pathname)) {
    response.writeHead(308, { Location: redirects.get(url.pathname) });
    response.end();
    return;
  }
  if (request.method === "GET" && url.pathname === "/health") {
    sendJson(response, 200, { ok: true, service: "nejimakidori-web" });
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/live") {
    sendLiveState(response);
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/live/screenshot") {
    try {
      const snapshot = JSON.parse(fs.readFileSync(liveStatePath, "utf8"));
      if ((!["connected", "in_room", "finished"].includes(snapshot.status) &&
           !(snapshot.automationBlock && snapshot.status === "playing")) ||
          Date.now() - Date.parse(snapshot.updatedAt) > 5000) {
        sendJson(response, 409, { error: "No active lobby screenshot." });
        return;
      }
      send(response, 200, fs.readFileSync(`${liveScreenshotPath}.jpg`), "image/jpeg");
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      sendJson(response, 404, { error: "Screenshot not available yet." });
    }
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/live/autoqueue") {
    await liveCommand(request, response, "autoqueue");
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/live/login") {
    await liveCommand(request, response, "login");
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/live/click") {
    await liveCommand(request, response, "lobby_click");
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/live/join") {
    await liveCommand(request, response, "join");
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/replay") {
    await replay(request, response);
    return;
  }
  if (request.method === "GET") {
    staticFile(response, url.pathname);
    return;
  }
  sendJson(response, 405, { error: "method_not_allowed" });
});

server.listen(port, host, () => {
  console.log(`Nejimakidori web listening on http://${host}:${port}`);
  console.log(
    "Pages: GET /review/, GET /live/; " +
    "APIs: GET /api/live, POST /api/replay",
  );
  console.log(
    `Session-backed regions: ${[...new Set([...configuredRegions(),
      ...(reviewerAccounts?.accounts.map(account => account.region) || [])])].join(", ") || "none"}; ` +
    `reviewer accounts: ${reviewerAccounts?.accounts.length || 0}`,
  );
});
