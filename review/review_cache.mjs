import fs from "node:fs/promises";
import path from "node:path";
import util from "node:util";
import zlib from "node:zlib";
import {seatFromLink} from "./public/review/link.js";
import {annotateGame} from "./replay_annotations.mjs";

export async function readCache(filename) {
  let data;
  try {
    data = await fs.readFile(filename);
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
  return JSON.parse(await util.promisify(zlib.gunzip)(data));
}

export async function writeCache(filename, value) {
  await fs.mkdir(path.dirname(filename), {recursive: true, mode: 0o700});
  await fs.writeFile(`${filename}.tmp`,
    await util.promisify(zlib.gzip)(JSON.stringify(value)), {mode: 0o600});
  await fs.rename(`${filename}.tmp`, filename);
}

export class ReviewCache {
  constructor({directory, modelKey, download, predict}) {
    this.directory = directory;
    this.modelKey = modelKey;
    this.download = download;
    this.predict = predict;
    this.jobs = new Map();
    this.memory = new Map();
    this.games = new Map();
  }

  async load(source, seat, onProgress, signal) {
    if (seat !== "auto" && ![0, 1, 2, 3].includes(seat)) {
      throw new Error("Choose a valid seat.");
    }
    onProgress({message: "Loading…"});
    let gameJob = this.games.get(source.recordId);
    if (!gameJob) {
      gameJob = (async () => {
        const filename = path.join(this.directory, "games", `${source.recordId}.json.gz`);
        const saved = await readCache(filename);
        if (saved) return saved;
        const game = await this.download(source);
        await writeCache(filename, game);
        return game;
      })().finally(() => this.games.delete(source.recordId));
      this.games.set(source.recordId, gameJob);
    }
    const game = await gameJob;
    seat = seat === "auto" ? seatFromLink(source, game.players) ?? 0 : seat;
    const key = `${source.recordId}-seat-${seat}`;
    let job = this.jobs.get(key);
    if (!job) {
      job = {listeners: new Set(), progress: {message: "Loading…"}};
      job.promise = this.review(game, seat, key, message => {
        job.progress = message;
        for (const listener of job.listeners) listener(message);
      }).finally(() => this.jobs.delete(key));
      this.jobs.set(key, job);
    }
    const unsubscribe = () => job.listeners.delete(onProgress);
    job.listeners.add(onProgress);
    onProgress(job.progress);
    signal.addEventListener("abort", unsubscribe, {once: true});
    try {
      // Closing one tab detaches that subscriber; the shared review finishes.
      return await job.promise;
    } finally {
      unsubscribe();
      signal.removeEventListener("abort", unsubscribe);
    }
  }

  async review(game, seat, key, onProgress) {
    const model = await this.modelKey();
    if (this.memory.get(key)?.model === model) return this.memory.get(key).game;
    const filename = path.join(this.directory, "reviews", `${key}.json.gz`);
    const cached = await readCache(filename);
    const result = await annotateGame(game, {
      seat,
      signal: new AbortController().signal,
      onProgress: cached?.model === model ? () => {} : ({completed, total}) =>
        onProgress({message: "Reviewing…", completed, total}),
      predict: cached?.model === model ? async (round, eventCount) =>
        cached.game.annotations[game.rounds.indexOf(round)][eventCount]
        : this.predict,
    });
    if (await this.modelKey() !== model) {
      throw new Error("The model changed. Load the review again.");
    }
    if (cached?.model !== model) await writeCache(filename, {model, game: result});
    this.memory.set(key, {model, game: result});
    if (this.memory.size > 16) this.memory.delete(this.memory.keys().next().value);
    return result;
  }
}
