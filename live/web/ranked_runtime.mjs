import fs from "node:fs";

// Preserve queue decisions across service restarts, including deliberate stops.
export class RankedRuntime {
  constructor(path) {
    this.path = path;
    this.body = fs.existsSync(path) ? fs.readFileSync(path, "utf8") : null;
  }

  restore(queue, autoplay) {
    if (!this.body) {
      queue.enabled = true;
      return autoplay;
    }
    const saved = JSON.parse(this.body);
    Object.assign(queue, saved.queue);
    return saved.autoplay;
  }

  save(autoplay, queue) {
    const body = JSON.stringify({autoplay, queue: {
      enabled: queue.enabled, error: queue.error, games: queue.games,
      pauseUntil: queue.pauseUntil, completedGameId: queue.completedGameId,
      continueSession: queue.continueSession,
    }});
    if (body === this.body) return;
    fs.writeFileSync(`${this.path}.tmp`, body, {mode: 0o600});
    fs.renameSync(`${this.path}.tmp`, this.path);
    this.body = body;
  }
}
