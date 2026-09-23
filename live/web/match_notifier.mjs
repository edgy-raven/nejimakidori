import { createHash } from "node:crypto";
import fs from "node:fs";

export class MatchNotifier {
  constructor(settings, request = fetch) {
    this.settings = settings;
    this.request = request;
    this.token = fs.readFileSync(settings.tokenPath, "utf8").trim();
    this.reports = fs.existsSync(settings.statePath)
      ? JSON.parse(fs.readFileSync(settings.statePath, "utf8")) : {};
    this.busy = false;
  }

  save() {
    fs.writeFileSync(`${this.settings.statePath}.tmp`,
      JSON.stringify(this.reports, null, 2) + "\n", {mode: 0o600});
    fs.renameSync(`${this.settings.statePath}.tmp`, this.settings.statePath);
  }

  begin(gameId, account) {
    if (!gameId) throw new Error("Started game is missing its game ID");
    if (this.reports[gameId]) return;
    this.reports[gameId] = {kind: "match", gameId, account: structuredClone(account),
      accountAtStart: structuredClone(account), completedAt: null,
      nextAttemptAt: 0, attempts: 0, messageId: null, error: null, blocked: false};
    this.save();
  }

  finish(gameId, account, result, now = Date.now()) {
    if (!gameId) throw new Error("Completed game is missing its game ID");
    this.begin(gameId, account);
    const report = this.reports[gameId];
    if (report.completedAt) return;
    const player = result.players.find(player => player.seat === result.selfSeat);
    Object.assign(report, {placement: player.rank, score: player.score,
      completedAt: new Date(now).toISOString(), nextAttemptAt: now + 3000});
    this.save();
  }

  requestHelp(block, nickname) {
    if (this.attention) return;
    const gameId = `help:${block.id}`;
    this.reports[gameId] = {kind: "help", gameId, attention: block,
      content: `${nickname} needs help. ` +
        (block.returnToLobby
          ? "Gameplay and requeue are paused; known results will be cleared to the lobby.\n"
          : "Automatic clicks are paused.\n") +
        `Reason: ${block.reason}\nState: ${block.status}\n` +
        "Please fix the issue in the live controls, then click Resume automation. " +
        "The bot will remain paused until you explicitly resume it.",
      resolvedAt: null, nextAttemptAt: 0, attempts: 0,
      messageId: null, error: null, blocked: false};
    this.save();
  }

  gameplayFailure(failure, nickname) {
    const gameId = `gameplay:${failure.gameId}`;
    if (this.reports[gameId]) return;
    this.reports[gameId] = {kind: "gameplay", gameId,
      content: `${nickname} needs a controller fix.\n` +
        `Reason: ${failure.reason}\n` +
        "The failed move was abandoned. Play continues on the next decision; " +
        "requeue is disabled until explicitly re-enabled.\n" +
        `Game: ${failure.gameId}`,
      nextAttemptAt: 0, attempts: 0, messageId: null, error: null, blocked: false};
    this.save();
  }

  get attention() {
    return Object.values(this.reports).find(report =>
      report.kind === "help" && !report.resolvedAt)?.attention || null;
  }

  resolveHelp(id) {
    this.reports[`help:${id}`].resolvedAt = new Date().toISOString();
    this.save();
  }

  updateAccount(gameId, account, now = Date.now()) {
    const report = this.reports[gameId];
    if (!report || report.attempts || !account) return;
    report.account = structuredClone(account);
    report.nextAttemptAt = now + 3000;
    this.save();
  }

  get pendingReports() {
    return Object.values(this.reports).filter(report =>
      (report.kind !== "match" || report.completedAt) && !report.messageId &&
      !(report.kind === "help" && report.resolvedAt));
  }

  get snapshot() {
    const reports = Object.values(this.reports);
    return {enabled: true, pending: this.pendingReports.length,
      lastMessageId: reports.findLast(report => report.messageId)?.messageId || null,
      error: this.pendingReports.find(report => report.error)?.error || null};
  }

  async flush(now = Date.now()) {
    if (this.busy) return;
    const report = this.pendingReports.find(report =>
      !report.blocked && report.nextAttemptAt <= now);
    if (!report) return;
    this.busy = true;
    try {
      let content;
      if (report.kind === "help" || report.kind === "gameplay") {
        content = report.content;
      } else {
        const rank = report.account.rank;
        const tier = Math.floor(rank.id / 100) % 100;
        const rankName = {1: "Novice", 2: "Adept", 3: "Expert", 4: "Master",
          5: "Saint", 7: "Celestial"}[tier];
        const points = tier === 7 ? (rank.score / 100).toFixed(1) : rank.score;
        const oldRank = report.accountAtStart.rank;
        const oldTier = Math.floor(oldRank.id / 100) % 100;
        const oldPoints = oldTier === 7 ? oldRank.score / 100 : oldRank.score;
        const pointDelta = points - oldPoints;
        const copperDelta = report.account.copper - report.accountAtStart.copper;
        content = `${report.account.nickname} finished a game.\n` +
          `Placement: ${report.placement} · Score: ${report.score.toLocaleString("en-US")}\n` +
          `Rank: ${rankName} ${rank.id % 100} (${points} points; ${pointDelta >= 0 ? "+" : ""}${pointDelta} points)\n` +
          `Copper: ${report.account.copper.toLocaleString("en-US")} (${copperDelta >= 0 ? "+" : ""}${copperDelta.toLocaleString("en-US")})\n` +
          `Game: ${report.gameId}`;
      }
      report.attempts += 1;
      report.nextAttemptAt = now + 60000;
      this.save();
      let response;
      try {
        response = await this.request(
          `https://discord.com/api/v10/channels/${this.settings.channelId}/messages`, {
            method: "POST",
            headers: {Authorization: `Bot ${this.token}`, "Content-Type": "application/json"},
            body: JSON.stringify({content, allowed_mentions: {parse: []},
              nonce: createHash("sha256").update(report.gameId).digest("hex").slice(0, 24),
              enforce_nonce: true}),
            signal: AbortSignal.timeout(15000),
          });
      } catch (error) {
        if (!(error instanceof TypeError) && error.name !== "TimeoutError") throw error;
        report.error = "Discord connection failed; retrying in 60 seconds.";
        this.save();
        return;
      }
      if (response.ok) {
        report.messageId = (await response.json()).id;
        report.error = null;
      } else if (response.status === 429) {
        report.nextAttemptAt = now + Math.ceil((await response.json()).retry_after * 1000);
        report.error = "Discord rate limit; waiting to retry.";
      } else {
        report.error = `Discord rejected the report (HTTP ${response.status}).`;
        report.blocked = response.status < 500;
      }
      this.save();
    } finally {
      this.busy = false;
    }
  }
}
