import { ResultControls } from "./mahjongsoul_result_controls.mjs";
import { readFileSync } from "node:fs";

const rooms = JSON.parse(readFileSync(new URL("./ranked_rooms.json", import.meta.url)));
export const rankedSchedule = JSON.parse(
  readFileSync(new URL("./ranked_schedule.json", import.meta.url)),
);

export class Autoqueue extends ResultControls {
  enabled = false;
  account = null;
  fresh = false;
  pending = false;
  error = null;
  acknowledged = false;
  stalledSince = null;
  currentModeId = null;
  settlingUntil = 0;
  postgameDeadline = 0;
  resultStage = null;
  games = 0;
  pauseUntil = 0;
  completedGameId = null;
  continueSession = false;

  constructor(random = Math.random) {
    super(8);
    this.random = random;
  }

  schedule(now = Date.now()) {
    const hour = Number(new Intl.DateTimeFormat("en-GB", {
      timeZone: rankedSchedule.timeZone, hour: "2-digit", hourCycle: "h23",
    }).format(now));
    return {timeZone: rankedSchedule.timeZone,
      windows: rankedSchedule.windows.map(({startHour, endHour}) =>
        `${String(startHour).padStart(2, "0")}:00–${String(endHour).padStart(2, "0")}:00`),
      open: rankedSchedule.windows.some(({startHour, endHour}) =>
        hour >= startHour && hour < endHour),
      pauseUntil: this.pauseUntil || null, games: this.games,
      continuationChance: rankedSchedule.continuationBase ** this.games};
  }

  shouldIdleBrowser(now = Date.now()) {
    return !this.schedule(now).open || this.pauseUntil > now;
  }

  finish(gameId, now = Date.now()) {
    if (!gameId) throw new Error("Finished ranked game has no game ID.");
    if (gameId === this.completedGameId) return;
    this.completedGameId = gameId;
    this.games += 1;
    this.continueSession = this.random() < rankedSchedule.continuationBase **
      this.games;
    if (!this.continueSession) {
      this.pauseUntil = now + (
        rankedSchedule.breakMinutes.minimum +
        rankedSchedule.breakMinutes.range * this.random()
      ) * 60000;
    }
  }

  updateAccount(account) {
    this.account = account;
    this.fresh = Boolean(account?.rank);
    this.observed = null;
  }

  notification({name, data}, now = Date.now()) {
    if (name === "NotifyAccountUpdate" && this.account) {
      const copper = data.update.numerical.find(item => item.id === 100002);
      if (copper) {
        this.account.copper = copper.final;
        this.observed = null;
      }
    }
    if (name === "NotifyAccountLevelChange" && this.account &&
        data.final.id >= 10101 && data.final.id < 20000) {
      this.account.rank = data.final;
      this.observed = null;
    }
    if (["NotifyMatchGameStart", "NotifyGameEndResult", "NotifyGameTerminate"].includes(name)) {
      this.settlingUntil = name === "NotifyMatchGameStart" ? 0 : now + 3000;
      this.postgameDeadline = name === "NotifyMatchGameStart" ? 0 : now + 60000;
      this.pending = false;
      this.observed = null;
    }
    if (name === "NotifyMatchTimeout") {
      this.stop("Matchmaking timed out. Enable autoqueue to try again.");
    }
  }

  stop(reason) {
    this.error = reason;
    this.enabled = false;
    this.pending = false;
    this.observed = null;
  }

  get snapshot() {
    const room = this.account?.rank && rooms.findLast(room =>
      this.account.rank.id >= room.rankMin && this.account.rank.id <= room.rankMax);
    const blockedReason = this.error || (!this.fresh
      ? "Waiting for refreshed rank and copper."
      : !room ? "No eligible ranked room."
      : this.account.copper < room.copperMin
      ? `Insufficient copper: ${room.name} ${room.length} needs ${room.copperMin.toLocaleString("en-US")}.`
      : null);
    return {enabled: this.enabled, rank: this.account?.rank ?? null,
      copper: this.account?.copper ?? null, room: room || null,
      fresh: this.fresh, pending: this.pending, blockedReason,
      schedule: this.schedule()};
  }

  next(status, targets, now = Date.now(), screen = {}) {
    if (status === "playing") {
      this.reset();
      this.stalledSince = null;
    }
    if (!["finished", "in_room", "connected"].includes(status)) return null;
    if (this.pending && !this.acknowledged && now - this.clickedAt >= 15000) {
      throw new Error(`No acknowledgement after ${this.clickedControl}.`);
    }
    if (now < this.settlingUntil || this.pending) return null;
    if (screen.screen === "result waiting") {
      this.stalledSince = null;
      this.observed = null;
      return null;
    }
    if (screen.screen === "rank up") {
      throw new Error("Rank-up screen requires manual help.");
    }
    const schedule = this.schedule(now);
    if (this.pauseUntil && now >= this.pauseUntil) {
      this.pauseUntil = 0;
      this.games = 0;
      this.continueSession = false;
    }
    if (!schedule.open && status === "connected") this.games = 0;
    const canQueue = this.enabled && schedule.open && !this.pauseUntil;
    const {room, blockedReason} = this.snapshot;
    // Rematch only preserves our policy when the just-finished mode is still
    // the highest eligible room and its current copper requirement is met.
    let target = canQueue && this.continueSession && !blockedReason &&
      room.modeId === this.currentModeId && status === "finished" &&
      screen.screen === "match end"
      ? targets.find(target => ["one more game", "one more match",
          "another game", "play again"].includes(target.control)) : null;
    const allowedScreen = ["match end", "hand result", "rematch confirmation", "daily quest", "chest reward",
      "star up"].includes(screen.screen);
    if (allowedScreen && this.resultStage !== `${screen.screen}:${screen.stage || ""}`) {
      this.reset();
      this.resultStage = `${screen.screen}:${screen.stage || ""}`;
    }
    if (screen.screen === "rematch confirmation") {
      target = targets.find(target => target.control === (
        canQueue && this.continueSession && !blockedReason &&
        room.modeId === this.currentModeId ? "confirm" : "cancel"));
    }
    if (allowedScreen && screen.screen !== "rematch confirmation") target ||= targets.find(target => [
      "confirm", "next", "continue", "dismiss reward",
    ].includes(target.control));
    if (!target && canQueue && status !== "in_room" && !this.error) {
      if (!this.fresh && ["home", "ranked rooms", "ranked modes"].includes(screen.screen)) {
        target = {control: "refresh account", center: [0, 0]};
      } else if (!blockedReason) {
        if (screen.screen === "home") {
          target = targets.find(target => target.control === "ranked match");
        } else if (screen.screen === "ranked rooms") {
          target = targets.find(target => target.control === `${room.name.toLowerCase()} room`);
          if (!target) {
            const visible = targets.filter(target => target.room);
            if (visible.length) target = {control: "scroll rooms", center: [950, 460],
              deltaY: rooms.findIndex(item => item.name === room.name) <
                rooms.findIndex(item => item.name === visible[0].room) ? -360 : 360};
          }
        } else if (screen.screen === "ranked modes") {
          target = targets.find(target => target.control ===
            (screen.room === room.name ? `4 player ${room.length.toLowerCase()}` : "back"));
        }
      }
    }
    if (!target) {
      if ((canQueue && (!blockedReason || !this.fresh) && status !== "in_room") ||
          (status === "finished" && !["home", "ranked rooms", "ranked modes"].includes(screen.screen))) {
        this.stalledSince ??= now;
        if (now - this.stalledSince >= 15000 &&
            (status !== "finished" || now >= this.postgameDeadline)) {
          throw new Error("No verified control is available on the current screen.");
        }
      } else {
        this.stalledSince = null;
      }
      this.observed = null;
      return null;
    }
    this.stalledSince = null;
    if (!this.observe(target, now)) return null;
    this.acknowledged = false;
    // Never retry a queue submission. A response, timeout or manual restart
    // must resolve it first; the UI may lag behind the server acknowledgement.
    if (target.control.startsWith("4 player ") ||
        target.control === "refresh account" ||
        (screen.screen === "rematch confirmation" && target.control === "confirm")) {
      this.pending = true;
    }
    return target;
  }
}
