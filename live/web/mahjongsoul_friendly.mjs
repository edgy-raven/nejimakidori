// Dedicated room guests share inference, never browser identity or matchmaking.
export class FriendlySession {
  constructor(env) {
    this.instanceId = env.MAHJONG_SOUL_INSTANCE_ID || "nejimakidori";
    this.friendlyOnly = env.MAHJONG_SOUL_FRIENDLY_ONLY === "on";
    this.expectedAccountId = Number(env.MAHJONG_SOUL_EXPECTED_ACCOUNT_ID);
    this.expectedNickname = env.MAHJONG_SOUL_EXPECTED_NICKNAME;
    this.botAccountIds = new Set(
      (env.MAHJONG_SOUL_BOT_ACCOUNT_IDS || "").split(",").filter(Boolean).map(Number),
    );
    this.riichiOnly = env.MAHJONG_SOUL_RIICHI_ONLY === "on";
    this.roomPlayers = null;
    this.account = null;
    this.ready = false;
    this.cancelVote = null;
    this.voteClickedAt = -Infinity;
    if (this.friendlyOnly) {
      if (this.instanceId === "nejimakidori" ||
          !/^[a-z0-9_-]+$/.test(this.instanceId)) {
        throw new Error("Friendly bots need a distinct instance ID.");
      }
      for (const name of ["PROFILE", "SCREENSHOT", "LIVE_STATE", "CONTROL_URL",
        "CONTROL_PIN", "BUTTON_CAPTURES"]) {
        if (!env[`MAHJONG_SOUL_${name}`]) {
          throw new Error(`Friendly bots require MAHJONG_SOUL_${name}.`);
        }
      }
    }
  }

  updateRoomPlayers(update) {
    if (update.player_list.length) {
      this.roomPlayers = update.player_list.map(player => player.account_id);
    } else if (this.roomPlayers !== null) {
      this.roomPlayers = [...new Set([
        ...this.roomPlayers.filter(accountId => !update.remove_list.includes(accountId)),
        ...update.update_list.map(player => player.account_id),
      ])];
    }
  }

  get roomHasNoHumans() {
    return !this.riichiOnly && this.friendlyOnly && this.botAccountIds.has(this.expectedAccountId) &&
      this.roomPlayers !== null && this.roomPlayers.length > 0 &&
      this.roomPlayers.every(accountId => this.botAccountIds.has(accountId));
  }

  shouldVoteYes(now = Date.now()) {
    return this.friendlyOnly && this.accountVerified && this.cancelVote !== null &&
      now < (this.cancelVote.start_time + this.cancelVote.duration_time) * 1000 &&
      now - this.voteClickedAt >= 5000 &&
      !this.cancelVote.results.some(result => result.account_id === this.expectedAccountId);
  }

  roomAction(status) {
    if (!this.friendlyOnly || !this.accountVerified || !status.roomId) return null;
    if (this.roomHasNoHumans) return "reset";
    if (status.status === "in_room" && !this.ready && this.roomPlayers !== null &&
        (this.riichiOnly || this.roomPlayers.some(accountId => !this.botAccountIds.has(accountId)))) {
      return "ready";
    }
    return null;
  }

  get accountVerified() {
    return Boolean(this.account &&
      this.account.accountId === this.expectedAccountId &&
      this.account.nickname === this.expectedNickname);
  }

  requireIdentity() {
    if (!Number.isInteger(this.expectedAccountId) || this.expectedAccountId < 1 ||
        !this.expectedNickname) {
      throw new Error("Configure the expected account ID and nickname first.");
    }
    if (!this.accountVerified) {
      throw new Error("Official login does not match the expected account.");
    }
  }

  canPlay(friendlyRoom) {
    return !this.friendlyOnly || (friendlyRoom && this.accountVerified);
  }

  async summon(roomId, io) {
    if (!this.friendlyOnly) throw new Error("Summon requires a dedicated friendly bot.");
    if (!Number.isInteger(roomId) || roomId < 1 || roomId > 999999) {
      throw new TypeError("Enter a numeric room ID (1–999999).");
    }
    if (!Number.isInteger(this.expectedAccountId) || this.expectedAccountId < 1 ||
        !this.expectedNickname) {
      throw new Error("Configure the expected account ID and nickname first.");
    }
    if (io.status().status === "hibernating") await io.open();
    for (let attempt = 0; !this.account && attempt < 120; attempt += 1) {
      await io.wait(250);
    }
    this.requireIdentity();
    const status = io.status();
    if (status.status === "in_room" && status.roomId === roomId) {
      if (this.ready) return {roomId, ready: true};
    } else {
      if (!["connected", "finished"].includes(status.status) || status.roomId) {
        throw new Error("Bot is already playing or in another room.");
      }
      await io.home();
      this.requireIdentity();
      await io.join(roomId);
    }
    for (let attempt = 0; attempt < 20; attempt += 1) {
      this.requireIdentity();
      if (io.status().status !== "in_room" || io.status().roomId !== roomId) {
        throw new Error("Room changed before Ready could be verified.");
      }
      const room = await io.room();
      const control = await io.controls();
      if (control.error) {
        await io.stop(control.error.message);
        throw new Error(control.error.message);
      }
      const target = control.targets.find(target => target.control === "ready");
      if (room.in_room && target) {
        this.requireIdentity();
        if (io.status().status !== "in_room" || io.status().roomId !== roomId) {
          throw new Error("Room changed before Ready could be clicked.");
        }
        await io.click(...target.center);
        for (let confirmation = 0; confirmation < 40; confirmation += 1) {
          if (this.ready) return {roomId, ready: true};
          await io.wait(100);
        }
        return {roomId, ready: false, needs: "Ready was not confirmed by Mahjong Soul; inspect the room."};
      }
      await io.wait(250);
    }
    return {roomId, ready: false, needs: "Ready button was not visually verified; inspect the room."};
  }
}
