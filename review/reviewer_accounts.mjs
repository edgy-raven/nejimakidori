import fs from "node:fs";
import util from "node:util";
import {fetchSessionRecord, hasSession} from "./session_record.mjs";
import {readCache, writeCache} from "./review_cache.mjs";

export class ReviewerAccounts {
  constructor({accounts, stateFile, fetchRecord = fetchSessionRecord, now = Date.now}) {
    this.accounts = accounts;
    this.stateFile = stateFile;
    this.fetchRecord = fetchRecord;
    this.now = now;
    this.queue = Promise.resolve();
    this.state = null;
  }

  async fetch(source) {
    const previous = this.queue;
    let release;
    this.queue = new Promise(resolve => { release = resolve; });
    await previous;
    try {
      this.state ??= await readCache(this.stateFile) || {next: 0, cooldowns: {}};
      for (let offset = 0; offset < this.accounts.length; offset += 1) {
        const index = (this.state.next + offset) % this.accounts.length;
        const account = this.accounts[index];
        if (account.region !== source.region ||
            (this.state.cooldowns[account.name] || 0) > this.now()) continue;
        const credentials = util.parseEnv(fs.readFileSync(account.envFile, "utf8"));
        if (!hasSession(account.region, credentials)) {
          throw new Error(`Reviewer ${account.name} has no configured session.`);
        }
        try {
          const record = await this.fetchRecord(source.recordId, account.region, credentials);
          this.state.next = (index + 1) % this.accounts.length;
          delete this.state.cooldowns[account.name];
          await writeCache(this.stateFile, this.state);
          return record;
        } catch (error) {
          if (Number(error.code ?? error?.error?.code) !== 540 &&
              !/^(session_oauth2(?:Auth|Login)_error_|yostar__user_quick-login_)/.test(error.message || "")) throw error;
          this.state.cooldowns[account.name] = this.now() + 30 * 60 * 1000;
          await writeCache(this.stateFile, this.state);
        }
      }
      throw new Error("Reviewer accounts are cooling down or unavailable. " +
        "Saved games and reviews remain available; retry new logs later.");
    } finally {
      release();
    }
  }
}
