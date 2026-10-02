import assert from "node:assert/strict";
import test from "node:test";
import { Autoqueue, rankedSchedule } from "./mahjongsoul_ui_controls.mjs";

const confirm = {control: "confirm", center: [1160, 660]};
const rematch = {control: "one more match", center: [970, 660]};
const east = {control: "4 player east", center: [930, 260]};
const south = {control: "4 player south", center: [930, 370]};

test("rank progression selects the highest room and enforces its copper minimum", () => {
  const queue = new Autoqueue(() => 0);
  queue.enabled = true;
  for (const [id, name, length, minimum] of [
    [10101, "Bronze", "East", 0], [10201, "Silver", "East", 2500],
    [10301, "Gold", "South", 7000], [10401, "Jade", "South", 14000],
    [10501, "Throne", "South", 14000], [10720, "Throne", "South", 14000],
  ]) {
    queue.updateAccount({rank: {id, score: 0}, copper: minimum});
    assert.equal(queue.snapshot.room.name, name);
    assert.equal(queue.snapshot.room.length, length);
    assert.equal(queue.snapshot.blockedReason, null);
    if (minimum) {
      queue.notification({name: "NotifyAccountUpdate", data: {update: {
        numerical: [{id: 100001, final: 999999}, {id: 100002, final: minimum - 1}],
      }}});
      assert.equal(queue.snapshot.copper, minimum - 1);
      assert.match(queue.snapshot.blockedReason, /Insufficient copper/);
      assert.equal(queue.snapshot.room.name, name); // never downgrade for funds
      for (const now of [10000, 20000]) assert.equal(queue.next("connected",
        [east, south], now, {screen: "ranked modes", room: name}), null);
    }
  }
});

test("postgame exits, refreshes account, then navigates to the promoted room once", () => {
  const queue = new Autoqueue(() => 0);
  queue.enabled = true;
  queue.updateAccount({rank: {id: 10203, score: 700}, copper: 10000});
  queue.notification({name: "NotifyMatchGameStart", data: {}});
  queue.notification({name: "NotifyAccountLevelChange", data: {final: {id: 10301, score: 300}}});
  queue.currentModeId = 5;
  queue.notification({name: "NotifyGameEndResult", data: {}}, -3000);
  assert.equal(queue.next("finished", [rematch, confirm], 0, {screen: "match end"}), null);
  assert.equal(queue.next("finished", [rematch, confirm], 1500, {screen: "match end"}), confirm);
  queue.fresh = false; // a missing session snapshot requires refresh
  const home = {control: "ranked match", center: [930, 220]};
  assert.equal(queue.next("finished", [home], 6500, {screen: "home"}), null);
  assert.equal(queue.next("finished", [home], 8000,
    {screen: "home"}).control, "refresh account");
  assert.equal(queue.next("finished", [home], 14000, {screen: "home"}), null);
  queue.pending = false; // login response
  queue.updateAccount({rank: {id: 10301, score: 300}, copper: 9000});
  assert.equal(queue.next("connected", [home], 15000, {screen: "home"}), null);
  assert.equal(queue.next("connected", [home], 16500, {screen: "home"}), home);
  const gold = {control: "gold room", room: "Gold", center: [930, 493]};
  assert.equal(queue.next("connected", [gold], 22000, {screen: "ranked rooms"}), null);
  assert.equal(queue.next("connected", [gold], 23500, {screen: "ranked rooms"}), gold);
  assert.equal(queue.next("connected", [east, south], 29000,
    {screen: "ranked modes", room: "Gold"}), null);
  assert.equal(queue.next("connected", [east, south], 30500,
    {screen: "ranked modes", room: "Gold"}), south);
  assert.equal(queue.next("connected", [east, south], 40000,
    {screen: "ranked modes", room: "Gold"}), null);
  queue.stop("Insufficient copper reported by the server.");
  assert.equal(queue.snapshot.enabled, false);
  assert.match(queue.snapshot.blockedReason, /Insufficient copper/);
  assert.equal(queue.next("connected", [south], 50000,
    {screen: "ranked modes", room: "Gold"}), null);
});

test("room changes, missing data and balance updates cannot queue the wrong mode", () => {
  const queue = new Autoqueue(() => 0);
  queue.enabled = true;
  assert.match(queue.snapshot.blockedReason, /Waiting/);
  queue.updateAccount({rank: {id: 10401, score: 0}, copper: 14000});
  const bronze = {control: "bronze room", room: "Bronze", center: [930, 271]};
  queue.next("connected", [bronze], 0, {screen: "ranked rooms"});
  assert.equal(queue.next("connected", [bronze], 1500,
    {screen: "ranked rooms"}).deltaY, 360);
  const back = {control: "back", center: [816, 177]};
  queue.next("connected", [east, south, back], 7000, {screen: "ranked modes", room: "Bronze"});
  assert.equal(queue.next("connected", [east, south, back], 8500,
    {screen: "ranked modes", room: "Bronze"}), back);
  queue.next("connected", [south], 14000, {screen: "ranked modes", room: "Jade"});
  queue.notification({name: "NotifyAccountUpdate", data: {update: {
    numerical: [{id: 100002, final: 13999}],
  }}});
  assert.equal(queue.next("connected", [south], 15500,
    {screen: "ranked modes", room: "Jade"}), null);
  queue.notification({name: "NotifyAccountLevelChange", data: {final: {id: 10203, score: 400}}});
  assert.equal(queue.snapshot.room.name, "Silver");
  queue.notification({name: "NotifyAccountLevelChange", data: {final: {id: 20501, score: 0}}});
  assert.equal(queue.snapshot.room.name, "Silver");
  queue.updateAccount(null);
  assert.equal(queue.snapshot.room, null);
  assert.equal(queue.snapshot.copper, null);
});

test("safe result controls require consecutive observations and never retry a stuck click", () => {
  const queue = new Autoqueue(() => 0);
  for (const enabled of [false, true]) {
    queue.enabled = enabled;
    for (const control of ["confirm", "dismiss reward", "next", "continue"]) {
      const target = {control, center: [640, 560]};
      queue.next("playing", [], 0);
      assert.equal(queue.next("finished", [rematch, target], 0, {screen: "chest reward"}), null);
      assert.equal(queue.next("in_room", [rematch, target], 1500, {screen: "chest reward"}), target);
      assert.equal(queue.next("in_room", [target], 3000, {screen: "chest reward"}), null);
      assert.equal(queue.next("in_room", [target], 6500, {screen: "chest reward"}), null);
      assert.equal(queue.next("in_room", [], 8000), null);
      assert.throws(() => queue.next("in_room", [target], 12000, {screen: "chest reward"}), /manual help/);
      assert.throws(() => queue.next("in_room", [target], 13500, {screen: "chest reward"}), /manual help/);
      assert.equal(queue.next("playing", [target], 15000), null);
    }
    assert.equal(queue.next("finished", [rematch], 20000), null);
    assert.equal(queue.next("finished", [rematch], 21500), null);
  }
});


test("affordable matching room rematches; lost funds or a new rank returns to lobby", () => {
  for (const [rank, copper, expected] of [
    [10202, 2500, rematch], [10202, 2499, confirm],
    [10301, 10000, confirm], [10103, 10000, confirm],
  ]) {
    const queue = new Autoqueue(() => 0);
    queue.enabled = true;
    queue.currentModeId = 5;
    queue.finish("game", 0);
    queue.updateAccount({rank: {id: 10202, score: 0}, copper: 9000});
    queue.notification({name: "NotifyGameEndResult", data: {}}, 0);
    assert.equal(queue.next("finished", [rematch, confirm], 1500, {screen: "match end"}), null);
    queue.notification({name: "NotifyAccountUpdate", data: {update: {
      numerical: [{id: 100002, final: copper}],
    }}}, 2000);
    queue.notification({name: "NotifyAccountLevelChange", data: {
      final: {id: rank, score: 400},
    }}, 2000);
    assert.equal(queue.next("finished", [rematch, confirm], 3500, {screen: "match end"}), null);
    assert.equal(queue.next("finished", [rematch, confirm], 5000, {screen: "match end"}), expected);
    if (expected === rematch) {
      assert.equal(queue.snapshot.pending, false);
      queue.next("finished", [confirm], 11000, {screen: "rematch confirmation"});
      assert.equal(queue.next("finished", [confirm], 12500,
        {screen: "rematch confirmation"}), confirm);
      assert.equal(queue.snapshot.pending, true);
    }
  }
});

test("configured queue windows include their start and exclude their end", () => {
  assert.deepEqual(rankedSchedule, {
    timeZone: "Asia/Manila",
    windows: [
      {startHour: 8, endHour: 11},
      {startHour: 12, endHour: 17},
      {startHour: 18, endHour: 23},
    ],
    continuationBase: 0.75,
    breakMinutes: {minimum: 5, range: 10},
  });
  const queue = new Autoqueue();
  queue.enabled = true;
  queue.updateAccount({rank: {id: 10101, score: 0}, copper: 10000});
  for (const [time, open] of [["07:59:59", false], ["08:00:00", true],
    ["10:59:59", true], ["11:00:00", false], ["12:00:00", true],
    ["16:59:59", true], ["17:00:00", false], ["18:00:00", true],
    ["22:59:59", true], ["23:00:00", false], ["00:00:00", false]]) {
    const now = Date.parse(`2026-09-18T${time}+08:00`);
    queue.reset(); queue.pending = false;
    assert.equal(queue.schedule(now).open, open, time);
    queue.next("connected", [east], now, {screen: "ranked modes", room: "Bronze"});
    assert.equal(queue.next("connected", [east], now, {screen: "ranked modes", room: "Bronze"}), open ? east : null, time);
  }
});

test("one decision per game decays continuation odds and pauses 5–15 minutes before a new session", () => {
  for (const durationRoll of [0, 0.5, 0.999999]) {
    const rolls = [0.74, 0.57, durationRoll, 0];
    const queue = new Autoqueue(() => rolls.shift());
    queue.enabled = true; queue.currentModeId = 2;
    queue.updateAccount({rank: {id: 10101, score: 0}, copper: 10000});
    const now = Date.parse("2026-09-18T10:50:00+08:00");
    queue.finish("first", now);
    queue.finish("first", now + 1);
    assert.equal(queue.games, 1);
    queue.next("finished", [rematch, confirm], now, {screen: "match end"});
    assert.equal(queue.next("finished", [rematch, confirm], now + 1500, {screen: "match end"}), rematch);
    queue.pending = false; queue.next("playing", [], now + 2000);
    queue.finish("second", now + 3000);
    assert.equal(queue.schedule(now).continuationChance, 0.5625);
    assert.equal(queue.pauseUntil, now + 3000 + (5 + 10 * durationRoll) * 60000);
    queue.next("finished", [rematch, confirm], now + 3000, {screen: "match end"});
    assert.equal(queue.next("finished", [rematch, confirm], now + 4500, {screen: "match end"}), confirm);
    assert.equal(queue.next("connected", [east], queue.pauseUntil - 1, {screen: "ranked modes", room: "Bronze"}), null);
    const resume = Math.max(queue.pauseUntil, Date.parse("2026-09-18T12:00:00+08:00"));
    queue.next("connected", [east], resume, {screen: "ranked modes", room: "Bronze"});
    assert.equal(queue.next("connected", [east], resume + 1500, {screen: "ranked modes", room: "Bronze"}), east);
    assert.equal(queue.games, 0);
    queue.finish("third", resume + 100000);
    assert.equal(queue.schedule(resume).continuationChance, 0.75);
  }
});

test("ranked clicks only recognized results, quests, chests and star-ups; rank-up and unknown dialogs need help", () => {
  for (const screen of ["match end", "daily quest", "chest reward", "star up"]) {
    const queue = new Autoqueue();
    queue.next("finished", [confirm], 0, {screen});
    assert.equal(queue.next("finished", [confirm], 1500, {screen}), confirm);
  }
  const queue = new Autoqueue();
  assert.throws(() => queue.next("finished", [confirm], 0, {screen: "rank up"}), /Rank-up/);
  const skip = {control: "skip", center: [1200, 30]};
  assert.equal(queue.next("finished", [skip, confirm], 0, {screen: "unknown"}), null);
  assert.throws(() => queue.next("finished", [skip, confirm], 15000, {screen: "unknown"}), /No verified control/);
});

test("known ranked buttons tolerate small OCR movement between observations", () => {
  const queue = new Autoqueue();
  queue.next("finished", [confirm], 0, {screen: "match end"});
  const moved = {...confirm, center: [1165, 657], box: [1140, 645, 50, 25]};
  assert.equal(queue.next("finished", [moved], 1500, {screen: "match end"}), moved);
});

test("postgame confirms hand settlement and each distinct result stage without retrying unchanged screens", () => {
  const queue = new Autoqueue();
  let now = 0;
  for (const screen of [{screen: "hand result"},
    {screen: "match end", stage: "results"},
    {screen: "match end", stage: "rank progress"},
    {screen: "match end", stage: "rematch"}]) {
    assert.equal(queue.next("finished", [confirm], now, screen), null);
    assert.equal(queue.next("finished", [confirm], now + 1500, screen), confirm);
    now += 7000;
  }
  assert.equal(queue.next("finished", [confirm], now,
    {screen: "match end", stage: "rematch"}), null);
  assert.equal(queue.next("finished", [], now + 100000,
    {screen: "result waiting"}), null);
});

test("One More Match opens a dialog; only its confirmation submits matchmaking", () => {
  const queue = new Autoqueue(() => 0);
  queue.enabled = true; queue.currentModeId = 2;
  queue.updateAccount({rank: {id: 10101, score: 0}, copper: 10000});
  queue.finish("finished-game", 0);
  queue.next("finished", [rematch, confirm], 0, {screen: "match end"});
  assert.equal(queue.next("finished", [rematch, confirm], 1500,
    {screen: "match end"}), rematch);
  assert.equal(queue.pending, false);
  const modal = {screen: "rematch confirmation"};
  const yes = {control: "confirm", center: [524, 529]};
  const no = {control: "cancel", center: [756, 529]};
  queue.next("finished", [yes, no], 3000, modal);
  assert.equal(queue.next("finished", [yes, no], 4500, modal), yes);
  assert.equal(queue.pending, true);
  queue.acknowledged = true;
  assert.equal(queue.next("finished", [], 60000), null);
  queue.pending = false; queue.resultStage = null; queue.enabled = false;
  queue.next("finished", [yes, no], 61000, modal);
  assert.equal(queue.next("finished", [yes, no], 62500, modal), no);
});

test("final win animations may outlast 15 seconds; visible Confirm is clicked without waiting out the grace period", () => {
  const queue = new Autoqueue();
  queue.notification({name: "NotifyGameEndResult", data: {}}, 0);
  for (const now of [3000, 19000, 25000]) {
    assert.equal(queue.next("finished", [], now), null);
  }
  const screen = {screen: "match end", stage: "results"};
  assert.equal(queue.next("finished", [confirm], 30000, screen), null);
  assert.equal(queue.next("finished", [confirm], 31500, screen), confirm);
  assert.equal(queue.next("finished", [confirm], 36500, screen), null);
  const stuck = new Autoqueue();
  stuck.notification({name: "NotifyGameEndResult", data: {}}, 0);
  stuck.next("finished", [], 3000);
  assert.throws(() => stuck.next("finished", [], 60000), /No verified control/);
});


test("ranked browser sleeps outside Manila queue hours and through breaks", () => {
  const queue = new Autoqueue();
  assert.equal(queue.shouldIdleBrowser(Date.parse("2026-09-19T07:59:00+08:00")), true);
  assert.equal(queue.shouldIdleBrowser(Date.parse("2026-09-19T08:00:00+08:00")), false);
  assert.equal(queue.shouldIdleBrowser(Date.parse("2026-09-19T11:00:00+08:00")), true);
  queue.pauseUntil = Date.parse("2026-09-19T12:05:00+08:00");
  assert.equal(queue.shouldIdleBrowser(Date.parse("2026-09-19T12:00:00+08:00")), true);
  assert.equal(queue.shouldIdleBrowser(queue.pauseUntil), false);
});
