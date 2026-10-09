import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import { installFrameLimit } from "./browser_rendering.mjs";

test("unattended frames are capped, cancellable, and restored for watching", () => {
  const queued = new Map();
  let next = 0;
  const window = {
    requestAnimationFrame(callback) {queued.set(++next, callback); return next;},
    cancelAnimationFrame(id) {queued.delete(id);},
  };
  vm.runInNewContext(`(${installFrameLimit})(true)`, {window});
  const tick = timestamp => {
    for (const [id, callback] of [...queued]) {
      queued.delete(id);
      callback(timestamp);
    }
  };
  const frames = [];
  const animate = timestamp => {
    frames.push(timestamp);
    window.requestAnimationFrame(animate);
  };
  window.requestAnimationFrame(animate);
  tick(0);
  let cancelled = false;
  const id = window.requestAnimationFrame(() => {cancelled = true;});
  tick(20); // Cancellation must still work after the native request reschedules.
  window.cancelAnimationFrame(id);
  tick(40);
  tick(60);
  tick(80);
  assert.deepEqual(frames, [0, 80]);
  assert.equal(cancelled, false);
  window.__nejimakidoriUnattended = false;
  tick(100);
  tick(120);
  assert.deepEqual(frames, [0, 80, 100, 120]);
  window.__nejimakidoriUnattended = true;
  tick(140);
  tick(200);
  assert.deepEqual(frames, [0, 80, 100, 120, 200]);
});
