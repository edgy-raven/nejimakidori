import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {FailureCapture} from "./failure_capture.mjs";

test("failure pixels and observations survive later scans and repeated incidents", t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "failure-capture-"));
  t.after(() => fs.rmSync(root, {recursive: true}));
  const captures = new FailureCapture(root);
  captures.observe("ranked", Buffer.from("failed pixels"), {screen: "unknown"});
  captures.observe("preview", Buffer.from("preview pixels"));
  const saved = captures.save({id: "first", reason: "Unknown screen"},
    {status: "finished", position: null});
  captures.observe("ranked", Buffer.from("later pixels"), {screen: "home"});
  captures.save({id: "second"}, {status: "connected"});
  assert.equal(fs.readFileSync(path.join(saved, "control.png"), "utf8"), "failed pixels");
  const incident = JSON.parse(fs.readFileSync(path.join(saved, "incident.json")));
  assert.equal(incident.frames.control.detection.screen, "unknown");
  assert.equal(incident.frames.preview.file, "preview.jpg");
  assert.equal(fs.statSync(path.join(saved, "incident.json")).mode & 0o777, 0o600);
});
