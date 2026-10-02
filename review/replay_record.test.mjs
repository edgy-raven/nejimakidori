import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import {decodeGame} from "./replay_record.mjs";

test("four-riichi settlement retains all deposits in the following round", () => {
  const game = decodeGame("260915-f6a20193-257e-464c-a3ff-6b84d9da084e",
    fs.readFileSync(new URL("./fixtures/suucha-riichi.record", import.meta.url)));
  const roundIndex = game.rounds.findIndex(
    round => round.label === "East 2 · 1 honba",
  );
  const round = game.rounds[roundIndex];
  const result = round.events.at(-1);
  assert.equal(result.label, "Four riichi");
  assert.equal(result.liqi, null);
  assert.deepEqual(result.scores, [25500, 24500, 23800, 21200]);
  assert.deepEqual(result.scores, game.rounds[roundIndex + 1].scores);
  assert.equal(game.rounds[roundIndex + 1].riichiSticks, 5);
  assert.equal(round.scores.reduce((sum, score) => sum + score, 0) +
    1000 * round.riichiSticks - result.scores.reduce((sum, score) => sum + score, 0),
  5000);
  assert.ok(game.rounds.every(round => round.events.at(-1).scores.length === 4));
});
