import assert from "node:assert/strict";
import test from "node:test";
import { accountRankText } from "./public/shared/ranks.js";

test("account ranks use compact tiers and the client's point units", () => {
  for (const [id, score, expected] of [
    [10101, 0, "N1:0"], [10202, 500, "A2:500"],
    [10303, 1900, "E3:1900"], [10403, 2300, "M3:2300"],
    [10501, 3500, "S1:3500"], [10712, 1234, "C12:12.3"],
  ]) assert.equal(accountRankText({id, score}), expected);
  assert.equal(accountRankText(null), "");
  assert.equal(accountRankText(undefined), "");
  assert.equal(accountRankText({id: 0, score: 0}), "");
});
