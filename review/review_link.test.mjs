import assert from "node:assert/strict";
import test from "node:test";
import {parseReviewLink, seatFromLink} from "./public/review/link.js";

test("shared log text identifies the linked player by recorded seat, not list order", () => {
  const link = parseReviewLink("Mahjong Soul Game Log:" +
    "https://mahjongsoul.game.yo-star.com/?paipu=" +
    "260915-049268c8-1bd7-49f7-8b1c-a53234d336f6_a829099891");
  const players = [
    {accountId: 125574578, seat: 2}, {accountId: 117490313, seat: 3},
    {accountId: 119585345, seat: 0}, {accountId: 119601951, seat: 1},
  ];
  assert.equal(link.recordId, "260915-049268c8-1bd7-49f7-8b1c-a53234d336f6");
  assert.equal(link.region, "en");
  assert.equal(parseReviewLink(link.url.replace(link.recordId, link.recordId.toUpperCase())).recordId, link.recordId);
  assert.ok(link.url.startsWith("https://"));
  assert.equal(seatFromLink(link, players), 2);
  assert.deepEqual(parseReviewLink(`  ${link.url}\n`), link);
  const otherPlayer = parseReviewLink(link.url.replace("829099891", "887693680"));
  assert.equal(otherPlayer.recordId, link.recordId);
  assert.equal(seatFromLink(otherPlayer, players), 1);
  assert.equal(seatFromLink(parseReviewLink(link.url.replace("829099891", "887777150")), players), 0);
  assert.equal(seatFromLink(parseReviewLink(link.url.replace("_a829099891", "")), players), null);
  assert.equal(seatFromLink(link, []), null);
  assert.equal(seatFromLink(parseReviewLink(link.url.replace("829099891", "99999999999999999999")), players), null);
});

test("link parsing accepts supported regions and rejects unrelated or malformed input", () => {
  for (const [host, region] of [["game.mahjongsoul.com", "jp"], ["game.maj-soul.com", "cn"]]) {
    assert.equal(parseReviewLink(`Game Log: https://${host}/?paipu=260915-049268c8-1bd7-49f7-8b1c-a53234d336f6`).region, region);
  }
  for (const input of ["", "https://", "https://example.com/?paipu=260915-049268c8-1bd7-49f7-8b1c-a53234d336f6",
    "https://mahjongsoul.game.yo-star.com/?paipu=bad-id"]) {
    assert.throws(() => parseReviewLink(input), /Mahjong Soul replay link/);
  }
});
