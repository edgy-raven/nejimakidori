const regions = new Map([
  ["mahjongsoul.game.yo-star.com", "en"],
  ["game.mahjongsoul.com", "jp"],
  ["game.maj-soul.com", "cn"],
]);

export function parseReviewLink(value) {
  const match = String(value ?? "").match(/https:\/\/[^\s<>"']+/i);
  if (!match || !URL.canParse(match[0])) {
    throw new Error("Enter a Mahjong Soul replay link or shared log text.");
  }
  const url = new URL(match[0]);
  const record = url.searchParams.get("paipu")?.match(
    /^((?:\d{6}-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:_a(\d+))?$/i,
  );
  if (!regions.has(url.hostname) || !record) {
    throw new Error("Enter a Mahjong Soul replay link or shared log text.");
  }
  return {url: url.toString(), recordId: record[1].toLowerCase(), archiveId: record[0].toLowerCase(),
    linkId: record[2] ?? null, region: regions.get(url.hostname)};
}

export function seatFromLink(link, players) {
  // The share ID is encoded, rather than the account ID from the record.
  // https://github.com/SAPikachu/amae-koromo/blob/master/src/data/types/record.ts
  return players.find(player => String(
    1358437 + ((7 * player.accountId + 1117113) ^ 86216345),
  ) === link.linkId)?.seat ?? null;
}
