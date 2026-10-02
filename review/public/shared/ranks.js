// AccountLevel IDs and Celestial point scaling follow the game client config.
export function accountRankText(level) {
  if (!level) return "";
  const tier = Math.floor(level.id / 100) % 100;
  const prefix = {1: "N", 2: "A", 3: "E", 4: "M", 5: "S", 7: "C"}[tier];
  if (!prefix) return "";
  const points = tier === 7 ? (Math.floor(level.score / 10) / 10).toFixed(1) : level.score;
  return `${prefix}${level.id % 100}:${points}`;
}
