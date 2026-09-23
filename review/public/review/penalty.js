export function penaltyCategory(points) {
  return points > 1 ? "blunder" : points > 0.25 ? "mistake"
    : points > 0 ? "inaccuracy" : null;
}
