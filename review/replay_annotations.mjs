// Readiness indices: ready=5, 1 away=4, 2 away=3, 3+ away=2.
// F(0)=0, F(1)=1; combined readiness ranges from 4 through 10.
const FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];

export function assessDecision(payload) {
  const model = payload.predictions[0];
  const checks = [];
  for (const [head, action] of Object.entries(payload.expert_decisions)) {
    let probabilities = model[`${head}_probabilities`];
    let played = probabilities[action];
    if (head === "discard_policy") {
      // Combine identical tile choices; keep red fives distinct.
      const tiles = new Map();
      for (const option of payload.discard_options) {
        tiles.set(option.tile,
          (tiles.get(option.tile) || 0) + probabilities[option.action]);
      }
      played = tiles.get(payload.discard_options.find(
        (option) => option.action === action,
      ).tile);
      probabilities = [...tiles.values()];
    }
    const best = Math.max(...probabilities);
    const total = head === "discard_policy" ? probabilities.length
      : probabilities.filter((value, index) => value > 0 || index === action).length;
    if (total === 1) continue;
    checks.push({head, played, best, gap: best - played, total,
      rank: 1 + probabilities.filter(value => value > played).length});
  }
  if (!checks.length) return null;
  const bestShanten = payload.discard_options.length
    ? Math.min(...payload.discard_options.map(option => option.all_shanten))
    : payload.actual_shanten[payload.actor];
  const readiness = 5 - Math.min(3, Math.max(0, bestShanten));
  const weight = Math.max(
    ...model.opponent_shanten_probabilities.map(probabilities =>
      probabilities.reduce((value, probability, shanten) =>
        value + probability * FIBONACCI[readiness + 5 - shanten], 0)));
  // Hinge on the probability gap: only the excess above 5 points is lost.
  const credit = Math.min(1, ...checks.map(check =>
    (check.played + 0.05) / check.best));
  return {checks, best_shanten: bestShanten, weight, credit,
    penalty: weight * (1 - credit),
    mistake: credit < 1};
}

export async function annotateGame(game, {seat, predict, onProgress, signal}) {
  const annotations = game.rounds.map(round =>
    Array(round.events.length).fill(null));
  const grades = Array.from({length: 4}, () => ({
    checked: 0, graded: 0, matched: 0, mistakes: [], weight: 0, penalty: 0, score: null,
  }));
  const total = game.rounds.reduce((count, round) => count +
    round.events.filter(event =>
      event.seat === seat && ["discard", "call", "kan"].includes(event.type)).length, 0);
  let completed = 0;
  signal.throwIfAborted();
  onProgress({completed, total});
  for (const [roundIndex, round] of game.rounds.entries()) {
    for (const [eventCount, event] of round.events.entries()) {
      if (event.seat !== seat || !["discard", "call", "kan"].includes(event.type)) continue;
      signal.throwIfAborted();
      const payload = await predict(round, eventCount, signal);
      signal.throwIfAborted();
      const review = assessDecision(payload);
      annotations[roundIndex][eventCount] = {...payload, review};
      grades[event.seat].checked += 1;
      if (review !== null) {
        grades[event.seat].graded += 1;
        grades[event.seat].matched += Number(
          review.checks.every(check => check.rank === 1));
        grades[event.seat].penalty += review.penalty;
        grades[event.seat].weight += review.weight;
        if (review.mistake) grades[event.seat].mistakes.push({roundIndex, eventCount});
      }
      onProgress({completed: ++completed, total});
    }
  }
  for (const grade of grades) {
    grade.score = grade.graded
      ? 100 * (1 - grade.penalty / grade.weight) : null;
  }
  return {...game, seat, annotations, grades};
}
