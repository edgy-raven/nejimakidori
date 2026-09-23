import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import {decodeGame} from "./replay_record.mjs";
import {assessDecision, annotateGame} from "./replay_annotations.mjs";
import {penaltyCategory} from "./public/review/penalty.js";
import {readReview} from "./public/review/load.js";

const opponentModel = {opponent_shanten_probabilities:
  Array.from({length: 3}, () => [0, 0, 0, 1])};
const handContext = {actor: 0, discard_options: [], actual_shanten: [3, 0, 0, 0]};

test("the supplied 11-hand log preserves opening draws and repeat payments", () => {
  const game = decodeGame("260913-beaf262a-d315-47fa-b4ee-791d1bd2fc43",
    fs.readFileSync(new URL("./fixtures/review-sample.bin", import.meta.url)));
  assert.equal(game.rounds.length, 11);
  assert.equal(game.rounds.reduce((count, round) => count +
    round.events.filter(event => ["discard", "call", "kan"].includes(event.type)).length, 0), 520);
  for (const round of game.rounds) {
    assert.deepEqual(round.hands.map(hand => hand.length), [13, 13, 13, 13]);
    assert.equal(round.events[0].type, "draw");
    assert.equal(round.events[0].seat, round.kyoku - 1);
    assert.equal(round.events[0].remaining, 69);
    assert.equal(round.events.at(-1).deltas.length, 4);
  }
  assert.equal(game.rounds[5].events[0].tile, "1m");
  assert.equal(game.rounds[5].hands[3].includes("1m"), false);
  assert.equal(game.rounds[5].events[1].tsumogiri, true);
  assert.deepEqual(game.rounds[2].events.at(-1).deltas,
    [-1100, -2100, -1100, 5300]);
});

test("full annotation awards partial credit once per turn and excludes forced moves", async () => {
  const game = {rounds: [{events: [
    {type: "discard", seat: 0}, {type: "draw", seat: 1},
    {type: "discard", seat: 1}, {type: "call", seat: 0},
    {type: "discard", seat: 0}, {type: "kan", seat: 2},
  ]}]};
  const payloads = new Map([
    [0, {actor: 0, expert_decisions: {discard_policy: 0}, discard_options: [
      {all_shanten: 3, action: 0, tile: "1m"}, {all_shanten: 3, action: 1, tile: "2m"},
    ], predictions: [{...opponentModel, discard_policy_probabilities: [0.9, 0.1]}]}],
    [2, {actor: 1, expert_decisions: {discard_policy: 1}, discard_options: [
      {all_shanten: 2, action: 0, tile: "1m"}, {all_shanten: 3, action: 1, tile: "2m"},
    ], predictions: [{...opponentModel, discard_policy_probabilities: [0.6, 0.4]}]}],
    [3, {...handContext, actual_shanten: [1, 0, 0, 0], expert_decisions: {response_action: 1, chi_pattern: 0},
      predictions: [{...opponentModel, response_action_probabilities: [0.5, 0.4, 0.1, 0],
        chi_pattern_probabilities: [0.1, 0.8, 0.1]}]}],
    [4, {actor: 0, expert_decisions: {discard_policy: 0},
      discard_options: [{all_shanten: 3, action: 0, tile: "1m"}],
      predictions: [{...opponentModel, discard_policy_probabilities: [1]}]}],
    [5, {...handContext, actor: 2, actual_shanten: [3, 2, 0, 1], expert_decisions: {kan_action: 1},
      predictions: [{...opponentModel, kan_action_probabilities: [0.8, 0.2, ...Array(33).fill(0)]}]}],
  ]);
  const visits = [];
  const progress = [];
  const result = await annotateGame(game, {seat: 0,
    predict: async (round, eventCount) => {
      visits.push(eventCount);
      assert.equal(game.annotations, undefined);
      return payloads.get(eventCount);
    },
    onProgress: value => progress.push(value),
    signal: new AbortController().signal,
  });
  assert.deepEqual(visits, [0, 3, 4]);
  assert.equal(result.annotations[0][1], null);
  assert.equal(result.annotations[0][4].review, null);
  assert.deepEqual(result.grades.map(grade => grade.score === null
    ? null : Math.round(grade.score * 100) / 100), [40.91, null, null, null]);
  assert.ok(Math.abs(result.annotations[0][3].review.credit - 0.1875) < 1e-12);
  assert.deepEqual(result.grades.map(grade => grade.weight), [11, 0, 0, 0]);
  assert.equal(result.grades[0].penalty, 6.5);
  assert.equal(result.annotations[0][0].review.penalty, 0);
  assert.deepEqual(result.grades[0].mistakes, [{roundIndex: 0, eventCount: 3}]);
  assert.deepEqual(progress.at(-1), {completed: 3, total: 3});

  const controller = new AbortController();
  await assert.rejects(annotateGame(game, {seat: 0,
    predict: async () => { controller.abort(); return {}; },
    onProgress: () => {}, signal: controller.signal,
  }), {name: "AbortError"});
  await assert.rejects(annotateGame(game, {seat: 0,
    predict: async () => { throw new Error("Model unavailable"); },
    onProgress: () => {}, signal: new AbortController().signal,
  }), /Model unavailable/);
  assert.equal(game.annotations, undefined);
});

test("equivalent discards are combined and red fives retain their rank", () => {
  const payload = {...handContext, expert_decisions: {discard_policy: 2}, discard_options: [
    {all_shanten: 3, action: 0, tile: "5m"}, {all_shanten: 3, action: 1, tile: "0m"},
    {all_shanten: 3, action: 2, tile: "5m"},
  ], predictions: [{...opponentModel, discard_policy_probabilities: [0.97, 0.02, 0.01]}]};
  assert.equal(assessDecision(payload).mistake, false);
  assert.equal(assessDecision(payload).checks[0].rank, 1);
  assert.equal(assessDecision(payload).credit, 1);
  payload.expert_decisions.discard_policy = 1;
  assert.deepEqual(assessDecision(payload).checks[0], {head: "discard_policy",
    played: 0.02, best: 0.98, gap: 0.96, total: 2, rank: 2});
  assert.equal(assessDecision({...handContext, expert_decisions: {response_action: 1},
    predictions: [{...opponentModel, response_action_probabilities: [0.48, 0.04, 0.48, 0]}],
  }).checks[0].rank, 3);
  const tied = assessDecision({...handContext, expert_decisions: {riichi_action: 0},
    predictions: [{...opponentModel, riichi_action_probabilities: [0.5, 0.5]}],
  });
  assert.equal(tied.mistake, false);
  assert.equal(tied.credit, 1);
  const second = assessDecision({...handContext, expert_decisions: {response_action: 1},
    predictions: [{...opponentModel, response_action_probabilities: [0.5, 0.4, 0.1, 0]}],
  });
  assert.equal(second.checks[0].rank, 2);
  assert.equal(second.credit, 0.9);
  const distantSecond = assessDecision({...handContext, expert_decisions: {response_action: 1},
    predictions: [{...opponentModel, response_action_probabilities: [0.9, 0.09, 0.01, 0]}],
  });
  assert.equal(distantSecond.checks[0].rank, 2);
  assert.ok(Math.abs(distantSecond.credit - 0.14 / 0.9) < 1e-12);
  assert.equal(assessDecision({...handContext, expert_decisions: {discard_policy: 1},
    discard_options: [{all_shanten: 3, action: 0, tile: "1m"}, {all_shanten: 3, action: 1, tile: "2m"}],
    predictions: [{...opponentModel, discard_policy_probabilities: [1, 0]}],
  }).credit, 0.05);
  assert.equal(assessDecision({...handContext, expert_decisions: {riichi_action: 0},
    predictions: [{...opponentModel, riichi_action_probabilities: [0.02, 0.98]}],
  }).mistake, true);
});

test("the five-point hinge ignores small differences while exact matches and graded counts stay separate", async () => {
  const game = {rounds: [{events: Array.from({length: 5}, () =>
    ({type: "discard", seat: 0}))}]};
  const result = await annotateGame(game, {seat: 0,
    predict: async (round, eventCount) => ({...handContext,
      expert_decisions: {response_action: 1},
      predictions: [{...opponentModel, response_action_probabilities: [
        [0.5, 0.5, 0, 0], [0.5, 0.48, 0.02, 0],
        [0.5, 0.45, 0.05, 0], [0.5, 0.44, 0.06, 0],
        [0.5, 0.4, 0.1, 0],
      ][eventCount]}],
    }),
    onProgress: () => {}, signal: new AbortController().signal,
  });
  assert.deepEqual(result.annotations[0].map(value => value.review.mistake),
    [false, false, false, true, true]);
  assert.deepEqual(result.annotations[0].map(value => value.review.credit),
    [1, 1, 1, 0.98, 0.9]);
  assert.equal(result.grades[0].graded, 5);
  assert.equal(result.grades[0].matched, 1);
  assert.equal(result.grades[0].score, 97.6);
  assert.deepEqual(result.grades[0].mistakes,
    [{roundIndex: 0, eventCount: 3}, {roundIndex: 0, eventCount: 4}]);
  assert.equal(result.grades[1].score, null);
  assert.equal(result.grades[1].matched, 0);
});

test("Fibonacci applies to combined readiness before expectation and opponent maximum", () => {
  for (const [shanten, weight] of [[5, 3], [3, 3], [2, 5], [1, 8], [0, 13]]) {
    const review = assessDecision({...handContext,
      expert_decisions: {discard_policy: 0},
      discard_options: [
        {action: 0, tile: "1m", all_shanten: shanten + 1},
        {action: 1, tile: "2m", all_shanten: shanten},
      ], predictions: [{...opponentModel, discard_policy_probabilities: [0.4, 0.6]}],
    });
    assert.equal(review.weight, weight);
    assert.equal(review.best_shanten, shanten);
    assert.ok(Math.abs(review.credit - 0.75) < 1e-12);
    assert.ok(Math.abs(review.penalty - weight / 4) < 1e-12);
  }
  const payload = {...handContext, actor: 2,
    actual_shanten: [0, 0, 2, 0], expert_decisions: {response_action: 1},
    predictions: [{response_action_probabilities: [0.5, 0.4, 0.1, 0],
      opponent_shanten_probabilities: [
        [0.5, 0.5, 0, 0], [0, 0.25, 0.75, 0], [0, 0, 0, 1],
      ],
    }],
  };
  const review = assessDecision(payload);
  assert.equal(review.weight, 17);
  assert.ok(Math.abs(review.penalty - 1.7) < 1e-12);
  // Opponents' revealed hands never enter the weight, even in a full replay.
  payload.actual_shanten = [3, 3, 2, 3];
  assert.deepEqual(assessDecision(payload), review);
});

test("loading waits for the complete annotation stream and rejects partial reviews", async () => {
  let stream;
  const body = new ReadableStream({start(controller) { stream = controller; }});
  const progress = [];
  let opened = false;
  const loaded = readReview(body, message => progress.push(message)).then(game => {
    opened = true;
    return game;
  });
  stream.enqueue(new TextEncoder().encode('{"type":"progress","message":"Annotating…"}\n'));
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(opened, false);
  assert.deepEqual(progress, [{type: "progress", message: "Annotating…"}]);
  const result = '{"type":"complete","game":{"gameId":"sample"}}\n';
  stream.enqueue(new TextEncoder().encode(result.slice(0, 20)));
  stream.enqueue(new TextEncoder().encode(result.slice(20)));
  stream.close();
  assert.deepEqual(await loaded, {gameId: "sample"});
  for (const [text, message] of [
    ['{"type":"progress","message":"Annotating…"}\n', /did not finish/],
    ['{"type":"error","message":"Model unavailable"}\n', /Model unavailable/],
  ]) {
    await assert.rejects(readReview(new Response(text).body, () => {}), message);
  }
});

test("error categories use strict, unrounded percentage-point thresholds", () => {
  assert.deepEqual([0, 0.0001, 0.25, 0.250001, 1, 1.000001].map(penaltyCategory),
    [null, "inaccuracy", "inaccuracy", "mistake", "mistake", "blunder"]);
});
