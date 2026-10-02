import assert from "node:assert/strict";
import test from "node:test";
import {paymentEstimates} from "./public/shared/outcomes.js";

test("typed payments distinguish ron, pao tsumo and draw transfers", () => {
  const model = {
    structured_outcome: {types: ["ron", "tsumo", "draw", "bank"]},
    terminal_payment_probabilities: Array.from({length: 4}, () =>
      Array.from({length: 4}, () => Array.from({length: 4}, () => [1, 0]))),
  };
  model.terminal_payment_probabilities[2][0][1] = [0, 1];
  let estimates = paymentEstimates(model, [0, 1000]);
  assert.ok(estimates.outcomes.every(row => row.probability === 0));
  model.terminal_payment_probabilities[0][0][1] = [0, 1];
  estimates = paymentEstimates(model, [0, 1000]);
  assert.deepEqual(estimates.outcomes[0], {
    type: "Ron", payer: 0, recipient: 1, probability: 1,
  });
  model.terminal_payment_probabilities[0][0][1] = [1, 0];
  model.terminal_payment_probabilities[1][3][0] = [0, 1];
  estimates = paymentEstimates(model, [0, 1000]);
  assert.deepEqual(estimates.outcomes[0], {
    type: "Tsumo", recipient: 0, probability: 1,
  });
  assert.deepEqual(estimates.std, [0, 0, 0, 0, 0]);
});
