export function paymentEstimates(model, values) {
  const payments = model.terminal_payment_probabilities.map((channel) =>
    channel.map((row) => row.map((probabilities) => {
      const mean = probabilities.reduce((sum, probability, index) =>
        sum + probability * values[index], 0);
      return {
        variance: probabilities.reduce((sum, probability, index) =>
          sum + probability * (values[index] - mean) ** 2, 0),
        positive: probabilities.reduce((sum, probability, index) =>
          sum + (values[index] > 0 ? probability : 0), 0),
      };
    })));
  const ron = payments[model.structured_outcome.types.indexOf("ron")];
  const tsumo = payments[model.structured_outcome.types.indexOf("tsumo")];
  return {
    outcomes: [0, 1, 2, 3].flatMap((recipient) => {
      const payers = [0, 1, 2, 3].filter((seat) => seat !== recipient);
      return [
        {type: "Tsumo", recipient, probability: 1 - payers.reduce((product, payer) =>
          product * (1 - tsumo[payer][recipient].positive), 1)},
        ...payers.map((payer) => ({type: "Ron", payer, recipient,
          probability: ron[payer][recipient].positive,
        })),
      ];
    }).sort((a, b) => b.probability - a.probability).slice(0, 4),
    std: [...[0, 1, 2, 3].map((seat) => Math.sqrt(
      payments.reduce((total, channel) => total +
        channel[seat].reduce((sum, payment) => sum + payment.variance, 0) +
        channel.reduce((sum, payer, index) =>
          sum + (index === seat ? 0 : payer[seat].variance), 0), 0)
    )), Math.sqrt(payments.reduce((total, channel) => total +
      channel.reduce((sum, row, seat) => sum + row[seat].variance, 0), 0))],
  };
}
