export function livePredictionBody(body) {
  return {
    ...body.decision,
    model_latency_ms: body.inference?.latency_ms,
    terminal_payment_values: body.inference?.metadata.payment_values,
    model: body.inference?.predictions[0] ?? null,
  };
}
