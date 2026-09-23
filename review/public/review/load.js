export async function loadReview(url, {seat, signal, onProgress}) {
  const response = await fetch("../api/replay", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({url, seat}), signal,
  });
  if (!response.ok) throw new Error(`Log request failed (${response.status}).`);
  return readReview(response.body, onProgress);
}

export async function readReview(body, onProgress) {
  let pending = "";
  let game = null;
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  try {
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      pending += value;
      let end;
      while ((end = pending.indexOf("\n")) !== -1) {
        const message = JSON.parse(pending.slice(0, end));
        pending = pending.slice(end + 1);
        if (message.type === "error") throw new Error(message.message);
        if (message.type === "complete") game = message.game;
        if (message.type === "progress") onProgress(message);
      }
    }
    if (!game || pending) throw new Error("Log annotation did not finish.");
    return game;
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}
