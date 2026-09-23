// Chrome's debugging pipe uses NUL-delimited JSON, independently of game state.
export class BrowserTransport {
  constructor(onMessage) {
    this.onMessage = onMessage;
    this.commandId = 0;
    this.incoming = Buffer.alloc(0);
    this.pending = new Map();
  }

  attach(chrome) {
    this.chrome = chrome;
    this.incoming = Buffer.alloc(0);
    chrome.stdio[4].on("data", (chunk) => this.receive(chunk));
  }

  send(method, params = {}, sessionId) {
    const id = ++this.commandId;
    this.chrome.stdio[3].write(
      `${JSON.stringify({id, method, params, sessionId})}\0`,
    );
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Browser command timed out: ${method}`));
      }, method === "Page.navigate" ? 60000 : 10000);
      this.pending.set(id, {resolve, reject, timeout});
    });
  }

  receive(chunk) {
    this.incoming = Buffer.concat([this.incoming, chunk]);
    for (let end; (end = this.incoming.indexOf(0)) !== -1;) {
      const message = JSON.parse(this.incoming.subarray(0, end).toString());
      this.incoming = this.incoming.subarray(end + 1);
      if (!message.id) {
        this.onMessage(message);
        continue;
      }
      const request = this.pending.get(message.id);
      if (!request) continue; // Timed-out responses can arrive late.
      this.pending.delete(message.id);
      clearTimeout(request.timeout);
      if (message.error) {
        request.reject(new Error(JSON.stringify(message.error)));
      } else {
        request.resolve(message.result);
      }
    }
  }

  close() {
    for (const request of this.pending.values()) clearTimeout(request.timeout);
  }
}
