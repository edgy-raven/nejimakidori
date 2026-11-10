import { MahjongSoulLiveState } from "../mahjongsoul_live_state.mjs";
import {
  decodeGameResponse, decodeNotification,
} from "../mahjongsoul_protocol.mjs";

// Each position owns its request. A later event invalidates even a response
// from a transport that ignores cancellation.
export class GameTracker {
  constructor({predict, publish}) {
    this.predict = predict;
    this.publish = publish;
    this.requests = new Map();
    this.version = 0;
    this.reset();
  }

  reset(status = "waiting") {
    this.pending?.abort();
    this.version += 1;
    this.requests.clear();
    this.state = new MahjongSoulLiveState();
    this.gameSocket = null;
    this.snapshot = {
      status, gameId: null, position: null, prediction: null, error: null,
    };
    this.publish(this.snapshot);
  }

  close() {
    this.reset("disconnected");
  }

  update(position, status = "playing") {
    this.pending?.abort();
    const version = ++this.version;
    this.snapshot = {...this.snapshot, status, position,
      prediction: null, error: null};
    this.publish(this.snapshot);
    if (!position?.phase) return;
    this.pending = new AbortController();
    // Freeze the payload before subsequent events mutate the tracked round.
    const body = structuredClone({
      round: position.round, actor: position.selfSeat,
    });
    this.predict(body, this.pending.signal).then(prediction => {
      if (version !== this.version) return;
      this.snapshot = {...this.snapshot, prediction};
      this.publish(this.snapshot);
    }).catch(error => {
      if (version !== this.version) return;
      this.snapshot = {...this.snapshot, status: "api_error",
        error: error.message};
      this.publish(this.snapshot);
    });
  }

  handle(message) {
    if (message.method === "Network.webSocketClosed") {
      if (message.params.requestId === this.gameSocket) this.close();
      return;
    }
    if (!["Network.webSocketFrameReceived", "Network.webSocketFrameSent"]
      .includes(message.method) || message.params.response.opcode !== 2) return;
    try {
      const response = decodeGameResponse(message.params.response.payloadData,
        this.requests, message.params.requestId);
      if (response?.errorCode) return;
      if (response?.method === "logout") {
        this.reset("logged_off");
        return;
      }
      if (response?.method === "authGame") {
        this.reset("reconnecting");
        if (response.selfSeat < 0) throw new Error("Player has no game seat");
        this.gameSocket = message.params.requestId;
        this.state.selfSeat = response.selfSeat;
        this.state.players = response.players;
        this.state.accountRanks = response.accountRanks;
        this.state.eastOnly = response.eastOnly;
        this.snapshot = {...this.snapshot, gameId: response.gameId};
        this.publish(this.snapshot);
        return;
      }
      if (message.params.requestId !== this.gameSocket) return;
      if (response?.method === "syncGame") {
        const restored = new MahjongSoulLiveState();
        restored.selfSeat = this.state.selfSeat;
        restored.players = this.state.players;
        restored.accountRanks = this.state.accountRanks;
        restored.eastOnly = this.state.eastOnly;
        this.state = restored;
        let position = null;
        for (const action of response.actions) position = this.state.apply(action);
        this.update(position, position?.round ? "playing" : "reconnecting");
        return;
      }
      if (message.method === "Network.webSocketFrameSent") return;
      const action = decodeNotification(message.params.response.payloadData);
      if (!action) return;
      if (["NotifyGameEndResult", "NotifyGameTerminate"].includes(action.name)) {
        this.update(null, "finished");
      } else if (action.name.startsWith("Action")) {
        this.update(this.state.apply(action));
      }
    } catch (error) {
      // A broken history cannot safely supply another prediction. Reconnect
      // and restore the game before accepting more action notifications.
      this.reset("tracking_error");
      this.snapshot = {...this.snapshot, error: error.message};
      this.publish(this.snapshot);
    }
  }
}
