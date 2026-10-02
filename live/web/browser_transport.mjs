import { spawn } from "node:child_process";
import { once } from "node:events";
import { installFrameLimit } from "./browser_rendering.mjs";

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

  async open({chromePath, profilePath, url, ranked, output, unattended = false}) {
    this.sessionId = undefined;
    this.renderingScript = undefined;
    this.attach(spawn(chromePath, [
      "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
      "--enable-webgl",
      ...(process.env.CHROME_HARDWARE_RENDERING === "on" ? [
        "--enable-gpu", "--use-angle=vulkan", "--enable-features=Vulkan",
        "--disable-vulkan-surface", "--disable-software-rasterizer",
      ] : ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"]),
      ...(ranked ? ["--autoplay-policy=no-user-gesture-required"] : []),
      "--ignore-gpu-blocklist",
      "--remote-debugging-pipe", `--user-data-dir=${profilePath}`,
      "--window-size=1280,720", "about:blank",
    ], {stdio: ["ignore", "ignore", "ignore", "pipe", "pipe"]}));
    const {targetId} = await this.send("Target.createTarget", {url: "about:blank"});
    output("startup", {stage: "target_created"});
    ({sessionId: this.sessionId} = await this.send(
      "Target.attachToTarget", {targetId, flatten: true},
    ));
    output("startup", {stage: "target_attached"});
    await this.send("Page.enable");
    output("startup", {stage: "page_enabled"});
    await this.send("Network.enable");
    output("startup", {stage: "network_enabled"});
    if (ranked) {
      await this.send("Runtime.enable");
      await this.send("Log.enable");
    }
    await this.send("Emulation.setDeviceMetricsOverride", {
      width: 1280, height: 720, deviceScaleFactor: 1, mobile: false,
    });
    output("startup", {stage: "metrics_configured"});
    if (ranked) {
      await this.setUnattended(unattended);
    }
    await this.send("Page.navigate", {url});
  }

  async setUnattended(enabled) {
    if (this.renderingScript) {
      await this.send("Page.removeScriptToEvaluateOnNewDocument", {
        identifier: this.renderingScript,
      });
    }
    ({identifier: this.renderingScript} = await this.send(
      "Page.addScriptToEvaluateOnNewDocument", {
        source: `(${installFrameLimit})(${JSON.stringify(enabled)})`,
      },
    ));
    await this.send("Runtime.evaluate", {
      expression: `window.__nejimakidoriUnattended = ${JSON.stringify(enabled)}`,
    });
  }

  send(method, params = {}, sessionId = this.sessionId) {
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

  async stop() {
    if (this.chrome) {
      const exited = once(this.chrome, "exit");
      this.chrome.kill("SIGTERM");
      await exited;
      this.chrome = undefined;
    }
  }

  close() {
    this.chrome?.kill("SIGTERM");
    for (const request of this.pending.values()) clearTimeout(request.timeout);
  }
}
