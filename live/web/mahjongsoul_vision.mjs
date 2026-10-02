import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

// Requests are serialized by the controller's screenshot queue.
export class VisionWorker {
  constructor(pythonPath, visionPath) {
    this.process = spawn(pythonPath, [visionPath, "--serve"], {
      stdio: ["pipe", "pipe", "inherit"],
    });
    this.lines = createInterface({ input: this.process.stdout })
      [Symbol.asyncIterator]();
  }

  async analyze(image, mode, context = null) {
    this.process.stdin.write(`${JSON.stringify({ image, mode, context })}\n`);
    let timeout;
    try {
      const response = await Promise.race([
        this.lines.next(),
        new Promise((resolve, reject) => {
          timeout = setTimeout(() => {
            this.close();
            reject(new Error("Vision worker timed out"));
          }, 10000);
        }),
      ]);
      if (response.done) throw new Error("Vision worker exited without a result");
      return JSON.parse(response.value);
    } finally {
      clearTimeout(timeout);
    }
  }

  close() {
    this.process.kill("SIGTERM");
  }
}
