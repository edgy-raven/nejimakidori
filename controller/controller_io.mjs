import { rename, writeFile } from "node:fs/promises";

// Serialized screenshots and atomic status files shared by both controllers.
export class ControllerIO {
  constructor({send, vision, failureCapture, screenshotPath, liveStatePath,
    output, resetViewport}) {
    this.send = send;
    this.vision = vision;
    this.failureCapture = failureCapture;
    this.screenshotPath = screenshotPath;
    this.liveStatePath = liveStatePath;
    this.output = output;
    this.resetViewport = resetViewport;
    this.capturePending = 0;
    this.captureChain = Promise.resolve();
    this.lastPreviewTimingAt = -Infinity;
    this.stateWrite = Promise.resolve();
  }

  publish(state) {
    const body = JSON.stringify(state);
    const temporary = `${this.liveStatePath}.${process.pid}.tmp`;
    this.stateWrite = this.stateWrite
      .then(() => writeFile(temporary, body))
      .then(() => rename(temporary, this.liveStatePath))
      .catch(error => {
        process.stderr.write(`live state write failed: ${error.message}\n`);
      });
  }

  capture = (mode = "control", position = null) => {
    const queuedAt = performance.now();
    this.capturePending += 1;
    const capture = this.captureChain.then(async () => {
      const started = performance.now();
      const preview = mode === "preview";
      if (this.resetViewport) {
        await this.send("Emulation.setDeviceMetricsOverride", {
          width: 1280, height: 720, deviceScaleFactor: 1, mobile: false,
        });
      }
      const result = await this.send("Page.captureScreenshot", {
        format: preview ? "jpeg" : "png",
        // Scaled clips resize the game's canvas and invalidate click positions.
        ...(preview ? { quality: 60 } : {}),
        captureBeyondViewport: false,
      });
      const capturedAt = performance.now();
      const image = Buffer.from(result.data, "base64");
      this.failureCapture.observe(mode, image);
      const path = preview ? `${this.screenshotPath}.jpg` : this.screenshotPath;
      await writeFile(`${path}.tmp`, image);
      await rename(`${path}.tmp`, path);
      const savedAt = performance.now();
      const detection = preview ? {width: 1280, height: 720}
        : await this.vision.analyze(path, mode, position ? {
          hand: position.hand, buttons: position.buttons,
        } : null);
      this.failureCapture.observe(mode, image, detection);
      const finishedAt = performance.now();
      if (!preview || finishedAt - this.lastPreviewTimingAt >= 30000) {
        this.output("capture_timing", {
          mode, queue_ms: started - queuedAt,
          screenshot_ms: capturedAt - started, save_ms: savedAt - capturedAt,
          vision_ms: finishedAt - savedAt, total_ms: finishedAt - queuedAt,
        });
        if (preview) this.lastPreviewTimingAt = finishedAt;
      }
      return {
        ...detection,
        ...(mode === "control" ? { image } : {}),
      };
    }).finally(() => {
      this.capturePending -= 1;
    });
    // The caller receives failures; the queue must still accept the next capture.
    this.captureChain = capture.catch(() => {});
    return capture;
  };
}
