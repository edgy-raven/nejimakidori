import fs from "node:fs";
import path from "node:path";

// Keep the pixels that produced the observation, before later scans replace them.
export class FailureCapture {
  constructor(directory) {
    this.directory = directory;
    this.frames = {};
  }

  observe(mode, image, detection = null) {
    this.frames[mode === "preview" ? "preview" : "control"] = {
      mode, image, detection, at: new Date().toISOString(),
    };
  }

  save(incident, state) {
    const directory = path.join(this.directory, incident.id);
    fs.mkdirSync(directory, {recursive: true, mode: 0o700});
    const frames = {};
    for (const [name, frame] of Object.entries(this.frames)) {
      const filename = `${name}.${name === "preview" ? "jpg" : "png"}`;
      fs.writeFileSync(path.join(directory, filename), frame.image, {mode: 0o600});
      frames[name] = {file: filename, mode: frame.mode, at: frame.at,
        detection: frame.detection};
    }
    fs.writeFileSync(path.join(directory, "incident.json"),
      JSON.stringify({incident, frames, status: state.status,
        position: state.position, prediction: state.prediction}, null, 2) + "\n",
      {mode: 0o600});
    return directory;
  }
}
