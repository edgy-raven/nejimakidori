// Shared result-page verification. This class cannot select a ranked room.
export class ResultControls {
  observed = null;
  clickedAt = -Infinity;
  clickedControl = null;

  constructor(tolerance = 0) {
    this.tolerance = tolerance;
  }

  reset() {
    this.observed = null;
    this.clickedAt = -Infinity;
    this.clickedControl = null;
  }

  observe(target, now) {
    if (!target) {
      this.observed = null;
      return null;
    }
    if (target.control === this.clickedControl) {
      if (now - this.clickedAt >= 10000) {
        throw new Error(`${target.control} did not advance the screen; manual help required.`);
      }
      return null;
    }
    if (!this.observed || this.observed.control !== target.control ||
        target.center.some((value, axis) =>
          Math.abs(value - this.observed.center[axis]) > this.tolerance)) {
      this.observed = target;
      return null;
    }
    if (now - this.clickedAt < 5000) return null;
    this.clickedAt = now;
    this.clickedControl = target.control;
    return target;
  }

  next(status, targets, now = Date.now()) {
    if (status === "playing") this.reset();
    if (!["finished", "in_room", "connected"].includes(status)) return null;
    return this.observe(targets.find(target => [
      "confirm", "next", "continue", "skip", "dismiss reward",
    ].includes(target.control)), now);
  }
}
