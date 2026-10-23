// Installed before page scripts; throttle animation callbacks, not game timers.
export function installFrameLimit(unattended) {
  window.__nejimakidoriUnattended = unattended;
  const request = window.requestAnimationFrame.bind(window);
  const cancel = window.cancelAnimationFrame.bind(window);
  const pending = new Map();
  let nextId = 0;
  let lastFrame = -Infinity;
  let checkedFrame = -Infinity;
  let deliver = true;
  window.requestAnimationFrame = callback => {
    const id = ++nextId;
    const frame = timestamp => {
      if (timestamp !== checkedFrame) {
        checkedFrame = timestamp;
        deliver = !window.__nejimakidoriUnattended || timestamp - lastFrame >= 1000 / 15;
        if (deliver) lastFrame = timestamp;
      }
      if (deliver) {
        pending.delete(id);
        callback(timestamp);
      } else {
        pending.set(id, request(frame));
      }
    };
    pending.set(id, request(frame));
    return id;
  };
  window.cancelAnimationFrame = id => {
    cancel(pending.get(id));
    pending.delete(id);
  };
}
