const {contextBridge, ipcRenderer} = require("electron");
contextBridge.exposeInMainWorld("overlay", {
  ready: () => ipcRenderer.invoke("overlay", "ready"),
  configure: value => ipcRenderer.invoke("overlay", "configure", value),
  start: () => ipcRenderer.invoke("overlay", "start"),
  toggleClickThrough: () => ipcRenderer.invoke("overlay", "click-through"),
  quit: () => ipcRenderer.invoke("overlay", "quit"),
  subscribe: callback => {
    const listener = (_event, snapshot) => callback(snapshot);
    ipcRenderer.on("snapshot", listener);
    return () => ipcRenderer.removeListener("snapshot", listener);
  },
});
