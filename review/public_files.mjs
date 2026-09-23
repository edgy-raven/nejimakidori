import fs from "node:fs";
import path from "node:path";

// Build a closed route map: requests never become filesystem paths.
export function publicFiles(directory) {
  const types = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".ttf": "font/ttf",
    ".woff": "font/woff",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".json": "application/json",
  };
  const files = new Map();
  for (const name of fs.readdirSync(directory, {recursive: true})) {
    const file = path.join(directory, name);
    if (!fs.statSync(file).isFile()) continue;
    const type = types[path.extname(name)];
    if (!type) throw new Error(`Unsupported public asset: ${name}`);
    files.set(`/${name.replace(/index\.html$/, "")}`, [file, type]);
  }
  return files;
}
