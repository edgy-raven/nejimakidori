import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline/promises";
import { fileURLToPath } from "node:url";

import { loginWithEmailCode, sendEmailCode } from "./yostar.mjs";

const projectPath = path.dirname(fileURLToPath(import.meta.url));
const prompt = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

try {
  const email = (await prompt.question("Yostar account email: ")).trim();
  if (!email.includes("@")) {
    throw new Error("invalid_email");
  }
  const deviceId = crypto.randomUUID();
  await sendEmailCode(email, deviceId);
  console.log("Yostar sent a verification code to that address.");
  const code = (await prompt.question("Verification code: ")).trim();
  if (!/^\d{6}$/.test(code)) {
    throw new Error("verification_code_must_be_six_digits");
  }
  const session = await loginWithEmailCode(email, code, deviceId);
  fs.writeFileSync(
    path.join(projectPath, ".env.local"),
    [
      "# Local Yostar session; ignored by git.",
      `MAJSOUL_EN_YOSTAR_UID=${session.uid}`,
      `MAJSOUL_EN_YOSTAR_TOKEN=${session.token}`,
      `MAJSOUL_EN_YOSTAR_DEVICE_ID=${deviceId}`,
      "MAJSOUL_EN_OAUTH2_TYPE=22",
      "MAJSOUL_EN_CLIENT_VERSION=WebGL_2022-0.16.236",
      "",
    ].join("\n"),
    { mode: 0o600 },
  );
  fs.chmodSync(path.join(projectPath, ".env.local"), 0o600);
  console.log("Saved the local session to .env.local.");
  console.log("Restart the app tmux pane; keep the tunnel running.");
} finally {
  prompt.close();
}
