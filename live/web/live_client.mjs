import { controlPin, liveControlUrl } from "./live_control.mjs";
let body;
if (process.argv[2] !== "status") {
  body = "";
  for await (const chunk of process.stdin) body += chunk;
  body = JSON.stringify(JSON.parse(body));
}
const response = await fetch(`${liveControlUrl}/${body ? "command" : "status"}`, {
  method: body ? "POST" : "GET",
  headers: {
    "Content-Type": "application/json", Authorization: `Bearer ${controlPin}`,
  },
  body,
  signal: AbortSignal.timeout(30000),
});
console.log(JSON.stringify(await response.json()));
if (!response.ok) process.exitCode = 1;
