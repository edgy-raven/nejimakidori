import crypto from "node:crypto";

const baseUrl = "https://en-sdk-api.yostarplat.com";
const signingSalt = "347467131a466f6865d7f2662e38841fbe2adb23";

function authorization(body, deviceId, uid, token) {
  const head = {
    Region: "US",
    PID: "US-MAJONGSOUL",
    Channel: "web",
    Platform: "pc",
    Version: "4.16.2",
    Lang: "en",
    DeviceID: deviceId,
  };
  if (uid) {
    head.UID = uid;
  }
  if (token) {
    head.Token = token;
  }
  head.Time = Math.floor(Date.now() / 1000);
  const text = JSON.stringify(body);
  return JSON.stringify({
    Head: head,
    Sign: crypto
      .createHash("md5")
      .update(`${JSON.stringify(head)}${text}${signingSalt}`)
      .digest("hex")
      .toUpperCase(),
  });
}

async function post(path, body, deviceId, uid, token) {
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: {
      Accept: "application/json, text/plain, */*",
      "Content-Type": "application/json",
      Authorization: authorization(body, deviceId, uid, token),
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(15000),
  });
  const payload = await response.json();
  if (!response.ok || payload.Code !== 200) {
    throw new Error(`yostar_${path.replaceAll("/", "_")}_${payload.Code}`);
  }
  return payload.Data || {};
}

export async function sendEmailCode(email, deviceId) {
  await post("/yostar/send-code", { Account: email }, deviceId);
}

export async function loginWithEmailCode(email, code, deviceId) {
  const auth = await post(
    "/yostar/get-auth",
    { Account: email, Code: code },
    deviceId,
  );
  if (!auth.Token) {
    throw new Error("yostar_email_code_returned_no_token");
  }
  const login = await post(
    "/user/login",
    {
      Type: "yostar",
      OpenID: email,
      Token: auth.Token,
      Secret: "",
      CheckAccountPlus: 0,
    },
    deviceId,
  );
  if (!login.UserInfo?.ID || !login.UserInfo?.Token) {
    throw new Error("yostar_login_returned_no_session");
  }
  return {
    uid: String(login.UserInfo.ID),
    token: String(login.UserInfo.Token),
  };
}

export async function refreshYostarSession(uid, token, deviceId) {
  const login = await post("/user/quick-login", {}, deviceId, uid, token);
  if (!login.UserInfo?.Token) {
    throw new Error("yostar_quick_login_returned_no_token");
  }
  return String(login.UserInfo.Token);
}
