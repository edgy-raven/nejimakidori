import crypto from "node:crypto";

import MJSoul from "mjsoul";
import protobuf from "protobufjs";

import { refreshYostarSession } from "./yostar.mjs";

const gateways = {
  cn: "wss://gateway-cdn.maj-soul.com/gateway",
  en: "wss://engs.mahjongsoul.com:443/gateway",
  jp: "wss://jpgs.mahjongsoul.com:443/gateway",
};
const webOrigins = {
  cn: "https://game.maj-soul.com/1",
  en: "https://mahjongsoul.game.yo-star.com",
  jp: "https://game.mahjongsoul.com",
};
const maximumRecordBytes = 8 * 1024 * 1024;
const routeNames = { cn: "route-2", en: "en-1", jp: "jp-1" };
const packageVersions = { cn: "4.0.47", en: "4.0.10", jp: "4.0.12" };

function accessToken(region, credentials) {
  return credentials[`MAJSOUL_${region.toUpperCase()}_ACCESS_TOKEN`] || "";
}

function yostarSession(region, credentials) {
  const prefix = `MAJSOUL_${region.toUpperCase()}_YOSTAR_`;
  const uid = credentials[`${prefix}UID`] || "";
  const token = credentials[`${prefix}TOKEN`] || "";
  const deviceId = credentials[`${prefix}DEVICE_ID`] || "";
  return uid && token && deviceId ? { uid, token, deviceId } : null;
}

function oauthType(region, credentials) {
  const value = credentials[`MAJSOUL_${region.toUpperCase()}_OAUTH2_TYPE`];
  return value ? Number(value) : region === "cn" ? 7 : 22;
}

function clientDevice() {
  return {
    platform: "pc",
    hardware: "pc",
    os: "windows",
    os_version: "Windows 10",
    is_browser: true,
    software: "Chrome",
    sale_platform: "web",
    hardware_vendor: "",
    model_number: "",
    screen_width: 1920,
    screen_height: 1080,
    user_agent:
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " +
      "AppleWebKit/537.36 (KHTML, like Gecko) " +
      "Chrome/151.0.0.0 Safari/537.36",
    screen_type: 1,
  };
}

function addField(type, name, id, fieldType) {
  if (!type.fields[name]) {
    type.add(new protobuf.Field(name, id, fieldType));
  }
}

function patchProtocol(client) {
  const lq = client.root.lookup("lq");
  if (!lq.get("ReqRequestConnection")) {
    lq.add(
      new protobuf.Type("ReqRequestConnection")
        .add(new protobuf.Field("type", 2, "uint32"))
        .add(new protobuf.Field("route", 3, "string"))
        .add(new protobuf.Field("timestamp", 4, "uint32"))
        .add(new protobuf.Field("platform", 6, "string")),
    );
    lq.add(
      new protobuf.Service("Route").add(
        new protobuf.Method(
          "requestConnection",
          "rpc",
          "ReqRequestConnection",
          "ResCommon",
        ),
      ),
    );
  }
  const lobby = lq.get("Lobby");
  if (!lobby.methods.addRoomRobot) {
    lq.add(
      new protobuf.Type("ReqAddRoomRobot").add(
        new protobuf.Field("position", 1, "uint32"),
      ),
    );
    lobby.add(
      new protobuf.Method(
        "addRoomRobot",
        "rpc",
        "ReqAddRoomRobot",
        "ResCommon",
      ),
    );
  }
  const device = client.root.lookupType("ClientDeviceInfo");
  addField(device, "screen_width", 10, "uint32");
  addField(device, "screen_height", 11, "uint32");
  addField(device, "user_agent", 12, "string");
  addField(device, "screen_type", 13, "uint32");
  addField(device, "device_id", 14, "string");
  addField(client.root.lookupType("ReqOauth2Login"), "lang", 11, "string");
}

async function clientVersion(region, credentials) {
  const configured = credentials[
    `MAJSOUL_${region.toUpperCase()}_CLIENT_VERSION`
  ];
  if (configured) {
    return {
      resource: configured.replace(/^WebGL_2022-/, ""),
      string: configured,
      package: packageVersions[region],
    };
  }
  if (region === "en" || region === "jp") {
    return {
      resource: "0.16.236",
      string: "WebGL_2022-0.16.236",
      package: packageVersions[region],
    };
  }
  const response = await fetch(`${webOrigins[region]}/version.json`, {
    cache: "no-store",
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) {
    throw new Error(`session_version_http_${response.status}`);
  }
  const value = String((await response.json()).version || "").replace(
    /\.w$/,
    "",
  );
  if (!value) {
    throw new Error("session_version_missing");
  }
  return {
    resource: value,
    string: `web-${value}`,
    package: packageVersions[region],
  };
}

function openClient(client) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      client.close();
      reject(new Error("session_gateway_timeout"));
    }, 10000);
    const onError = (error) => {
      clearTimeout(timer);
      reject(error);
    };
    client.once("error", onError);
    client.open(() => {
      clearTimeout(timer);
      client.off("error", onError);
      resolve();
    });
  });
}

function upstreamCode(error) {
  return Number(error?.error?.code);
}

async function send(client, name, data) {
  try {
    return await client.sendAsync(name, data);
  } catch (error) {
    const failure = new Error(`session_${name}_error_${upstreamCode(error) || "unknown"}`);
    failure.code = upstreamCode(error);
    throw failure;
  }
}

async function fetchAuthorizedRecord(client, recordId, version) {
  const request = {
    game_uuid: recordId,
    client_version_string: version.string,
  };
  try {
    return await client.sendAsync("fetchGameRecord", request);
  } catch (error) {
    if (upstreamCode(error) !== 151) {
      throw error;
    }
    await send(client, "readGameRecord", request);
    return send(client, "fetchGameRecord", request);
  }
}

async function downloadDataUrl(value) {
  const url = new URL(value);
  if (url.protocol !== "https:") {
    throw new Error("session_record_url_not_https");
  }
  const response = await fetch(url, { signal: AbortSignal.timeout(20000) });
  if (!response.ok) {
    throw new Error(`session_record_http_${response.status}`);
  }
  const data = Buffer.from(await response.arrayBuffer());
  if (!data.length || data.length > maximumRecordBytes) {
    throw new Error(data.length ? "record_too_large" : "record_not_found");
  }
  return data;
}

export function hasSession(region, credentials = process.env) {
  return Boolean(accessToken(region, credentials) || yostarSession(region, credentials));
}

export function configuredRegions() {
  return Object.keys(gateways).filter(region => hasSession(region));
}

async function openSessionClient(region, credentials = process.env) {
  const version = await clientVersion(region, credentials);
  const client = new MJSoul({
    url: process.env[`MAJSOUL_${region.toUpperCase()}_GATEWAY_URL`] ||
      gateways[region],
    timeout: 15000,
  });
  patchProtocol(client);
  try {
    await openClient(client);
    client.service = ".lq.Route.";
    await send(client, "requestConnection", {
      type: 1,
      route: process.env[`MAJSOUL_${region.toUpperCase()}_ROUTE_NAME`] ||
        routeNames[region],
      timestamp: Math.floor(Date.now() / 1000),
      platform: "Web",
    });
    client.service = ".lq.Lobby.";
    let token = accessToken(region, credentials);
    const yostar = yostarSession(region, credentials);
    if (!token && yostar) {
      const code = await refreshYostarSession(
        yostar.uid,
        yostar.token,
        yostar.deviceId,
      );
      const auth = await send(client, "oauth2Auth", {
        type: oauthType(region, credentials),
        code,
        uid: yostar.uid,
        client_version_string: version.string,
      });
      token = String(auth.access_token || "");
    }
    if (!token) {
      throw new Error("session_access_token_missing");
    }
    const device = clientDevice();
    device.device_id = yostar?.deviceId || crypto.randomUUID();
    await send(client, "oauth2Login", {
      type: oauthType(region, credentials),
      access_token: token,
      reconnect: true,
      device,
      random_key: device.device_id,
      client_version: {
        resource: version.resource,
        package: version.package,
      },
      client_version_string: version.string,
      currency_platforms: region === "cn"
        ? [1, 2, 5, 6, 8, 10, 11]
        : [1, 4, 5, 9, 12],
      version: 0,
      lang: region,
    });
    return { client, version };
  } catch (error) {
    client.close();
    throw error;
  }
}

export async function fetchSessionRecord(recordId, region, credentials) {
  const { client, version } = await openSessionClient(region, credentials);
  try {
    const response = await fetchAuthorizedRecord(
      client, recordId, version);
    const players = response.head.accounts.map(player => ({
      accountId: player.account_id, seat: player.seat, nickname: player.nickname,
    }));
    if (response.data?.length) {
      return {data: Buffer.from(response.data), players};
    }
    if (response.data_url) {
      return {data: await downloadDataUrl(response.data_url), players};
    }
    throw new Error("session_record_response_empty");
  } finally {
    client.close();
  }
}
