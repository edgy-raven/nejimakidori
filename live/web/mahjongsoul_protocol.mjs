import { readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";

import protobuf from "protobufjs";

const root = protobuf.Root.fromJSON(JSON.parse(readFileSync(
  new URL("./mahjongsoul_liqi.json", import.meta.url), "utf8")));
const wrapperType = root.lookupType("Wrapper");
const actionPrototypeType = root.lookupType("ActionPrototype");
const xorKeys = [0x84, 0x5e, 0x4e, 0x42, 0x39, 0xa2, 0x1f, 0x60, 0x1c];

function object(type, data) {
  return type.toObject(type.decode(data), {
    defaults: true,
    enums: String,
    longs: Number,
  });
}

function decryptAction(data) {
  const decoded = Buffer.from(data);
  const base = 23 ^ decoded.length;
  for (let index = 0; index < decoded.length; index += 1) {
    decoded[index] ^=
      (base + 5 * index + xorKeys[index % xorKeys.length]) & 0xff;
  }
  return decoded;
}

export function decodeNotification(frame) {
  const data = Buffer.isBuffer(frame) ? frame : Buffer.from(frame, "base64");
  if (data[0] !== 1) {
    return null;
  }
  const wrapper = wrapperType.decode(data.subarray(1));
  const name = String(wrapper.name).replace(/^\.lq\./, "");
  if (name !== "ActionPrototype") {
    return {
      name,
      data: object(root.lookupType(wrapper.name), wrapper.data),
    };
  }
  const action = actionPrototypeType.decode(wrapper.data);
  return {
    name: String(action.name),
    step: Number(action.step),
    data: object(
      root.lookupType(action.name),
      decryptAction(action.data),
    ),
  };
}

export function decodeGameResponse(frame, requests, connectionId) {
  const data = Buffer.isBuffer(frame) ? frame : Buffer.from(frame, "base64");
  if (data[0] !== 2 && data[0] !== 3) return null;
  const requestId = `${connectionId}:${data.readUInt16LE(1)}`;
  const wrapper = wrapperType.decode(data.subarray(3));
  if (data[0] === 2) {
    if ([".lq.Lobby.joinRoom", ".lq.Lobby.fetchRoom", ".lq.Lobby.leaveRoom", ".lq.FastTest.authGame",
         ".lq.FastTest.syncGame", ".lq.Lobby.login", ".lq.Lobby.emailLogin",
         ".lq.Lobby.oauth2Login", ".lq.Lobby.logout", ".lq.Lobby.readyPlay",
         ".lq.Lobby.fetchAccountInfo", ".lq.Lobby.matchGame", ".lq.Lobby.cancelMatch"].includes(wrapper.name)) {
      const method = wrapper.name.split(".").at(-1);
      requests.set(requestId, {
        method,
        ...(method === "readyPlay" ? {ready: object(
          root.lookupType("ReqRoomReady"), wrapper.data).ready} : {}),
        gameId: method === "authGame"
          ? object(root.lookupType("ReqAuthGame"), wrapper.data).game_uuid : null,
        accountId: method === "authGame"
          ? object(root.lookupType("ReqAuthGame"), wrapper.data).account_id
          : null,
      });
    }
    return null;
  }
  const request = requests.get(requestId);
  if (!request) return null;
  const { method } = request;
  requests.delete(requestId);
  const response = object(root.lookupType({
    joinRoom: "ResJoinRoom", fetchRoom: "ResSelfRoom", authGame: "ResAuthGame", syncGame: "ResSyncGame",
    login: "ResLogin", emailLogin: "ResLogin", oauth2Login: "ResLogin",
    fetchAccountInfo: "ResAccountInfo", matchGame: "ResCommon", cancelMatch: "ResCommon",
    logout: "ResLogout", readyPlay: "ResCommon", leaveRoom: "ResCommon",
  }[method]), wrapper.data);
  if (["login", "emailLogin", "oauth2Login", "logout", "fetchAccountInfo"].includes(method)) return {
    method, errorCode: response.error?.code || 0,
    ...(method !== "logout" && !response.error?.code ? {account: {
      accountId: response.account_id || response.account.account_id,
      nickname: response.account.nickname,
      copper: response.account.gold, rank: response.account.level,
    }} : {}),
  };
  if (["leaveRoom", "matchGame", "cancelMatch"].includes(method)) return {
    method, errorCode: response.error?.code || 0,
  };
  if (method === "readyPlay") return {
    method, ready: request.ready, errorCode: response.error?.code || 0,
  };
  if (method === "authGame") return {
    method, errorCode: response.error?.code || 0,
    gameId: request.gameId,
    roomId: response.game_config?.meta?.room_id || 0,
    modeId: response.game_config?.meta?.mode_id || 0,
    eastOnly: response.game_config?.mode?.mode === 1,
    selfSeat: response.error?.code ? null : response.seat_list.indexOf(request.accountId),
    players: response.seat_list.map((accountId, seat) => ({seat, accountId,
      nickname: response.players.find(player => player.account_id === accountId)?.nickname || "CPU"})),
    accountRanks: response.seat_list.map((accountId) => {
      const player = response.players.find((player) => player.account_id === accountId);
      return accountId && player?.level?.id
        ? {id: player.level.id, score: player.level.score} : null;
    }),
  };
  if (method === "syncGame") return {
    method, errorCode: response.error?.code || 0,
    actions: (response.game_restore?.actions || []).map((action) => ({
      name: action.name, step: action.step,
      data: object(root.lookupType(action.name), action.data),
    })),
  };
  return {
    method, roomId: response.room?.room_id,
    accountIds: response.room?.persons.map(player => player.account_id) || [],
    errorCode: response.error?.code || 0,
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  for await (const line of createInterface({ input: process.stdin })) {
    const decoded = decodeNotification(line.trim());
    if (decoded) {
      process.stdout.write(`${JSON.stringify(decoded)}\n`);
    }
  }
}
