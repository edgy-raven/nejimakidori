import {createRequire} from "node:module";
import protobuf from "protobufjs";
import {terminalEvent} from "../live/web/mahjongsoul_terminal.mjs";
import {playbackState} from "./public/shared/replay.js";
import {tileBase} from "./public/shared/tile_values.js";

const require = createRequire(import.meta.url);
const root = protobuf.Root.fromJSON(require("mjsoul/liqi.json"));
const wrapperType = root.lookupType("Wrapper");
const gameDetailType = root.lookupType("GameDetailRecords");

function decodeWrapped(data) {
  const wrapper = wrapperType.decode(data);
  if (!wrapper.name) {
    return null;
  }
  const messageType = root.lookupType(wrapper.name);
  return {
    name: wrapper.name.replace(/^\.lq\./, ""),
    data: messageType.toObject(messageType.decode(wrapper.data), {
      defaults: true,
      enums: String,
      longs: Number,
    }),
  };
}

function sameFirstRecord(left, right) {
  return Boolean(
    left?.name === right?.name &&
      JSON.stringify(left?.data) === JSON.stringify(right?.data),
  );
}

function decodeRecord(data) {
  const outer = wrapperType.decode(data);
  if (!outer.data?.length) {
    throw new Error("missing_game_detail_records");
  }
  const detail = gameDetailType.decode(outer.data);
  const oldRecords = (detail.records || []).map(decodeWrapped).filter(Boolean);
  const actionRecords = (detail.actions || [])
    .filter((action) => action.type === 1 && action.result?.length)
    .map((action) => decodeWrapped(action.result))
    .filter(Boolean);
  if (!oldRecords.length) {
    return actionRecords;
  }
  if (!actionRecords.length) {
    return oldRecords;
  }
  if (
    actionRecords.length > oldRecords.length &&
    sameFirstRecord(actionRecords[0], oldRecords[0])
  ) {
    return actionRecords;
  }
  return actionRecords.length > oldRecords.length
    ? [...oldRecords, ...actionRecords]
    : oldRecords;
}

function strings(value) {
  return Array.isArray(value) ? value.map(String) : [];
}

function numbers(value) {
  return Array.isArray(value) ? value.map(Number) : [];
}

function doraIndicators(data) {
  const doras = strings(data.doras);
  return doras.length ? doras : data.dora ? [String(data.dora)] : [];
}

function acceptedRiichi(data) {
  if (!data.liqi || typeof data.liqi !== "object") {
    return null;
  }
  return {
    seat: Number(data.liqi.seat),
    score: Number(data.liqi.score),
    sticks: Number(data.liqi.liqibang),
  };
}

function normalizeEvent(record) {
  const name = record.name.replace(/^(Record|Action)/, "");
  const data = record.data;
  if (name === "DealTile") {
    return {
      type: "draw",
      seat: Number(data.seat),
      tile: String(data.tile || ""),
      remaining: Number(data.left_tile_count),
      doras: doraIndicators(data),
      liqi: acceptedRiichi(data),
    };
  }
  if (name === "DiscardTile") {
    return {
      type: "discard",
      seat: Number(data.seat),
      tile: String(data.tile || ""),
      tsumogiri: Boolean(data.moqie),
      riichi: Boolean(data.is_liqi || data.is_wliqi),
      doras: doraIndicators(data),
    };
  }
  if (name === "ChiPengGang") {
    const callTypes = ["chii", "pon", "daiminkan"];
    return {
      type: "call",
      seat: Number(data.seat),
      callType: callTypes[Number(data.type)] || "call",
      tiles: strings(data.tiles),
      froms: numbers(data.froms),
      liqi: acceptedRiichi(data),
    };
  }
  if (name === "AnGangAddGang") {
    const callType = Number(data.type) === 3 ? "ankan" : "kakan";
    const tile = String(data.tiles || "");
    return {
      type: "kan",
      seat: Number(data.seat),
      callType,
      tiles: callType === "ankan" ? [tile, tile, tile, tile] : [tile],
      doras: doraIndicators(data),
    };
  }
  return terminalEvent(name, data);
}

function roundLabel(data) {
  const winds = ["East", "South", "West", "North"];
  return `${winds[Number(data.chang)] || "Round"} ${Number(data.ju) + 1}` +
    (Number(data.ben) ? ` · ${Number(data.ben)} honba` : "");
}

export function decodeGame(recordId, bytes) {
  const rounds = [];
  let round = null;
  for (const record of decodeRecord(bytes)) {
    if (record.name.replace(/^(Record|Action)/, "") === "NewRound") {
      const data = record.data;
      const hands = [0, 1, 2, 3].map((seat) =>
        strings(data[`tiles${seat}`]),
      );
      if (hands.some((hand, seat) =>
        hand.length !== (seat === Number(data.ju) ? 14 : 13))) {
        throw new Error("record_does_not_contain_four_complete_hands");
      }
      round = {
        label: roundLabel(data),
        prevailingWind: Number(data.chang),
        kyoku: Number(data.ju) + 1,
        honba: Number(data.ben),
        riichiSticks: Number(data.liqibang),
        scores: numbers(data.scores),
        doraIndicators: doraIndicators(data),
        hands,
        events: [normalizeEvent({name: "RecordDealTile", data: {
          seat: Number(data.ju), tile: hands[Number(data.ju)].pop(),
          left_tile_count: data.left_tile_count,
        }})],
      };
      rounds.push(round);
      continue;
    }
    const event = normalizeEvent(record);
    if (round && event) {
      if (event.type === "kan" && event.callType === "ankan") {
        event.tiles = playbackState(round).hands[event.seat]
          .filter(tile => tileBase(tile) === tileBase(event.tiles[0]));
      }
      // The opening discard can omit the drawn-discard flag even when
      // the dealer has no copy of that tile among the original thirteen.
      if (event.type === "discard" && round.events.length === 1 &&
          event.tile === round.events[0].tile &&
          !round.hands[event.seat].includes(event.tile)) {
        event.tsumogiri = true;
      }
      round.events.push(event);
      if (event.type === "ryuukyoku" && !event.scores.length) {
        event.scores = playbackState(round).scores;
      }
    }
  }
  if (!rounds.length) {
    throw new Error("record_contains_no_four_player_rounds");
  }
  return { gameId: recordId, rounds };
}
