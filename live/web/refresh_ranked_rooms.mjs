// Regenerate from the official client's ConfigTables binary and schema.
// node refresh_ranked_rooms.mjs /path/to/config.proto /path/to/lqc.lqbin
import fs from "node:fs/promises";
import protobuf from "protobufjs";

const root = await protobuf.load(process.argv[2]);
const tables = root.lookupType("lq.config.ConfigTables").decode(
  await fs.readFile(process.argv[3]));
const schema = tables.schemas.find(table => table.name === "desktop")
  .sheets.find(sheet => sheet.name === "matchmode");
const rowType = new protobuf.Type("MatchMode");
for (const field of schema.fields) {
  rowType.add(new protobuf.Field(field.fieldName, field.pbIndex,
    field.pbType, field.arrayLength ? "repeated" : undefined));
}
const rooms = tables.datas.find(data =>
  data.table === "desktop" && data.sheet === "matchmode").data
  .map(data => rowType.toObject(rowType.decode(data), {defaults: true}))
  .filter(row => row.type === 1 && row.is_open === 1 &&
    row.activity_id === 0 && row.mode === (row.room <= 2 ? 1 : 2))
  .sort((a, b) => a.room - b.room)
  .map(row => ({
    modeId: row.id, name: row.room_name_en.replace(" Room", ""),
    length: row.mode === 1 ? "East" : "South",
    rankMin: row.level_limit, rankMax: row.level_limit_ceil,
    copperMin: row.glimit_floor,
    fee: row.tip,
  }));
await fs.writeFile(new URL("./ranked_rooms.json", import.meta.url),
  JSON.stringify(rooms, null, 2) + "\n");
