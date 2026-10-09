function offeredButtons(position) {
  const controls = {
    1: "discard", 2: "chii", 3: "pon", 4: "kan", 5: "kan", 6: "kan",
    7: "reach", 8: "hora", 9: "hora", 10: "ryukyoku",
  };
  return [...new Set(position.operation.operation_list.map(
    (operation) => controls[operation.type],
  ).filter(Boolean)), "skip"];
}

function tileOrder(tile) {
  const suit = "mpsz".indexOf(tile.at(-1));
  const rank = Number(tile.slice(0, -1)) || 5;
  return suit * 20 + rank * 2 + (tile.startsWith("0") ? 0 : 1);
}

function soulTile(tile) {
  const honor = {
    E: "1z",
    S: "2z",
    W: "3z",
    N: "4z",
    P: "5z",
    F: "6z",
    C: "7z",
  };
  return honor[tile] || tile.replace(/^5([mps])r$/, "0$1");
}

function controllerTile(tile) {
  const honor = {
    "1z": "east",
    "2z": "south",
    "3z": "west",
    "4z": "north",
    "5z": "white",
    "6z": "green",
    "7z": "red",
  };
  return honor[tile] || tile.replace(/^0/, "5");
}

function doraTiles(position) {
  let indicators = position.round?.doraIndicators || [];
  for (const event of position.round?.events || []) {
    if (event.doras?.length) indicators = event.doras;
  }
  return indicators.map((tile) => {
    const rank = Number(tile[0]) || 5;
    const suit = tile[1];
    return `${suit !== "z" ? rank % 9 + 1 :
      rank <= 4 ? rank % 4 + 1 : (rank - 4) % 3 + 5}${suit}`;
  });
}

function discardTarget(action, position, control) {
  const tile = soulTile(action.pai);
  const label = controllerTile(tile);
  const handTiles = [
    ...position.hand.concealed,
    ...(position.hand.drawn ? [position.hand.drawn] : []),
  ];
  const screenHand = control.hand.tiles.slice(
    0, handTiles.length,
  );
  const dora = new Set(doraTiles(position).map(controllerTile));
  // Carry origin through sorting so identical held/drawn copies stay distinct.
  const entries = handTiles.map((tile, index) => ({
    tile, drawn: Boolean(position.hand.drawn && index === handTiles.length - 1),
  }));
  const compare = (left, right) => tileOrder(left.tile) - tileOrder(right.tile);
  const layouts = [...new Map([
    { name: "protocol", entries },
    { name: "sorted-concealed", entries: [
      ...entries.filter((entry) => !entry.drawn).sort(compare),
      ...entries.filter((entry) => entry.drawn),
    ] },
    { name: "sorted-all", entries: [...entries].sort(compare) },
  ].map((value) => ({
    name: value.name,
    tiles: value.entries.map((entry) => entry.tile),
    drawnIndex: value.entries.findIndex((entry) => entry.drawn),
  })).map((value) => [JSON.stringify([value.tiles, value.drawnIndex]), value])).values()]
    .filter((value) => screenHand.every((screenTile, index) =>
      (!screenTile.drawn || position.action === "ActionNewRound" ||
       value.drawnIndex === index) &&
      (!screenTile.trusted ||
       (screenTile.glare && (dora.has(controllerTile(value.tiles[index])) ||
         value.tiles[index].startsWith("0"))) ||
       controllerTile(value.tiles[index]) === screenTile.label),
    ))
    .map((value) => ({
      ...value,
      score: value.tiles.reduce((total, value, index) =>
        total + Math.log(Math.max(
          Number(screenHand[index]?.scores[controllerTile(value)] || 0),
          0.0001,
        )), 0),
    }))
    .sort((left, right) => right.score - left.score);
  if (!layouts.length) {
    throw new Error(`screen slot for ${label}: no layout agrees with recognized tiles`);
  }
  const index = action.tsumogiri === true
    ? layouts[0].drawnIndex
    : layouts[0].tiles.findIndex((value, index) => value === tile &&
      (action.tsumogiri !== false || index !== layouts[0].drawnIndex));
  if (index === -1 || layouts[0].tiles[index] !== tile) {
    throw new Error(`tracked hand does not contain the requested copy of ${tile}`);
  }
  const screenTile = screenHand[index];
  const score = Number(screenTile?.scores[label] || 0);
  const protocolDrawn = Boolean(
    screenTile?.drawn && position.hand.drawn === tile &&
    index === handTiles.length - 1,
  );
  if (!screenTile || (score < 0.7 && !protocolDrawn)) {
    throw new Error(
      `screen slot for ${label} failed visual verification (${score})`,
    );
  }
  const [x, y, width, height] = screenTile.box;
  return {
    x: x + Math.floor(width / 2),
    y: y + Math.floor(height / 2),
    control: "discard",
    label,
    score,
    layout: layouts[0].name,
    layoutScore: layouts[0].score,
    alternativeScore: layouts[1]?.score,
    verifiedBy: protocolDrawn && score < 0.7 ? "drawn-slot" : "visual",
  };
}

export { soulTile, discardTarget, doraTiles, offeredButtons };
