import {soulTile} from "./mahjongsoul_targeting.mjs";

// Test accounts keep their hands closed and never end a hand by claiming a win.
export function riichiOnlyDecision(position, body) {
  const riichi = position.operation.operation_list.find(item => item.type === 7);
  if (riichi) {
    const discard = [body.riichi_discard, ...body.possible_actions].find(
      action => action?.type === "dahai" &&
        riichi.combination.includes(soulTile(action.pai))) || {
      actor: position.selfSeat, type: "dahai", pai: riichi.combination[0],
      tsumogiri: riichi.combination[0] === position.hand.drawn,
    };
    return {...body, action: {actor: position.selfSeat, type: "reach"},
      riichi_discard: discard, action_kind: "riichi-only debug"};
  }
  if (position.hand.riichi && position.operation.operation_list.some(
    operation => [4, 5, 6, 8, 9].includes(operation.type))) {
    return {...body, action: {actor: position.selfSeat, type: "none"},
      action_kind: "riichi-only debug"};
  }
  const action = body.action.type === "dahai" ? body.action
    : body.possible_actions.find(item => item.type === "dahai") ||
      {actor: position.selfSeat, type: "none"};
  return {...body, action, action_kind: "riichi-only debug"};
}
