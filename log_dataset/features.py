"""Actor-relative public tensors and deterministic post-action hand banks."""

import json

import numpy
import riichienv

from log_dataset import _native_collection, shanten

PHASES = {"DISCARD": 0, "RESPONSE": 1, "KAN": 2, "RIICHI": 3, "WIN": 4}


def native_tile_code(tile):
    return 34 + (tile // 4) // 9 if tile in (16, 52, 88) else tile // 4


def action_event(action, drawn_tile=None):
    event = json.loads(action.to_mjai())
    if event["type"] == "dahai":
        event["tsumogiri"] = action.tile == drawn_tile
    return event


def single_legal_action(legal, drawn_tile):
    """Physical copies of the same held tile are one available action."""
    events = [action_event(action, drawn_tile) for action in legal]
    for event in events:
        if event["type"] == "dahai":
            del event["tsumogiri"]
    if events and all(event == events[0] for event in events):
        return next(
            (action for action in legal if action.tile == drawn_tile), legal[0]
        )
    return None


def prefer_red_calls(legal):
    selected = {}
    for index, action in enumerate(legal):
        if action.action_type in (
            riichienv.ActionType.CHI,
            riichienv.ActionType.PON,
        ):
            key = (
                action.action_type,
                action.tile // 4,
                tuple(sorted(tile // 4 for tile in action.consume_tiles)),
            )
            if key in selected and any(
                tile in (16, 52, 88) for tile in selected[key].consume_tiles
            ):
                continue
        else:
            key = index
        selected[key] = action
    return list(selected.values())


def preferred_recorded_call(kind, consumed, legal):
    return any(
        event["type"] == kind and sorted(event["consumed"]) == sorted(consumed)
        for action in prefer_red_calls(legal)
        if action.action_type
        in (riichienv.ActionType.CHI, riichienv.ActionType.PON)
        for event in [action_event(action)]
    )


def drawn_tile_code(packed):
    tile = int(packed["current_draw_tile_id"])
    return 34 + tile // 9 if packed["current_draw_is_red"] else tile


def chii_pattern(called, consumed):
    called = (
        shanten.mjai_tile_code(called)
        if isinstance(called, str)
        else native_tile_code(called)
    )
    consumed = [
        (
            shanten.mjai_tile_code(tile)
            if isinstance(tile, str)
            else native_tile_code(tile)
        )
        for tile in consumed
    ]
    return shanten.base_id(called) - min(
        shanten.base_id(tile) for tile in [called] + consumed
    )


def pack(
    state,
    actor,
    phase,
    east_only=False,
    *,
    include_focus=False,
):
    """Expose shared C++ feature buffers for the current public history."""
    if state.feature_replay is None:
        state.feature_replay = _native_collection.FeatureReplay()
    state.feature_replay.apply(
        json.dumps(state.history[state.feature_cursor :])
    )
    state.feature_cursor = len(state.history)
    buffers = state.feature_replay.pack(
        actor, PHASES[phase], east_only, include_focus
    )
    packed = {
        name: numpy.frombuffer(buffers[name], spec["dtype"])
        .reshape(spec["shape"])
        .copy()
        for name, spec in _native_collection.INPUTS.items()
    }

    return (packed, buffers["focus"]) if include_focus else packed
