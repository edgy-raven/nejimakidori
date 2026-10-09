"""Read cached browser rounds as MJAI events and decision-time state."""

from log_dataset import replay


def tile_name(tile):
    if tile == "?":
        return tile
    if not isinstance(tile, str) or len(tile) != 2 or tile[1] not in "mpsz":
        raise ValueError("Invalid tile: " + str(tile))
    if tile[1] == "z":
        return ("E", "S", "W", "N", "P", "F", "C")[int(tile[0]) - 1]
    return "5" + tile[1] + "r" if tile[0] == "0" else tile


def round_events(round_data):
    if len(round_data["hands"]) != 4 or len(round_data["scores"]) != 4:
        raise ValueError("A round requires four players.")
    if not 1 <= round_data["kyoku"] <= 4:
        raise ValueError("Invalid round number.")
    events = [
        {
            "type": "start_kyoku",
            "bakaze": ("E", "S", "W", "N")[round_data["prevailingWind"]],
            "kyoku": round_data["kyoku"],
            "honba": round_data["honba"],
            "kyotaku": round_data["riichiSticks"],
            "oya": round_data["kyoku"] - 1,
            "scores": round_data["scores"],
            "dora_marker": tile_name(round_data["doraIndicators"][0]),
            "tehais": [
                [tile_name(tile) for tile in hand]
                for hand in round_data["hands"]
            ],
        }
    ]
    cursors = []
    dora_count = len(round_data["doraIndicators"])
    accepted = set()
    last_discard = None
    for event in round_data["events"]:
        if event.get("liqi") and event["liqi"]["seat"] not in accepted:
            events.append(
                {"type": "reach_accepted", "actor": event["liqi"]["seat"]}
            )
            accepted.add(event["liqi"]["seat"])
        cursors.append(len(events))
        if event["type"] == "draw":
            events.append(
                {
                    "type": "tsumo",
                    "actor": event["seat"],
                    "pai": tile_name(event["tile"]),
                }
            )
        elif event["type"] == "discard":
            if event["riichi"]:
                events.append({"type": "reach", "actor": event["seat"]})
            last_discard = event["seat"]
            events.append(
                {
                    "type": "dahai",
                    "actor": event["seat"],
                    "pai": tile_name(event["tile"]),
                    "tsumogiri": event["tsumogiri"],
                }
            )
        elif event["type"] == "call":
            called = next(
                index
                for index, seat in enumerate(event["froms"])
                if seat != event["seat"]
            )
            events.append(
                {
                    "type": (
                        "chi"
                        if event["callType"] == "chii"
                        else event["callType"]
                    ),
                    "actor": event["seat"],
                    "target": event["froms"][called],
                    "pai": tile_name(event["tiles"][called]),
                    "consumed": [
                        tile_name(tile)
                        for index, tile in enumerate(event["tiles"])
                        if index != called
                    ],
                }
            )
        elif event["type"] == "kan":
            converted = {
                "type": event["callType"],
                "actor": event["seat"],
                "consumed": [tile_name(tile) for tile in event["tiles"]],
            }
            if event["callType"] == "kakan":
                converted["pai"] = converted["consumed"].pop()
            events.append(converted)
        elif event["type"] == "agari":
            for index, winner in enumerate(event["hules"]):
                events.append(
                    {
                        "type": "hora",
                        "actor": winner["seat"],
                        "target": (
                            winner["seat"] if winner["zimo"] else last_discard
                        ),
                        "deltas": event["deltas"] if index == 0 else [0] * 4,
                    }
                )
        elif event["type"] == "ryuukyoku":
            events.append({"type": "ryukyoku", "deltas": event["deltas"]})
        else:
            raise ValueError("Unsupported round event: " + event["type"])
        for tile in event.get("doras", [])[dora_count:]:
            events.append({"type": "dora", "dora_marker": tile_name(tile)})
            dora_count += 1
    return events, cursors


def reconstruct(round_data, cursor):
    try:
        events, cursors = round_events(round_data)
    except (KeyError, TypeError, IndexError, StopIteration) as error:
        raise ValueError("Invalid round structure: " + str(error)) from error
    if type(cursor) is not int or not 0 <= cursor <= len(cursors):
        raise ValueError("Event cursor is outside this hand.")
    state = replay.Replay.from_start("browser", events[0])
    stop = cursors[cursor] if cursor < len(cursors) else len(events)
    for event in events[1:stop]:
        state.apply(event)
    return state, events
