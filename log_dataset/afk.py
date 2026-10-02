"""Conservative recorded behavioral evidence for excluding AFK rounds."""

import numpy

from model import replay

from . import shanten


def suspected_afk_kyokus(rounds):
    evidence = {}
    streaks = [[] for _ in range(4)]
    for number, events in enumerate(rounds):
        state = replay.Replay.from_start("quality", events[0])
        runs = [[] for _ in range(4)]
        held = [None] * 4
        active = [False] * 4
        all_discards = [[] for _ in range(4)]
        candidates = []
        for index, event in enumerate(events[1:], 1):
            state.apply(event)
            actor = event.get("actor")
            if event["type"] == "dahai":
                all_discards[actor].append(index)
                if event["tsumogiri"] and not state.declared[actor]:
                    if not runs[actor]:
                        held[actor] = state.concealed_counts(actor).copy()
                    runs[actor].append(index)
                else:
                    if runs[actor]:
                        candidates.append((actor, runs[actor], held[actor]))
                    runs[actor] = []
                    active[actor] = True
            elif event["type"] in {
                "reach",
                "chi",
                "pon",
                "ankan",
                "kakan",
                "daiminkan",
                "hora",
            }:
                if runs[actor]:
                    candidates.append((actor, runs[actor], held[actor]))
                runs[actor] = []
                active[actor] = True
        candidates.extend(
            (actor, runs[actor], held[actor])
            for actor in range(4)
            if runs[actor]
        )
        for actor, indices, counts in candidates:
            if len(indices) < 12 or any(
                event["type"] == "reach" and event["actor"] != actor
                for event in events[: indices[-1] + 1]
            ):
                continue
            distance = shanten.calculate_shanten(counts)
            ignored = [
                index
                for index in indices
                if shanten.calculate_shanten(
                    counts
                    + (
                        numpy.arange(34)
                        == shanten.base_id(events[index]["pai"])
                    )
                )
                < distance
            ]
            if distance > 0 and len(ignored) >= 2:
                evidence.setdefault(number, []).append(
                    {
                        "actor": actor,
                        "discards": len(indices),
                        "held_shanten": distance,
                        "event_indices": indices,
                        "ignored_improving_draws": ignored,
                        "rule": "midhand_tsumogiri",
                    }
                )
        for actor in range(4):
            if not active[actor] and len(all_discards[actor]) >= 8:
                streaks[actor].append((number, all_discards[actor]))
                if len(streaks[actor]) >= 2:
                    for previous, indices in streaks[actor]:
                        if previous not in evidence:
                            evidence[previous] = [
                                {
                                    "actor": actor,
                                    "discards": len(indices),
                                    "event_indices": indices,
                                    "rule": "whole_hand_tsumogiri",
                                }
                            ]
            else:
                streaks[actor] = []
    return evidence


def retained_hands(events):
    """Remove every player's decisions in a suspected AFK hand."""
    rounds = []
    for event in events:
        if event["type"] == "start_kyoku":
            rounds.append([])
        if rounds and event["type"] != "end_game":
            rounds[-1].append(event)
    excluded = suspected_afk_kyokus(rounds)
    return [
        hand for number, hand in enumerate(rounds) if number not in excluded
    ]
