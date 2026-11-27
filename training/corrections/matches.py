"""Mine all-seat Mortal match trajectories with a frozen scoring policy."""

import gzip
import json
import pathlib

import numpy

from log_dataset import _native_collection, call_states, sakigiri_regret
from training import self_play


def sources(directory):
    """Index a complete standard-South Mortal match for all-seat replay."""
    directory = pathlib.Path(directory).resolve()
    manifest = json.loads((directory / "evaluation.json").read_text())
    if (
        manifest["format"] != "riichienv_standard_south"
        or not manifest["mortal"]
    ):
        raise ValueError("Root selection requires a standard Mortal match")
    games = [
        json.loads(line)
        for line in (directory / "games.jsonl").read_text().splitlines()
    ]
    if len(games) != manifest["games"] or len(games) != 10000:
        raise ValueError("Root selection requires all 10,000 match games")
    if len({game["replay"] for game in games}) != len(games):
        raise ValueError("Match contains duplicate replay paths")
    return [
        {
            "path": str(directory / game["replay"]),
            "seed": game["seed"],
            "new_seats": game["new_seats"],
        }
        for game in sorted(games, key=lambda game: game["replay"])
    ]


def discover(predict, games, capture):
    """Replay seeded matches, verifying generated events against every log.

    Recorded choices advance discovery only. Root comparisons and subsequent
    counterfactual continuations use the supplied frozen policy instead.
    """
    collectors = {}
    order_actions = []
    for game, game_record in enumerate(games):
        with gzip.open(game_record["path"], "rt") as stream:
            events = [json.loads(line) for line in stream]
        collectors[game] = _native_collection.NativeCollector.from_match(
            json.dumps(events), game_record["seed"], False
        )
        starts = [
            index
            for index, event in enumerate(events)
            if event["type"] == "start_kyoku"
        ]
        order_actions.append(
            {
                (start + index, events[start + index]["actor"]): actions
                for start, end in zip(starts, starts[1:] + [len(events)])
                for index, (actions, _) in sakigiri_regret.round_pairs(
                    str(game), events[start:end]
                ).items()
            }
        )
    statistics = self_play.collection_counts()
    while collectors:
        requests = {}
        for game, collector in list(collectors.items()):
            request = collector.next()
            if request is None:
                for name, value in collector.finish()[1].items():
                    statistics[name] += value
                del collectors[game]
            else:
                requests[game] = request
        if not requests:
            break
        features = call_states.concatenate(
            [call_states.from_request(request) for request in requests.values()]
        )
        outputs = predict(features)
        start = 0
        for game, request in requests.items():
            count = len(request["head"])
            batch = self_play.request_batch(request)
            batch["indices"][:, 0] = game
            capture(
                collectors[game],
                {
                    **batch,
                    "sakigiri_actions": [
                        numpy.asarray(
                            order_actions[game].get((event, seat), ()),
                            dtype=numpy.int32,
                        )
                        for event, seat in zip(
                            request["event"], request["seat"]
                        )
                    ],
                    "behavior": self_play.policy_log_probabilities(
                        {
                            name: value[start : start + count]
                            for name, value in outputs.items()
                        },
                        batch["head"],
                        batch["legal"],
                    ),
                },
            )
            collectors[game].submit_recorded()
            start += count
    return {**statistics, "logged_games": len(games)}
