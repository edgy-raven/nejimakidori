"""Mine logged trajectories while scoring roots with a frozen policy."""

import contextlib
import gzip
import json
import pathlib
import zipfile

import numpy

from log_dataset import _native_collection, replay
from model import actions
from training import self_play


def expert_sources(archives):
    """Index every game in the supplied expert archives in stable order."""
    sources = []
    for path in sorted(pathlib.Path(path).resolve() for path in archives):
        with zipfile.ZipFile(path) as archive:
            sources.extend(
                {
                    "archive": str(path),
                    "member": member,
                    "game_id": pathlib.Path(member).stem,
                }
                for member in sorted(archive.namelist())
                if member.endswith((".mjson", ".jsonl"))
            )
    if len({game["game_id"] for game in sources}) != len(sources):
        raise ValueError("Expert archives contain duplicate game IDs")
    return sources


class ExpertRoot:
    """Materialize only retained roots, using their observed hand prefix."""

    def __init__(self, prefix, actor, phase, row):
        self.prefix = prefix
        self.actor = actor
        self.phase = phase
        self.row = row
        self.native = None

    @property
    def decision(self):
        return self.native.decision

    def fork(self, copies, seed, **kwargs):
        if self.native is None:
            self.native = _native_collection.NativeRoot.from_expert_prefix(
                self.prefix, self.actor, self.phase, seed
            )
            actual = numpy.frombuffer(
                self.native.features, self_play.FEATURE_DTYPE
            )[0]
            different = [
                name
                for name in self_play.FEATURE_DTYPE.names
                if not numpy.array_equal(actual[name], self.row[name])
            ]
            if different:
                raise ValueError(
                    "Expert root features differ from replay: "
                    + ", ".join(different)
                )
        return self.native.fork(copies, seed, **kwargs)


class ExpertReplay:
    """Expose public decisions; recorded events only advance the trajectory."""

    def __init__(self, game_id, events):
        self.state = replay.Replay(game_id, events)
        self.hand = -1
        self.rows = []
        self.requests = self.iter_requests()
        self.pending = next(self.requests, None)

    def iter_requests(self):
        for index, event in self.state.replay_events():
            if event["type"] == "start_kyoku":
                self.hand += 1
            if event["type"] not in {"tsumo", "dahai"}:
                continue
            self.rows = []
            for actor in (
                (event["actor"],) if event["type"] == "tsumo" else range(4)
            ):
                observation = self.state.legal_observation(actor)
                legal = observation.legal_actions()
                if (
                    not legal
                    or actions.forced_policy_action(self.state, actor, legal)
                    is not None
                ):
                    continue
                _, request = actions.policy_request(
                    self.state, actor, observation
                )
                candidates = actions.policy_candidates(request)
                row = numpy.zeros((), self_play.FEATURE_DTYPE)
                for name, value in request["features"].items():
                    row[name] = value
                self.rows.append(
                    {
                        "actor": actor,
                        "features": row,
                        "head": actions.TRAIN_POLICY_HEADS.index(
                            candidates[0][0]
                        ),
                        "legal": numpy.isin(
                            numpy.arange(150), [c[1] for c in candidates]
                        ),
                    }
                )
            if self.rows:
                yield {
                    "features": numpy.array(
                        [row["features"] for row in self.rows],
                        dtype=self_play.FEATURE_DTYPE,
                    ).tobytes(),
                    "head": [row["head"] for row in self.rows],
                    "legal": numpy.array(
                        [row["legal"] for row in self.rows], dtype=numpy.bool_
                    ).tobytes(),
                    "hand": [self.hand] * len(self.rows),
                    "seat": [row["actor"] for row in self.rows],
                    "event": [index + 1] * len(self.rows),
                }

    def next(self):
        return self.pending

    def submit_recorded(self):
        self.pending = next(self.requests, None)

    def snapshot(self, row):
        return ExpertRoot(
            json.dumps(self.state.history),
            self.rows[row]["actor"],
            int(self.rows[row]["features"]["decision_phase"]),
            self.rows[row]["features"].copy(),
        )

    def finish(self):
        assert self.pending is None
        return [], {
            "games": 1,
            "expert_games": 1,
            "expert_hands": self.hand + 1,
        }


def discover(predict, games, capture):
    """Replay seeded matches or archived expert logs; retain selected roots.

    Seeded native replay verifies every generated event against the log.
    Expert roots reconstruct simulator state from their observed prefix only.
    Recorded choices advance discovery only. Root comparisons and subsequent
    counterfactual continuations use the supplied frozen policy instead.
    """
    collectors = []
    with contextlib.ExitStack() as stack:
        archives = {
            path: stack.enter_context(zipfile.ZipFile(path))
            for path in sorted(
                {game["archive"] for game in games if "archive" in game}
            )
        }
        for game in games:
            if "archive" in game:
                events = [
                    json.loads(line)
                    for line in archives[game["archive"]]
                    .read(game["member"])
                    .splitlines()
                ]
                collectors.append(ExpertReplay(game["game_id"], events))
            else:
                with gzip.open(game["path"], "rt") as stream:
                    events = [json.loads(line) for line in stream]
                collectors.append(
                    _native_collection.NativeCollector.from_match(
                        json.dumps(events), game["seed"], game["duplicate"]
                    )
                )
    statistics = self_play.collection_counts()
    statistics.update(expert_games=0, expert_hands=0)
    active = dict(enumerate(collectors))
    while active:
        requests = {}
        for game, collector in list(active.items()):
            request = collector.next()
            if request is None:
                for name, value in collector.finish()[1].items():
                    statistics[name] += value
                del active[game]
            else:
                requests[game] = request
        if not requests:
            break
        features = numpy.concatenate(
            [
                numpy.frombuffer(request["features"], self_play.FEATURE_DTYPE)
                for request in requests.values()
            ]
        )
        outputs = predict(features)
        start = 0
        for game, request in requests.items():
            count = len(request["head"])
            heads = numpy.asarray(request["head"])
            legal = numpy.frombuffer(request["legal"], numpy.bool_).reshape(
                count, 150
            )
            scores = numpy.full((count, 150), -1e9, numpy.float32)
            for head, name in enumerate(actions.TRAIN_POLICY_HEADS):
                rows = heads == head
                scores[rows, : actions.ACTOR_OUTPUTS[name][0]] = outputs[name][
                    start : start + count
                ][rows]
            scores[~legal] = -1e9
            scores -= scores.max(axis=1, keepdims=True)
            behavior = scores - numpy.log(
                numpy.exp(scores).sum(axis=1, keepdims=True)
            )
            behavior[~legal] = -1e9
            capture(
                collectors[game],
                {
                    "features": features[start : start + count],
                    "head": heads,
                    "legal": legal,
                    "behavior": behavior,
                    "indices": numpy.column_stack(
                        (
                            numpy.full(count, game, numpy.int64),
                            *[
                                request[name]
                                for name in ("hand", "seat", "event")
                            ],
                        )
                    ),
                },
            )
            collectors[game].submit_recorded()
            start += count
    return {**statistics, "logged_games": len(games)}
