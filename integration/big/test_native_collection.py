"""Native collection retains the exact public feature contract."""

import base64
import json
import pathlib

import numpy
import riichienv

from integration import collection_reference, feature_reference
from log_dataset import _native_collection, features, replay
from model import actions, self_play


def test_native_features_match_python_during_complete_games(monkeypatch):
    pack = features.pack
    histories = {}
    coverage = set()

    def checked_pack(*args, **kwargs):
        result = pack(*args, **kwargs)
        state = kwargs["state"]
        assert result["opponent_discard_count"][0] == len(
            state.rivers[kwargs["actor"]]
        )
        expected = feature_reference.pack(
            **{
                **kwargs,
                "legal_observation": state.legal_observation(kwargs["actor"]),
            }
        )
        if state not in histories:
            histories[state] = [_native_collection.FeatureReplay(), 0]
        native, cursor = histories[state]
        native.apply(json.dumps(state.history[cursor:]))
        histories[state][1] = len(state.history)
        actual = native.pack(kwargs["actor"], features.PHASES[kwargs["phase"]])
        for name, spec in _native_collection.INPUTS.items():
            numpy.testing.assert_array_equal(result[name], expected[name])
            numpy.testing.assert_array_equal(
                numpy.frombuffer(actual[name], spec["dtype"]).reshape(
                    spec["shape"]
                ),
                expected[name],
                err_msg=(
                    f"{name}: hand={state.kyoku_id}, "
                    f"event={state.event_index}, actor={kwargs['actor']}, "
                    f"phase={kwargs['phase']}"
                ),
            )
        coverage.add(kwargs["phase"])
        if expected["candidate_hand_red"].any():
            coverage.add("red")
        if expected["meld_valid"].any():
            coverage.add("meld")
        return result

    def predict(inputs):
        return {
            name: numpy.zeros(
                (
                    len(inputs["decision_phase"]),
                    actions.ACTOR_OUTPUTS[name][0],
                ),
                numpy.float32,
            )
            for name in actions.TRAIN_POLICY_HEADS
        }

    monkeypatch.setattr(features, "pack", checked_pack)
    _, statistics = collection_reference.collect(predict, 2, 12)
    assert statistics["games"] == 2
    assert {"DISCARD", "RESPONSE", "red", "meld"} <= coverage


def test_native_collector_features_and_counters_match_recorded_play():
    collector = _native_collection.NativeCollector(2, 13)
    contexts = [
        dict(cursor=0, state=None, pending={}, hand=-1) for _ in range(2)
    ]
    statistics = self_play.collection_counts()
    generator = numpy.random.default_rng(13)
    checked = 0
    while (request := collector.next()) is not None:
        logs = [json.loads(events) for events in collector.logs()]
        assert request["event"] == [len(logs[game]) for game in request["game"]]
        for ctx, events in zip(contexts, logs):
            collection_reference.advance(
                ctx, events[ctx["cursor"] :], statistics
            )
        actual = numpy.frombuffer(request["features"], self_play.FEATURE_DTYPE)
        legal = numpy.frombuffer(request["legal"], numpy.uint8).reshape(-1, 150)
        for position, observation in enumerate(collector.observations()):
            observation = riichienv.Observation.deserialize_from_base64(
                base64.b64encode(observation.encode()).decode()
            )
            state = contexts[request["game"][position]]["state"]
            expected = feature_reference.pack(
                state=state,
                actor=request["seat"][position],
                phase=next(
                    name
                    for name, code in features.PHASES.items()
                    if code == actual["decision_phase"][position]
                ),
                legal_observation=observation,
            )
            for name in _native_collection.INPUTS:
                numpy.testing.assert_array_equal(
                    actual[name][position], expected[name], err_msg=name
                )
            candidates = actions.policy_candidates(
                dict(
                    features=expected,
                    actor=request["seat"][position],
                    legal_actions=observation.legal_actions(),
                    drawn_tile=observation.drawn_tile,
                    phase=next(
                        name
                        for name, code in features.PHASES.items()
                        if code == actual["decision_phase"][position]
                    ),
                )
            )
            assert request["head"][position] == (
                actions.TRAIN_POLICY_HEADS.index(candidates[0][0])
            )
            assert set(numpy.flatnonzero(legal[position])) == {
                candidate[1] for candidate in candidates
            }
            checked += 1
        collector.submit(
            [int(generator.choice(numpy.flatnonzero(row))) for row in legal]
        )
        try:
            logs, counts = collector.finish()
        except ValueError as error:
            assert str(error) == "collection is not complete"
        else:
            break
    else:
        raise AssertionError("expected completion immediately after submit")
    for ctx, events in zip(contexts, map(json.loads, logs)):
        collection_reference.advance(ctx, events[ctx["cursor"] :], statistics)
    assert checked > 100
    assert counts == statistics
    assert counts["games"] == 2


def test_native_feature_parity_covers_riichi_kan_and_scoring():
    coverage = set()
    for filename in (
        "payment_chankan_ippatsu",
        "payment_dealer_90fu",
        "riichi_push",
    ):
        events = json.loads(
            (
                pathlib.Path(__file__).parent.parent
                / f"fixtures/model/{filename}.json"
            ).read_text()
        )
        state = None
        for event in events:
            if event["type"] == "start_kyoku":
                state = replay.Replay.from_start(filename, event)
                native = _native_collection.FeatureReplay()
                known = [
                    seat
                    for seat, hand in enumerate(event["tehais"])
                    if "?" not in hand
                ]
            elif state is None:
                continue
            else:
                state.apply(event)
            native.apply(json.dumps([event]))
            for actor in known:
                observation = state.legal_observation(actor)
                if not observation.legal_actions():
                    continue
                phase = actions.phase_for_actions(observation.legal_actions())
                expected = feature_reference.pack(
                    state, actor, phase, observation
                )
                actual = native.pack(actor, features.PHASES[phase])
                for name, spec in _native_collection.INPUTS.items():
                    numpy.testing.assert_array_equal(
                        numpy.frombuffer(actual[name], spec["dtype"]).reshape(
                            spec["shape"]
                        ),
                        expected[name],
                        err_msg=(
                            f"{filename}: {name}, event={state.event_index}, "
                            f"actor={actor}, phase={phase}"
                        ),
                    )
                coverage.add(phase)
                if expected["action_value_valid"].any():
                    coverage.add("value_ranges")
                if expected["riichi_push_count"].any():
                    coverage.add("riichi_push")
    assert {"RIICHI", "KAN", "WIN", "value_ranges", "riichi_push"} <= coverage
