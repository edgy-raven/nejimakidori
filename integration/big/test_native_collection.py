"""Native collection retains the exact public feature contract."""

import json
import pathlib

import numpy

from integration import collection_reference, feature_reference
from log_dataset import (
    _native_collection,
    call_states,
    features,
    replay,
    shanten,
)
from model import actions
from training import self_play


def test_south_hands_remaining_preserves_round_wind_and_extensions():
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    for wind, kyoku, remaining in (
        ("E", 1, 7),
        ("E", 4, 4),
        ("S", 4, 0),
        ("W", 1, -1),
        ("W", 2, -2),
    ):
        state = replay.Replay.from_start(
            "round-progress", {**events[0], "bakaze": wind, "kyoku": kyoku}
        )
        state.apply(events[1])
        actual = features.pack(state, 0, "DISCARD")
        expected = feature_reference.pack(
            state=state,
            actor=0,
            phase="DISCARD",
            legal_observation=state.legal_observation(0),
        )
        assert actual["hands_remaining"] == remaining
        assert actual["prevailing_wind"] == "ESWN".index(wind)
        for name in _native_collection.INPUTS:
            numpy.testing.assert_array_equal(actual[name], expected[name])


def test_scores_and_ranks_follow_current_winds_for_every_actor():
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    for actor in range(4):
        state = replay.Replay.from_start(
            "wind-ordered-scores",
            {
                **events[0],
                "oya": 2,
                "kyoku": 3,
                "scores": [50000, 35000, 35000, -20000],
                "tehais": [
                    events[0]["tehais"][(seat - actor) % 4] for seat in range(4)
                ],
            },
        )
        state.apply({**events[1], "actor": actor})
        for packed in (
            features.pack(state, actor, "DISCARD"),
            feature_reference.pack(
                state=state,
                actor=actor,
                phase="DISCARD",
                legal_observation=state.legal_observation(actor),
            ),
        ):
            numpy.testing.assert_array_equal(
                packed["scores"], [35000, -20000, 50000, 35000]
            )
            # At East 3, initial seat order still breaks ties. Current North
            # outranks current East despite East appearing first in the row.
            numpy.testing.assert_array_equal(packed["ranks"], [3, 4, 1, 2])
            assert packed["scores"].dtype == numpy.int32
            assert packed["ranks"].dtype == numpy.int8


def test_native_features_match_python_during_complete_games(monkeypatch):
    pack = features.pack
    coverage = set()

    def checked_pack(state, actor, phase):
        result = pack(state, actor, phase)
        assert result["opponent_discard_count"][0] == len(state.rivers[actor])
        expected = feature_reference.pack(
            state, actor, phase, state.legal_observation(actor)
        )
        for name in _native_collection.INPUTS:
            numpy.testing.assert_array_equal(
                result[name],
                expected[name],
                err_msg=(
                    f"{name}: hand={state.kyoku_id}, "
                    f"event={state.event_index}, actor={actor}, "
                    f"phase={phase}"
                ),
            )
        for code in numpy.flatnonzero(result["discard_action_legal"]):
            after = state.native_env.clone()
            after.apply_event(
                {
                    "type": "dahai",
                    "actor": actor,
                    "pai": shanten.TILE_NAMES[code],
                    "tsumogiri": bool(
                        result["current_draw_valid"]
                        and code == features.drawn_tile_code(result)
                    ),
                }
            )
            numpy.testing.assert_array_equal(
                result["candidate_hand_yaku_possibility"][
                    result["discard_candidate_hand"][code]
                ],
                feature_reference.yaku_flags(
                    after.get_observation(actor), actor
                ),
            )
            coverage.add("red_discard" if code >= 34 else "ordinary_discard")
            if state.melds[actor]:
                coverage.add("meld_discard")
        coverage.add(phase)
        if expected["candidate_hand_red"].any():
            coverage.add("red")
        if (expected["meld_tiles"][..., 0] < 37).any():
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
    assert {
        "DISCARD",
        "RESPONSE",
        "red",
        "meld",
        "red_discard",
        "ordinary_discard",
        "meld_discard",
    } <= coverage
    assert [
        shanten.mjai_tile_code(tile) for tile in shanten.TILE_NAMES
    ] == list(range(37))


def test_native_collector_features_and_counters_match_recorded_play():
    collector = _native_collection.NativeCollector(2, 13)
    contexts = [
        dict(cursor=0, state=None, pending={}, hand=-1) for _ in range(2)
    ]
    statistics = self_play.collection_counts()
    # The Python collection wrapper owns this counter, not NativeCollector.
    del statistics["simulated_decisions"]
    generator = numpy.random.default_rng(13)
    checked = 0
    while (request := collector.next()) is not None:
        assert set(request) == {
            "features",
            "post_call_features",
            "post_call_owner",
            "post_call_option",
            "reference_dora_tile_id",
            "reference_dora_valid",
            "head",
            "game",
            "hand",
            "seat",
            "event",
            "legal",
        }
        logs = [json.loads(events) for events in collector.logs()]
        assert request["event"] == [len(logs[game]) for game in request["game"]]
        for ctx, events in zip(contexts, logs):
            collection_reference.advance(
                ctx, events[ctx["cursor"] :], statistics
            )
        actual = numpy.frombuffer(request["features"], call_states.ROW_DTYPE)
        legal = numpy.frombuffer(request["legal"], numpy.uint8).reshape(-1, 150)
        for position, game in enumerate(request["game"]):
            state = contexts[game]["state"]
            observation = state.legal_observation(request["seat"][position])
            for name, value in features.reference_inputs(state).items():
                numpy.testing.assert_array_equal(
                    numpy.frombuffer(
                        request["reference_" + name], "int8"
                    ).reshape(-1, 5)[position],
                    value,
                )
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
                if expected["action_win_value_valid"].any():
                    coverage.add("value_ranges")
                if expected["riichi_push_count"].any():
                    coverage.add("riichi_push")
    assert {"RIICHI", "KAN", "WIN", "value_ranges", "riichi_push"} <= coverage
