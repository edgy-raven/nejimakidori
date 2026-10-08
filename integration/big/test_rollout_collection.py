"""Disagreements preserve decision boundaries and enter paired replay."""

import argparse
import json
import pathlib
import types

import numpy
from log_dataset import replay, rollout_collection, rollout_store, shanten
from model import (
    actions,
    collect_rollouts,
    features,
    outcomes,
    rollout_training,
)

import riichienv


def test_special_draw_separates_riichi_deposit_and_final_bank_award():
    recorded = json.loads(
        pathlib.Path(
            "integration/fixtures/model/nagashi_deposit.json"
        ).read_text()
    )
    for finished in (False, True):
        native_scores = numpy.array(recorded["scores"])
        if finished:
            native_scores[3] += 1000
        env = types.SimpleNamespace(
            mjai_log=recorded["events"],
            scores=lambda: native_scores.tolist(),
            ranks=lambda: recorded["ranks"],
            done=lambda: finished,
            round_wind=0,
            oya=3,
        )
        result = collect_rollouts.settlement(
            env, types.SimpleNamespace(accepted=recorded["accepted"])
        )
        numpy.testing.assert_array_equal(result["scores"], recorded["scores"])
        draw = next(
            event for event in result["events"] if event["type"] == "ryukyoku"
        )
        assert draw["deltas"] == [-2000, -2000, 8000, -4000]
        assert result["payments"][3][3][3] == -1000
        numpy.testing.assert_array_equal(
            outcomes.settlement(result["payments"]),
            [-2000, -2000, 8000, -5000, 1000],
        )
        assert result["native_scores"] == native_scores.tolist()


def test_disagreement_queue_preserves_phase_pairs_and_training_eligibility(
    tmp_path,
):
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    state = replay.Replay("queue", events)
    next(state.policy_decisions())
    actor = state.history[-1]["actor"]
    packed = features.pack(
        state, actor, "DISCARD", state.legal_observation(actor)
    )
    legal = numpy.flatnonzero(features.discard_legal(packed))
    prediction = {
        "riichi_policy_probabilities": [0.0] * 74,
        "response_policy_probabilities": [0.0] * 150,
        "kan_action_probabilities": [0.0] * 35,
        "discard_policy_probabilities": numpy.eye(37)[legal[0]].tolist(),
        "action_values": [
            {
                "action": int(code),
                "expected_points": [float(code)] * 4,
                "projected_st3": float(code),
            }
            for code in legal
        ],
    }
    assert not rollout_store.eligible(packed)
    queue = rollout_collection.Queue(tmp_path / "queue.sqlite")
    kwargs = dict(
        state={
            "events": state.history,
            "actor": actor,
            "game_id": "rollout-collection",
        },
        packed=packed,
        prediction=prediction,
        expert={"discard_policy": int(legal[1])},
        policy={"revision": "frozen"},
    )
    identity = queue.mark(**kwargs)
    assert queue.mark(**kwargs) == identity
    root = queue.next(1)
    assert root["mark"]["disagreement"]
    assert root["mark"]["nominations"]["actor"] == legal[0]
    assert root["mark"]["nominations"]["expert"] == legal[1]
    assert root["state"]["events"][-1]["type"] == "tsumo"
    assert root["state"]["phase"] == 0
    results = {
        code: {
            "payments": numpy.zeros((4, 4, 4), int).tolist(),
            "scores": [25000] * 4,
            "events": [{"type": "ryukyoku"}],
        }
        for code in set(root["mark"]["nominations"].values())
    }
    queue.complete(root, 123, results)
    assert queue.next(1) is None
    assert queue.next(2)["trial"] == 1
    count = len(results)
    assert (
        queue.connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        == count * (count - 1) // 2
    )
    queue.close()
    collect_rollouts.collect(
        argparse.Namespace(
            store=tmp_path / "queue.sqlite",
            seconds=1,
            reserve_seconds=30,
            trials=2,
            seed=123,
        )
    )
    queue = rollout_collection.Queue(tmp_path / "queue.sqlite")
    assert queue.next(2)["trial"] == 1
    queue.close()
    trainer = rollout_training.Replay(tmp_path / "queue.sqlite")
    assert len(trainer.positions) == 1
    assert trainer.batch(0, 1)[0][1].shape == (2,)
    trainer.close()


def test_native_recovery_precedes_riichi_call_and_kan_commitments():
    recovered = set()
    for name in ("pass_speed_round", "payment_dealer_90fu"):
        events = json.loads(
            pathlib.Path(f"integration/fixtures/model/{name}.json").read_text()
        )
        events = events[
            next(
                index
                for index, event in enumerate(events)
                if event["type"] == "start_kyoku"
            ) :
        ]
        state = replay.Replay(name, events)
        for _, actor, phase, _, _ in state.policy_decisions():
            after_kan = any(event["type"] == "dora" for event in state.history)
            key = "after_kan" if after_kan else phase
            if key in recovered or key not in {
                "RIICHI",
                "RESPONSE",
                "KAN",
                "after_kan",
            }:
                continue
            env, _ = collect_rollouts.recover(
                collect_rollouts.prepare(
                    {"events": state.history, "actor": actor}
                ),
                941,
            )
            assert env.scores() == state.native_env.scores()
            assert len(env.wall) == state.live_wall + 14
            kinds = {
                action.action_type
                for action in env.get_observation(actor).legal_actions()
            }
            if phase == "RIICHI":
                assert riichienv.ActionType.RIICHI in kinds
                assert not env.riichi_declared[actor]
            elif phase == "RESPONSE":
                assert riichienv.ActionType.PASS in kinds
            elif phase == "KAN":
                assert kinds & actions.KAN_ACTIONS
            recovered.add(key)
    assert recovered == {"RIICHI", "RESPONSE", "KAN", "after_kan"}


def test_riichi_nomination_uses_marginal_branch_before_discard():
    packed = {
        "decision_phase": 3,
        "normal_self_shanten": 1,
        "chiitoitsu_self_shanten": 5,
        "kokushi_self_shanten": 5,
        "live_wall_count": 40,
    }
    probability = numpy.zeros(74)
    probability[[0, 37, 38]] = [0.4, 0.3, 0.3]
    mark = rollout_collection.nominations(
        packed,
        {
            "riichi_policy_probabilities": probability.tolist(),
            "action_values": [
                {
                    "action": code,
                    "expected_points": [0] * 4,
                    "projected_st3": float(code),
                }
                for code in (37, 74, 75)
            ],
        },
        {"riichi_policy": 0},
    )
    assert mark["nominations"]["actor"] == 74
    assert mark["nominations"]["expert"] == 37
    assert mark["disagreement"]


def test_paired_branches_commit_riichi_and_call_with_the_nominated_discard():
    class PassivePolicy:
        def __init__(self):
            self.requests = []
            self.batches = []
            self.streams = {}

        def reset(self):
            self.streams.clear()

        def select_many(self, requests, batch_size):
            assert batch_size == 1
            self.batches.append({stream // 4 for stream, _, _ in requests})
            for stream, env, observation in requests:
                assert stream % 4 == observation.player_id
                assert self.streams.setdefault(stream, env) is env
            self.requests.extend(
                (obs.player_id, len(env.mjai_log)) for _, env, obs in requests
            )
            return [
                min(
                    observation.legal_actions(),
                    key=lambda action: (
                        action.action_type != riichienv.ActionType.PASS,
                        action.action_type not in actions.WIN_ACTIONS,
                        action.action_type != riichienv.ActionType.DISCARD,
                    ),
                )
                for _, _, observation in requests
            ]

    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    state = replay.Replay("compound", events)
    checked = set()
    for _, actor, phase, _, _ in state.policy_decisions():
        if phase not in {"RIICHI", "RESPONSE"} or phase in checked:
            continue
        packed = features.pack(
            state, actor, phase, state.legal_observation(actor)
        )
        _, request = actions.policy_request(
            state, actor, state.legal_observation(actor)
        )
        candidate = next(
            row
            for row in actions.policy_candidates(request)
            if row[2] is not None
            and row[2].action_type
            in {
                riichienv.ActionType.RIICHI,
                riichienv.ActionType.CHI,
                riichienv.ActionType.PON,
            }
        )
        cut = candidate[3]["discard_action"]
        identifier = candidate[1] + (37 if phase == "RIICHI" else 111)
        alternative = cut + 37 if phase == "RIICHI" else 111
        root = {
            "state": {"events": list(state.history), "actor": actor},
            "packed": packed,
            "mark": {
                "nominations": {"actor": identifier, "expert": alternative}
            },
        }
        native, _ = collect_rollouts.recover(
            collect_rollouts.prepare(root["state"]), 941
        )
        policy = PassivePolicy()
        trials = collect_rollouts.trial(root, [941, 942], policy)
        results = trials[941]
        assert (actor, len(native.mjai_log)) not in policy.requests
        assert set(results) == {identifier, alternative}
        assert any(len(batch) > 1 for batch in policy.batches)
        for seed, branches in trials.items():
            for action, result in branches.items():
                isolated = {
                    **root,
                    "mark": {"nominations": {"actor": action}},
                }
                assert (
                    collect_rollouts.trial(isolated, [seed], PassivePolicy())[
                        seed
                    ][action]
                    == result
                )
        suffix = results[identifier]["events"][len(native.mjai_log) - 1 :]
        committed = [
            event
            for event in suffix
            if event.get("actor") == actor
            and event["type"] in {"reach", "chi", "pon", "dahai"}
        ]
        assert committed[0]["type"] == ("reach" if phase == "RIICHI" else "pon")
        assert committed[1]["type"] == "dahai"
        assert shanten.mjai_tile_code(committed[1]["pai"]) == cut
        checked.add(phase)
    assert checked == {"RIICHI", "RESPONSE"}


def test_structural_nominations_cover_unanimous_riichi_and_late_keiten():
    packed = {
        "decision_phase": 3,
        "normal_self_shanten": 1,
        "chiitoitsu_self_shanten": 5,
        "kokushi_self_shanten": 5,
        "live_wall_count": 20,
        "discard_action_all_shanten": numpy.array([1, 2] + [9] * 35),
    }
    prediction = {
        "riichi_policy_probabilities": numpy.eye(74)[37].tolist(),
        "action_values": [
            {
                "action": action,
                "expected_points": [float(value)] * 4,
                "projected_st3": value,
                "immediate_ron_probabilities": risk,
            }
            for action, value, risk in (
                (37, 1, [0.1, 0, 0]),
                (38, 0, [0, 0, 0]),
                (74, 2, [0.1, 0, 0]),
            )
        ],
    }
    mark = rollout_collection.nominations(packed, prediction, {})
    assert set(mark["nominations"].values()) == {37, 74}
    # Even unanimous noten dama nominations must consider legal riichi.
    prediction["riichi_policy_probabilities"] = numpy.eye(74)[1].tolist()
    prediction["action_values"][1]["projected_st3"] = 3
    prediction["action_values"][1]["expected_points"] = [3] * 4
    mark = rollout_collection.nominations(packed, prediction, {})
    assert mark["nominations"]["actor"] == 38
    assert mark["nominations"]["riichi"] == 74
    assert mark["nominations"]["riichi_dama_37"] == 37
    packed["live_wall_count"] = 8
    mark = rollout_collection.nominations(packed, prediction, {})
    assert mark["nominations"]["keiten"] == 74
    assert mark["nominations"]["noten"] == 38
    assert set(mark["nominations"].values()) == {37, 38, 74}

    packed.update(
        decision_phase=1,
        trigger_tile_id=0,
        self_efficiency=numpy.array([1 / 8]),
        chi_discard_shanten=numpy.full((3, 37), 9),
        pon_discard_shanten=numpy.array([[1] + [9] * 36]),
    )
    prediction["response_policy_probabilities"] = numpy.eye(150)[0].tolist()
    prediction["action_values"] = [
        {**prediction["action_values"][1], "action": 111, "projected_st3": 3},
        {**prediction["action_values"][0], "action": 223},
    ]
    mark = rollout_collection.nominations(packed, prediction, {})
    assert mark["nominations"]["keiten"] == 223
    assert mark["nominations"]["noten"] == 111
