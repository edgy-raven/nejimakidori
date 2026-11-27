"""Recorded training rows follow the choices available to the policy."""

import json
import pathlib

import pytest
import riichienv

from log_dataset import replay


def test_dama_keeps_its_discard_and_riichi_keeps_only_optional_actions():
    path = pathlib.Path(__file__).parents[1] / "fixtures/model"
    events = json.loads((path / "payment_dealer_90fu.json").read_bytes())
    state = replay.Replay("dealer-90fu", events)
    decisions = set()
    declarations = []
    for event, seat, phase, following, _ in state.policy_decisions():
        decisions.add((event, seat, phase))
        if phase == "RIICHI":
            declarations.append((event, seat, following))
        if phase == "DISCARD":
            assert not state.declared[seat]

    assert {events[i]["type"] for _, _, i in declarations} == {"reach", "dahai"}
    for event, seat, following in declarations:
        if events[following]["type"] == "reach":
            assert (following, seat, "DISCARD") not in decisions
            assert (event, seat, "DISCARD") not in decisions
        else:
            assert (event, seat, "DISCARD") in decisions
    # The dealer can still choose a closed kan, then a win, after declaring.
    assert {(119, 0, "KAN"), (127, 0, "WIN")} <= decisions


def test_declining_ron_after_riichi_does_not_create_more_win_labels():
    path = pathlib.Path(__file__).parents[1] / "fixtures/model"
    fixture = json.loads((path / "declined_riichi_ron.json").read_bytes())
    state = replay.Replay(fixture["source"], fixture["events"])
    for index, _ in state.replay_events():
        if index not in {fixture["legal_ron"], fixture["furiten_ron"]}:
            continue
        choices = replay.recorded_win_choices(state, index, index + 1)
        legal = state.legal_observation(1).legal_actions()
        if index == fixture["legal_ron"]:
            assert choices[1] == (0, index)
            assert any(a.action_type == riichienv.ActionType.RON for a in legal)
        else:
            assert 1 not in choices
            assert all(a.action_type != riichienv.ActionType.RON for a in legal)


@pytest.mark.parametrize("archive_draw", [False, True])
def test_last_discard_ron_pass_survives_exhaustive_draw(archive_draw):
    path = pathlib.Path(__file__).parents[1] / "fixtures/model"
    events = json.loads((path / "last_discard_ron_pass.json").read_bytes())
    if archive_draw:
        for event in events:
            if event["type"] == "ryukyoku":
                del event["reason"]
    state = replay.Replay("2163021929-2-S4", events)
    for index, _ in state.replay_events():
        if index == len(events) - 3:
            assert replay.recorded_win_choices(state, index, index + 1)[0] == (
                0,
                index,
            )


def test_own_kan_has_no_phantom_self_win_before_the_rinshan_draw():
    from log_dataset import root_states

    # 2024052311gm-00a9-0000-64da92b0, South 4: the player declines
    # tsumo and declares kakan. Only opponents can win before rinshan.
    path = pathlib.Path(__file__).parents[1] / "fixtures/model"
    events = json.loads((path / "own_kakan_win_pass.json").read_bytes())
    decisions = list(replay.Replay("own-kan", events).policy_decisions())
    assert [
        (event, seat)
        for event, seat, phase, _, _ in decisions
        if phase == "WIN"
    ] == [(86, 0), (121, 0), (141, 2)]
    observations = list(replay.Replay("own-kan", events).observations())
    roots = root_states.prepare_hand("own-kan", events)
    assert set(roots) == {
        (row.decision_index, row.actor, int(row.features["decision_phase"]))
        for row in observations
    }
    # The former extra WIN row is now the actual post-rinshan discard.
    assert (86, 0, 0) in roots
    assert (86, 0, 4) not in roots
