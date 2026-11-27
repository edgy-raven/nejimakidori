"""Bonus scoring and one payment distribution are the critic contract."""

import json
import pathlib
import types

import numpy
import pytest
import riichienv
import tensorflow as tf

from log_dataset import _native_collection, call_states, scoring
from model import actions, critic, export, inference
from model import network as joint_network
from model import outcomes, payment_scoring


def test_score_states_are_canonical_and_respect_win_methods():
    expected = {
        1: tuple(range(30, 111, 10)),
        2: (20, 25, *range(30, 111, 10)),
        3: (20, 25, 30, 40, 50, 60),
        4: (20, 25, 30),
    }
    assert payment_scoring.SCORES == (
        tuple((han, fu) for han, values in expected.items() for fu in values)
        + ((5, 0), (6, 0), (8, 0), (11, 0), (13, 0))
        + ((0, 1), (0, 2), (0, 3), (0, 4))
    )
    for han, fu, target in (
        (3, 60, (3, 60)),
        (4, 30, (4, 30)),
        (3, 70, (5, 0)),
        (4, 40, (5, 0)),
        (5, 20, (5, 0)),
        (7, 30, (6, 0)),
        (10, 30, (8, 0)),
        (12, 30, (11, 0)),
        (26, 30, (13, 0)),
    ):
        assert (
            payment_scoring.SCORES[payment_scoring.score_class(han, fu, False)]
            == target
        )
    assert payment_scoring.SCORES[payment_scoring.score_class(26, 0, True)] == (
        0,
        2,
    )
    for han, fu in ((1, 20), (1, 25), (1, 120), (2, 120)):
        with pytest.raises(ValueError):
            payment_scoring.score_class(han, fu, False)
    legal = payment_scoring.LEGAL_SCORE_STATES.reshape(4, 129, 3, -1)
    for score, ron, tsumo in (
        ((2, 20), False, True),
        ((2, 25), True, False),
        ((3, 25), True, True),
        ((1, 110), True, False),
        ((2, 110), True, True),
    ):
        state = payment_scoring.SCORES.index(score)
        assert bool(legal[2, 0, 1, state]) == ron
        assert bool(legal[0, 0, 1, state]) == tsumo


def test_same_hand_scoring_separates_bonuses_and_respects_limits():
    ctx = types.SimpleNamespace(
        tiles=riichienv.parse_hand("234m234p234678s55z")[0],
        melds=[],
        agari_tile=100,
        dora_indicators=[],
        ura_indicators=[],
        conditions=riichienv.Conditions(
            riichi=True, player_wind=riichienv.Wind.South, round_wind=0
        ),
    )
    ctx.actual = _native_collection.score_hand(ctx, ctx.conditions, [])
    amounts, observed, details = scoring.bonus_han_scores(ctx)
    numpy.testing.assert_array_equal(
        details[observed], [ctx.actual.han, ctx.actual.fu, 0]
    )
    assert observed == 1
    assert amounts[1:6, 0].tolist() == [5200, 8000, 8000, 12000, 12000]
    assert amounts[13, 0] == 32000
    # Different bonus sources with equal han collapse to the same label.
    for conditions, ura in (
        (
            riichienv.Conditions(
                riichi=True, ippatsu=True, player_wind=riichienv.Wind.South
            ),
            [],
        ),
        (
            riichienv.Conditions(riichi=True, player_wind=riichienv.Wind.South),
            [0],
        ),
    ):
        ctx.conditions = conditions
        ctx.actual = _native_collection.score_hand(ctx, conditions, ura)
        same_amounts, bonus, _ = scoring.bonus_han_scores(ctx)
        assert bonus == 2
        numpy.testing.assert_array_equal(same_amounts, amounts)

    # A valid hand with two kans must remain scorable. This also exercises
    # repeated physical IDs from the replay's native meld representation.
    ctx.tiles = [24, 25, 84, 85, 86]
    ctx.melds = [
        riichienv.Meld(riichienv.MeldType.Chi, [53, 48, 56], True),
        riichienv.Meld(riichienv.MeldType.Kakan, [100] * 4, True),
        riichienv.Meld(riichienv.MeldType.Ankan, [96] * 4, False),
    ]
    ctx.agari_tile = 84
    ctx.dora_indicators = [80]
    ctx.conditions = riichienv.Conditions(
        tsumo=True, player_wind=riichienv.Wind.East, round_wind=0
    )
    ctx.actual = _native_collection.score_hand(ctx, ctx.conditions, [])
    amounts, observed, _ = scoring.bonus_han_scores(ctx)
    numpy.testing.assert_array_equal(amounts[observed], [0, 4000, 0])
    assert numpy.all(amounts[2:] < 0)
    for fixture, key, payer, winner, points in (
        ("payment_pao.json", "S3-2", 0, 1, 32000),
        ("payment_dealer_pao.json", "S1-0", 1, 0, 48000),
    ):
        pao = scoring.round_labels(
            json.loads(
                pathlib.Path(__file__)
                .parent.parent.joinpath("fixtures/model")
                .joinpath(fixture)
                .read_text()
            )
        )[key]
        assert pao["payment"][1, payer, winner] == points
        assert numpy.count_nonzero(pao["payment"]) == 1
        assert pao["bonus"][1, payer, winner] == -1
        assert all(
            payment_scoring.SCORES[index][0] == 0
            for index in pao["score"][1, payer, winner]
            if index >= 0
        )

    # Hidden indicators cannot add han without riichi.
    ctx.tiles = riichienv.parse_hand("234m234p234678s55z")[0]
    ctx.melds = []
    ctx.agari_tile = 100
    ctx.dora_indicators = []
    ctx.conditions = riichienv.Conditions(
        player_wind=riichienv.Wind.South, round_wind=0
    )
    without_ura = _native_collection.score_hand(ctx, ctx.conditions, [])
    ignored_ura = _native_collection.score_hand(ctx, ctx.conditions, [0])
    assert ignored_ura.han == without_ura.han
    assert ignored_ura.ron_agari == without_ura.ron_agari

    # Named double yakuman count once; separate yakuman still stack.
    for hand, tile, expected in (
        ("119m19p19s1234567z", 0, 32000),
        ("11122233344455z", 124, 96000),
    ):
        ctx.tiles = riichienv.parse_hand(hand)[0]
        ctx.agari_tile = tile
        score = _native_collection.score_hand(ctx, ctx.conditions, [])
        assert score.is_win and score.yakuman
        assert score.ron_agari == expected


def test_robbed_kan_preserves_ippatsu_in_scoring_labels():
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model")
        .joinpath("payment_chankan_ippatsu.json")
        .read_text()
    )
    key, ctx = next(scoring.win_contexts(events))
    assert key == "S4-0"
    assert ctx.conditions.chankan and ctx.conditions.ippatsu
    assert (ctx.actual.han, ctx.actual.fu, ctx.actual.ron_agari) == (
        8,
        40,
        24000,
    )
    target = scoring.round_labels(events)[key]
    scenario = target["bonus"][0, 2, 3]
    assert scenario == 4
    assert target["payment"][0, 2, 3] == 24000
    assert payment_scoring.SCORES[target["score"][0, 2, 3, scenario]] == (8, 0)


def test_checkpoint_export_and_placement_inference(
    tmp_path,
    monkeypatch,
):
    from log_dataset import afk, replay, training_data
    from model import critic

    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model")
        .joinpath("payment_dealer_90fu.json")
        .read_text()
    )
    targets = scoring.round_labels(events + [{"type": "end_game"}])
    key, ctx = next(scoring.win_contexts(events))
    assert (ctx.actual.han, ctx.actual.fu, ctx.actual.ron_agari) == (
        3,
        90,
        12000,
    )
    # Removing bonuses preserves the dealer's unusual 90-fu payment.
    cell = next(cell for cell in scoring.CELLS if targets[key]["payment"][cell])
    score_class = targets[key]["score"][cell + (0,)]
    assert payment_scoring.SCORES[score_class] == (1, 90)
    assert (
        payment_scoring.payment_amounts()[
            0, scoring.CELLS.index(cell), score_class
        ]
        == 4400
    )
    observations = list(
        replay.Replay(
            "joint-policy",
            afk.retained_hands(events)[0],
        ).observations()
    )[:8]
    inputs = training_data.parse_batch(
        tf.constant(
            [
                training_data.serialize(row, targets[row.kyoku_id])
                for row in observations
            ]
        )
    )
    assert inputs[2]["score"].shape[-1] == len(scoring.BONUS_HAN)
    assert inputs[2]["bonus"].shape == (len(observations), 24)
    # Ron and tsumo yaku/value share both their witness and unknown mask.
    # Non-winning seats receive labels when their replay reaches tenpai.
    for row in observations:
        numpy.testing.assert_array_equal(
            numpy.all(row.round_completion_yaku >= 0, axis=-1),
            row.round_completion_score >= 0,
        )
    assert any(
        (
            numpy.delete(
                row.round_completion_score, (0 - row.actor) % 4, axis=0
            )
            >= 0
        ).any()
        for row in observations
    )
    network = joint_network.JointPolicyPayment()
    network(inputs)
    assert network.count_params() <= 19_000_000
    network.compile_training()
    outputs = network(inputs)
    assert float(
        outputs["critic_statistics"][
            critic.CRITIC_METRICS.index("settlement_supported_fraction"), 1
        ]
    ) == len(observations)
    policy = network.actor(inputs[0])
    for name, mask in actions.legal_choices(inputs[0]).items():
        numpy.testing.assert_array_equal(policy[name], outputs[name])
        active = numpy.any(mask, axis=-1)
        assert numpy.all(
            tf.nn.softmax(policy[name]).numpy()[active][~mask.numpy()[active]]
            == 0
        )
    saved = tmp_path / "joint.weights.h5"
    network.save_weights(saved)
    restored = joint_network.JointPolicyPayment()
    restored(inputs)
    restored.compile_training()
    restored.load_weights(saved)
    for actual, expected in zip(restored.get_weights(), network.get_weights()):
        numpy.testing.assert_array_equal(actual, expected)
    restored_outputs = restored(inputs)
    for name in ("discard_policy_logits", "critic_statistics"):
        numpy.testing.assert_allclose(
            restored_outputs[name], outputs[name], atol=1e-7, rtol=1e-6
        )
    serving = export.ServingModel(restored)
    expected = serving.serve(**inputs[0])
    assert set(expected) == set(actions.SERVING_ACTOR_OUTPUTS) | {
        "terminal_payment_probabilities",
        "payment_values",
        "final_placement_probabilities",
        "review_outcome_probabilities",
        "settlement_mean",
        "settlement_variance",
        "critic_expected_points",
        "critic_action_indices",
    }

    assert "state_value" not in restored_outputs
    numpy.testing.assert_allclose(
        tf.reduce_sum(expected["final_placement_probabilities"], -1),
        1,
        atol=1e-5,
    )
    assert numpy.all(expected["final_placement_probabilities"] >= 0)
    monkeypatch.setattr(
        training_data,
        "dataset",
        lambda **kwargs: [
            tf.constant(
                [
                    training_data.serialize(row, targets[row.kyoku_id])
                    for row in observations
                ]
            )
        ],
    )
    export.run(
        types.SimpleNamespace(
            weights=str(saved),
            data=str(tmp_path / "data"),
            output=str(tmp_path / "release"),
        )
    )
    assert (tmp_path / "release/source/log_dataset/features.py").is_file()
    assert (tmp_path / "release/source/requirements.txt").read_bytes() == (
        pathlib.Path("requirements.txt").read_bytes()
    )
    verification = json.loads(
        (tmp_path / "release/export-verification.json").read_text()
    )
    assert verification["passed"]
    assert verification["weights"] == saved.name
    assert "data" not in verification
    assert str(tmp_path) not in json.dumps(verification)
    exported = tf.saved_model.load(
        str(tmp_path / "release/saved_model")
    ).signatures["serving_default"](**inputs[0])
    for name in expected:
        numpy.testing.assert_allclose(
            exported[name], expected[name], atol=1e-6, rtol=1e-5
        )

    predictor = inference.Predictor(tmp_path / "release/saved_model")
    win = next(
        row
        for row in replay.Replay("win-inference", events).observations()
        if row.win_action >= 0
    )
    win_inputs = training_data.parse_batch(
        tf.constant([training_data.serialize(win, targets[win.kyoku_id])])
    )[0]
    win_prediction = predictor.infer(
        [
            {
                name: (
                    value.numpy().tolist()
                    if name.startswith("post_call_")
                    else value[0].numpy().tolist()
                )
                for name, value in win_inputs.items()
            }
        ]
    )
    numpy.testing.assert_allclose(
        win_prediction.predictions[0]["win_decision_probabilities"],
        tf.nn.softmax(
            predictor.module.signatures["policy"](**win_inputs)[
                "win_decision_logits"
            ]
        )[0],
        atol=1e-6,
    )
    prediction = predictor.infer(
        [
            {
                name: (
                    value.tolist()
                    if name.startswith("post_call_")
                    else value[0].tolist()
                )
                for name, value in call_states.select(
                    {key: value.numpy() for key, value in inputs[0].items()},
                    [row],
                ).items()
            }
            for row in range(len(observations))
        ]
    )
    assert len(prediction.predictions) == len(observations)
    numpy.testing.assert_allclose(
        [
            row["final_placement_probabilities"]
            for row in prediction.predictions
        ],
        exported["final_placement_probabilities"],
        atol=1e-6,
    )
    for row in prediction.predictions:
        assert (
            len(row["structured_outcome"]["values"])
            == len(outcomes.PAYMENT_VALUES) + 4
        )
        assert (
            row["structured_outcome"]["uncertainty"] == "joint hand settlement"
        )
        assert all(
            numpy.isfinite(action["expected_points"]).all()
            for action in row["action_values"]
        )
