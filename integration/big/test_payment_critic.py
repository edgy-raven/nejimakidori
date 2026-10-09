"""Bonus scoring and one payment distribution are the critic contract."""

import json
import pathlib
import types

import numpy
import riichienv
import tensorflow as tf

from log_dataset import critic_data, payment_bonus
from model import (
    critic,
    export,
    inference,
    objectives,
    outcomes,
    payment_scoring,
)


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
    ctx.actual = payment_bonus.score_hand(ctx, ctx.conditions, [])
    amounts, observed, details = payment_bonus.winning_scenarios(ctx)
    numpy.testing.assert_array_equal(
        details[observed], [ctx.actual.han, ctx.actual.fu, 0]
    )
    assert payment_bonus.SCENARIOS[observed] == (2, 0, 0, 0)
    values = [
        amounts[payment_bonus.SCENARIOS.index((2, ura, 0, 0)), 0]
        for ura in range(5)
    ]
    assert values == [5200, 8000, 8000, 12000, 12000]
    assert amounts[payment_bonus.SCENARIOS.index((2, 0, 1, 0)), 0] == 8000
    assert numpy.all(amounts[payment_bonus.SCENARIOS.index((0, 0, 0, 0))] < 0)
    assert numpy.all(amounts[payment_bonus.SCENARIOS.index((2, 5, 0, 0))] < 0)

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
    ctx.actual = payment_bonus.score_hand(ctx, ctx.conditions, [])
    amounts, observed, _ = payment_bonus.winning_scenarios(ctx)
    numpy.testing.assert_array_equal(amounts[observed], [0, 4000, 0])
    assert all(
        numpy.all(amounts[index] < 0)
        for index, scenario in enumerate(payment_bonus.SCENARIOS)
        if scenario[0] >= 2
    )
    for fixture, key, payer, winner, points in (
        ("payment_pao.json", "S3-2", 0, 1, 32000),
        ("payment_dealer_pao.json", "S1-0", 1, 0, 48000),
    ):
        pao = payment_bonus.round_labels(
            json.loads(
                pathlib.Path(__file__)
                .parent.parent.joinpath("fixtures/model")
                .joinpath(fixture)
                .read_text()
            )
        )[key]
        assert pao["payment"][1, payer, winner] == points
        assert numpy.count_nonzero(pao["payment"]) == 1
        scenario = pao["bonus"][1, payer, winner]
        assert scenario >= 0
        assert (
            outcomes.PAYMENT_VALUES[pao["scenario"][1, payer, winner, scenario]]
            == points
        )

    # Hidden indicators cannot add han without riichi.
    ctx.tiles = riichienv.parse_hand("234m234p234678s55z")[0]
    ctx.melds = []
    ctx.agari_tile = 100
    ctx.dora_indicators = []
    ctx.conditions = riichienv.Conditions(
        player_wind=riichienv.Wind.South, round_wind=0
    )
    without_ura = payment_bonus.score_hand(ctx, ctx.conditions, [])
    ignored_ura = payment_bonus.score_hand(ctx, ctx.conditions, [0])
    assert ignored_ura.han == without_ura.han
    assert ignored_ura.ron_agari == without_ura.ron_agari

    # Named double yakuman count once; separate yakuman still stack.
    for hand, tile, expected in (
        ("119m19p19s1234567z", 0, 32000),
        ("11122233344455z", 124, 96000),
    ):
        ctx.tiles = riichienv.parse_hand(hand)[0]
        ctx.agari_tile = tile
        score = payment_bonus.score_hand(ctx, ctx.conditions, [])
        assert score.is_win and score.yakuman
        assert score.ron_agari == expected


def test_sparse_score_payments_preserve_probabilities_and_gradients():
    # Immediate ron uses three cells; continuation uses all 24. Both must
    # match the original dense map, including many scores sharing an amount.
    for cells in (3, 24):
        logits = tf.Variable(
            tf.reshape(
                tf.linspace(
                    -4.0, 4.0, 4 * cells * 2 * len(payment_scoring.SCORES)
                ),
                (4, cells, 2, len(payment_scoring.SCORES)),
            )
        )
        matches = tf.constant(
            payment_scoring.payment_amounts()[:, :cells, :, None]
            == outcomes.PAYMENT_VALUES
        )
        with tf.GradientTape(persistent=True) as tape:
            probability = tf.nn.softmax(
                tf.where(tf.reduce_any(matches, -1)[:, :, None], logits, -1e9)
            )
            sparse = payment_scoring.score_payments(probability, tf.range(4))
            dense = tf.einsum(
                "bhrs,bhsk->bhrk", probability, tf.cast(matches, tf.float32)
            )
            sparse_loss = tf.reduce_sum(tf.square(sparse))
            dense_loss = tf.reduce_sum(tf.square(dense))
        numpy.testing.assert_allclose(sparse, dense, atol=1e-7, rtol=1e-6)
        numpy.testing.assert_allclose(
            tape.gradient(sparse_loss, logits),
            tape.gradient(dense_loss, logits),
            atol=1e-7,
            rtol=1e-6,
        )
        del tape


def test_same_hand_dealer_90fu_dama_maps_to_payment_tensor():
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model")
        .joinpath("payment_dealer_90fu.json")
        .read_text()
    )
    key, ctx = next(payment_bonus.win_contexts(events))
    assert (ctx.actual.han, ctx.actual.fu, ctx.actual.ron_agari) == (
        3,
        90,
        12000,
    )
    target = payment_bonus.round_labels(events)[key]
    scenario = payment_bonus.SCENARIOS.index((1, 0, 0, 0))
    cell = next(
        cell for cell in payment_bonus.CELLS if target["payment"][cell] > 0
    )
    payment_class = target["scenario"][cell + (scenario,)]
    score_class = target["score"][cell + (scenario,)]
    assert outcomes.PAYMENT_VALUES[payment_class] == 4400
    assert payment_scoring.SCORES[score_class] == (1, 90)
    cell_index = payment_bonus.CELLS.index(cell)
    assert payment_scoring.payment_amounts()[0, cell_index, score_class] == 4400


def test_robbed_kan_preserves_ippatsu_in_scoring_labels():
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model")
        .joinpath("payment_chankan_ippatsu.json")
        .read_text()
    )
    key, ctx = next(payment_bonus.win_contexts(events))
    assert key == "S4-0"
    assert ctx.conditions.chankan and ctx.conditions.ippatsu
    assert (ctx.actual.han, ctx.actual.fu, ctx.actual.ron_agari) == (
        8,
        40,
        24000,
    )
    target = payment_bonus.round_labels(events)[key]
    scenario = target["bonus"][0, 2, 3]
    assert payment_bonus.SCENARIOS[scenario] == (2, 1, 1, 3)
    assert target["payment"][0, 2, 3] == 24000
    assert (
        outcomes.PAYMENT_VALUES[target["scenario"][0, 2, 3, scenario]] == 24000
    )


def test_joint_training_checkpoint_export_and_placement_inference(
    tmp_path,
    monkeypatch,
):
    from log_dataset import afk, critic_data, replay
    from model import (
        critic,
        critic_full,
        critic_utility,
        decision_layers,
        joint_settlement,
    )

    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model")
        .joinpath("payment_dealer_90fu.json")
        .read_text()
    )
    targets = payment_bonus.round_labels(events)
    observations = list(
        replay.Replay(
            "joint-policy",
            afk.retained_hands(events)[0],
            hanchan=replay.hanchan_targets(events),
        ).observations()
    )[:8]
    inputs = critic_data.parse_batch(
        tf.constant(
            [
                critic_data.serialize(row, targets[row.kyoku_id])
                for row in observations
            ]
        )
    )
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
    network = critic.JointPolicyPayment()
    network(inputs)
    network.compile_training()
    with tf.GradientTape(persistent=True) as tape:
        outputs = network(inputs)
        advantages = critic_utility.advantages(
            network,
            inputs,
            outputs,
            {
                "temperature": 3.0,
            },
        )
        loss, metrics = critic_full.objective_metrics(
            inputs,
            outputs,
            utility={
                "advantages": advantages,
                "enabled": True,
                "mix": 0.8,
            },
        )
    # Sharing position payoffs must preserve every action's AWR advantage.
    # This batch also exercises both padded position and action chunks.
    active = critic_utility.active_candidates(inputs[0], outputs["selected"])
    selected = tf.boolean_mask(outputs["selected"], active)
    candidate_inputs = {
        "state": tf.boolean_mask(outputs["critic_candidate_state"], active),
        "immediate_safe": tf.boolean_mask(outputs["immediate_safe"], active),
        "immediate_wait": tf.boolean_mask(outputs["immediate_wait"], active),
        "immediate_allowed": critic.immediate_decisions(inputs[0], selected),
        **{
            name: tf.gather(value, selected[:, 0])
            for name, value in joint_settlement.public_inputs(inputs[0]).items()
        },
        **critic.public_inputs(inputs[0], selected),
    }

    @tf.function(jit_compile=True)
    def reference_utilities(values):
        return joint_settlement.projected_st3(
            network.critic.joint_distributions(values), values
        )

    reference = tf.concat(
        [
            reference_utilities(
                {
                    name: value[begin : begin + 16]
                    for name, value in candidate_inputs.items()
                }
            )
            for begin in range(0, selected.shape[0], 16)
        ],
        0,
    )
    baseline = tf.math.unsorted_segment_sum(
        reference
        * tf.gather_nd(
            critic_utility.candidate_probabilities(inputs[0], outputs), selected
        ),
        selected[:, 0],
        len(observations),
    )
    recorded = (
        tf.gather(critic.chosen_actions(inputs[0], inputs[1]), selected[:, 0])
        == selected[:, 1]
    )
    numpy.testing.assert_allclose(
        advantages,
        (tf.boolean_mask(reference, recorded) - baseline) / 3,
        atol=2e-5,
        rtol=2e-5,
    )
    assert tape.gradient(advantages, network.trainable_variables) == [
        None
    ] * len(network.trainable_variables)
    assert all(numpy.isfinite(float(value)) for value in metrics.values())
    assert 0 < float(metrics["utility_blended_effective_sample_fraction"]) <= 1
    # Full-batch critic supervision includes every policy position.
    assert float(
        outputs["critic_statistics"][
            critic.CRITIC_METRICS.index("settlement_supported_fraction"), 1
        ]
    ) == len(inputs[2]["points"])
    # Whole-validation aggregation must match a single batch even when
    # completion labels occur only in a short, separate batch.
    validation_inputs = (
        inputs[0],
        {
            **inputs[1],
            "round_completion_score": tf.where(
                tf.range(tf.shape(inputs[2]["points"])[0])[:, None, None] == 0,
                inputs[1]["round_completion_score"],
                -1,
            ),
        },
        inputs[2],
    )
    totals, predictions = [], []
    for part in (slice(0, 1), slice(1, None)):
        batch = tf.nest.map_structure(
            lambda value: value[part], validation_inputs
        )
        prediction = network(batch)
        totals.append(critic_full.validation_statistics(batch, prediction))
        predictions.append(
            {
                **prediction,
                "selected": prediction["selected"] + [part.start, 0],
                "immediate_rows": prediction["immediate_rows"] + part.start,
            }
        )
    # Compare aggregation with fixed predictions. Separate BF16 forward
    # passes need not round identically at different batch shapes.
    validation_outputs = {
        name: (
            tf.add_n([row[name] for row in predictions])
            if name == "critic_statistics"
            else tf.nest.map_structure(
                lambda *values: tf.concat(values, axis=0),
                *[row[name] for row in predictions],
            )
        )
        for name in outputs
    }
    combined = critic_full.objective_metrics(
        validation_inputs,
        validation_outputs,
        statistics=tf.add_n([row["statistics"] for row in totals]),
    )[1]
    expected = critic_full.objective_metrics(
        validation_inputs, validation_outputs
    )[1]
    for name in (
        "completion_score_loss",
        "score_nll",
        "belief_loss",
        "specialist_loss",
        "critic_loss",
        "joint_loss",
    ):
        numpy.testing.assert_allclose(combined[name], expected[name], atol=2e-5)
    numpy.testing.assert_allclose(
        sum(float(row["counts"]["example_count"]) for row in totals),
        len(observations),
    )
    variables = network.trainable_variables
    gradients = tape.gradient(loss, variables)
    # Scored han/fu supervision reaches both kinds of yaku evidence.
    assert metrics["scoring_loss"] > 0
    assert metrics["completion_score_loss"] > 0
    for name in (
        "attack_yaku_logits",
        "round_completion_yaku_logits",
        "round_completion_score_logits",
    ):
        scoring_gradient = tape.gradient(metrics["scoring_loss"], outputs[name])
        assert scoring_gradient is not None
        assert numpy.isfinite(scoring_gradient).all()
        assert tf.linalg.global_norm([scoring_gradient]) > 0
    with tf.GradientTape() as unknown_tape:
        unknown_outputs = network(inputs)
        _, unknown_metrics = critic_full.objective_terms(
            (
                inputs[0],
                {
                    **inputs[1],
                    "round_completion_score": -tf.ones_like(
                        inputs[1]["round_completion_score"]
                    ),
                },
                inputs[2],
            ),
            unknown_outputs,
        )
    assert unknown_metrics["completion_score_loss"] == 0
    numpy.testing.assert_array_equal(
        unknown_tape.gradient(
            unknown_metrics["completion_score_loss"],
            unknown_outputs["round_completion_score_logits"],
        ),
        0,
    )
    del tape
    assert all(value is not None for value in gradients), [
        variable.path
        for variable, value in zip(variables, gradients)
        if value is None
    ]
    assert all(
        numpy.isfinite(tf.convert_to_tensor(value)).all() for value in gradients
    )
    assert float(tf.linalg.global_norm(gradients)) > 0
    # Policy inference needs visible features only, never outcome/action labels.
    policy = network.actor(inputs[0])
    for name, mask in decision_layers.legal_choices(inputs[0]).items():
        numpy.testing.assert_array_equal(policy[name], outputs[name])
        active = numpy.any(mask, axis=-1)
        assert numpy.all(
            tf.nn.softmax(policy[name]).numpy()[active][~mask.numpy()[active]]
            == 0
        )
    assert numpy.isfinite(float(metrics["policy_loss"]))
    before = network.beliefs(inputs[0])
    optimizer = critic_full.FusedAdamW(learning_rate=1e-4, clipnorm=1.0)
    optimizer.apply_gradients(zip(gradients, variables))
    assert int(optimizer.iterations) == 1
    for name, expected in before.items():
        assert numpy.any(
            network.beliefs(inputs[0])[name].numpy() != expected.numpy()
        )
    outputs = network(inputs)
    saved = tmp_path / "joint.weights.h5"
    network.save_weights(saved)
    restored = critic.JointPolicyPayment()
    restored(inputs)
    restored.compile_training()
    restored(inputs)
    restored.load_weights(saved)
    for actual, expected in zip(restored.get_weights(), network.get_weights()):
        numpy.testing.assert_array_equal(actual, expected)
    for name in ("discard_policy_logits", "critic_statistics"):
        numpy.testing.assert_allclose(
            restored(inputs)[name], outputs[name], atol=1e-7, rtol=1e-6
        )
    serving = export.ServingModel(restored)
    expected = serving.serve(**inputs[0])
    numpy.testing.assert_allclose(
        tf.reduce_sum(expected["final_placement_probabilities"], -1),
        1,
        atol=1e-5,
    )
    assert numpy.all(expected["final_placement_probabilities"] >= 0)
    assert {
        "final_placement_logits",
        "attack_policy_logits",
        "defense_policy_logits",
    }.isdisjoint(expected)
    for head in ("attack", "defense"):
        for phase, size in (("discard", 37), ("riichi", 74)):
            values = expected[f"{head}_{phase}_logits"]
            assert values.shape == expected[f"{phase}_policy_logits"].shape
            assert values.shape[-1] == size
            numpy.testing.assert_array_equal(
                values > -1e3,
                expected[f"{phase}_policy_logits"] > -1e3,
            )
    monkeypatch.setattr(
        critic_data,
        "dataset",
        lambda **kwargs: [
            tf.constant(
                [
                    critic_data.serialize(row, targets[row.kyoku_id])
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
    assert (tmp_path / "release/source/model/feature_vector.py").is_file()
    assert json.loads(
        (tmp_path / "release/export-verification.json").read_text()
    )["passed"]
    exported = tf.saved_model.load(
        str(tmp_path / "release/saved_model")
    ).signatures["serving_default"](**inputs[0])
    for name in expected:
        numpy.testing.assert_allclose(
            exported[name], expected[name], atol=1e-6, rtol=1e-5
        )

    predictor = inference.Predictor(tmp_path / "release/saved_model")
    prediction = predictor.infer(
        [
            {
                name: value[row].numpy().tolist()
                for name, value in inputs[0].items()
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
            numpy.isfinite(action["projected_st3"])
            for action in row["action_values"]
        )


def test_compact_score_targets_preserve_counterfactual_loss_and_gradients():
    labels = []
    cells = numpy.array(payment_bonus.CELLS[:24])
    for fixture in (
        "payment_dealer_90fu.json",
        "payment_chankan_ippatsu.json",
        "payment_pao.json",
    ):
        events = json.loads(
            pathlib.Path(__file__)
            .parent.parent.joinpath("fixtures/model", fixture)
            .read_text()
        )
        for target in payment_bonus.round_labels(events).values():
            labels.append(
                target["score"][cells[:, 0], cells[:, 1], cells[:, 2]]
            )
    labels.append(numpy.full_like(labels[0], -1))
    labels = tf.constant(numpy.stack(labels), tf.int32)
    compact = critic_data.score_targets(labels)
    numpy.testing.assert_array_equal(
        tf.where(
            labels >= 0,
            tf.gather(compact["score"], payment_bonus.SCORE_GROUPS, axis=2),
            -1,
        ),
        labels,
    )
    logits = tf.Variable(
        tf.random.stateless_normal(
            (*compact["score"].shape, len(payment_scoring.SCORES)), [9, 41]
        )
    )
    with tf.GradientTape(persistent=True) as tape:
        actual = critic.score_cross_entropy(
            compact["score"], compact["score_weight"], logits
        )
        expanded = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.maximum(labels, 0),
            logits=tf.gather(logits, payment_bonus.SCORE_GROUPS, axis=2),
        )
        valid = tf.cast(labels >= 0, tf.float32)
        expected = objectives.mean_valid(
            tf.math.divide_no_nan(
                tf.reduce_sum(expanded * valid, -1), tf.reduce_sum(valid, -1)
            ),
            tf.reduce_any(labels >= 0, -1),
        )
    numpy.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    numpy.testing.assert_allclose(
        tape.gradient(actual, logits),
        tape.gradient(expected, logits),
        rtol=1e-6,
        atol=1e-7,
    )
