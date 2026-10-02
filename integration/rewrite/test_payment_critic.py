"""Bonus scoring and one payment distribution are the critic contract."""

import json
import pathlib
import types

import mahjong.tile
import numpy
import riichienv
import tensorflow as tf

import export_policy
from model import critic, outcomes, payment_bonus, payment_scoring


def test_public_safety_context_tracks_tile_action_and_relative_seat():
    inputs = {
        "riichi_state": tf.constant([[0, 2, 0, 0]]),
        "ippatsu_alive": tf.constant([[0, 1, 0, 0]]),
        "opponent_called_tile_count": tf.constant([[0, 0, 3, 0]]),
        "opponent_discard_count": tf.constant([[8, 9, 8, 8]]),
        "riichi_push_count": tf.tensor_scatter_nd_update(
            tf.zeros((1, 4, 4), tf.int32), [[0, 2, 1]], [3]
        ),
        "dealer_relative_seat": tf.constant([2]),
        "genbutsu_to_seat": tf.tensor_scatter_nd_update(
            tf.zeros((1, 4, 34), tf.int32), [[0, 1, 4]], [1]
        ),
        "sotogawa_earliest_turn_to_seat": tf.zeros((1, 4, 34)),
        "passed_unchanged_to_seat": tf.tensor_scatter_nd_update(
            tf.zeros((1, 4, 34), tf.int32), [[0, 2, 4]], [1]
        ),
        "blocker_counts": tf.zeros((1, 34, 5)),
        "known_unavailable_counts": tf.zeros((1, 34)),
        "dora_multiplicity": tf.zeros((1, 34)),
    }
    # Normal five, red five, another discard, and a response pass.
    context = critic.PublicContext(dtype="float32")(
        [
            inputs,
            tf.constant([[0, 4], [0, 34], [0, 5], [0, 111]]),
            tf.constant([[4.0], [34.0], [5.0], [37.0]]),
        ]
    ).numpy()
    assert context.shape == (4, 113)
    numpy.testing.assert_array_equal(context[:, 106], [3 / 24] * 4)
    numpy.testing.assert_array_equal(context[:, 95], [1, 1, 0, 0])
    numpy.testing.assert_array_equal(context[0, 38:], context[1, 38:])
    assert context[0, 4] == context[1, 34] == 1
    numpy.testing.assert_array_equal(context[:, 67], [1, 1, 0, 0])
    numpy.testing.assert_array_equal(context[3, 66:81], numpy.zeros(15))
    assert context[0, 38] == context[3, 41] == 1
    numpy.testing.assert_allclose(context[:, 81:85], [[1, 16 / 18, 1, 1]] * 4)
    numpy.testing.assert_array_equal(context[0, 85:89], [1, 0, 1, 1])
    numpy.testing.assert_array_equal(context[3, 85:97], numpy.zeros(12))
    inputs["genbutsu_to_seat"] = tf.ones((1, 4, 34))
    closed = critic.PublicContext(dtype="float32")(
        [inputs, tf.constant([[0, 27]]), tf.constant([[27.0]])]
    ).numpy()
    numpy.testing.assert_array_equal(closed[:, 81:97], numpy.zeros((1, 16)))


def test_immediate_risk_composes_with_later_payments_without_erasing_them():
    payment_model = critic.PaymentCritic()
    inputs = {
        "state": tf.zeros((2, 16)),
        "opened": tf.zeros((2, 4), tf.bool),
        "deposit_allowed": tf.zeros((2, 4), tf.bool),
        "immediate_allowed": tf.constant([True, False]),
        "dealer": tf.zeros(2, tf.int32),
        "public_fu": tf.zeros((2, 4), tf.int32),
        "immediate_safe": tf.zeros((2, 3), tf.bool),
    }
    payment_model(inputs)
    state = tf.reshape(tf.range(32, dtype=tf.float32) / 32, (2, 16))
    with tf.GradientTape(persistent=True) as tape:
        full = tf.gather(
            payment_model.encode(state), critic.DEAL_IN_CELLS, axis=1
        )
        selected = payment_model.encode(state, critic.DEAL_IN_CELLS)
        full_sum, selected_sum = tf.reduce_sum(full), tf.reduce_sum(selected)
    numpy.testing.assert_allclose(selected, full, rtol=1e-5, atol=1e-6)
    variables = (
        payment_model.state.trainable_variables
        + payment_model.cell_state.trainable_variables
    )
    for actual, expected in zip(
        tape.gradient(selected_sum, variables),
        tape.gradient(full_sum, variables),
    ):
        numpy.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    payment_model.occurrence.kernel.assign(
        tf.zeros_like(payment_model.occurrence.kernel)
    )
    payment_model.occurrence.bias.assign([-20])
    payment_model.immediate_risk.kernel.assign(
        tf.zeros_like(payment_model.immediate_risk.kernel)
    )
    payment_model.immediate_risk.bias.assign([20])
    probability = payment_model(inputs)["payment_probability"].numpy()
    numpy.testing.assert_allclose(
        probability[0, critic.DEAL_IN_CELLS, 1], 0, atol=1e-6
    )
    numpy.testing.assert_allclose(
        probability[1, critic.DEAL_IN_CELLS, 1], 1, atol=1e-6
    )
    payment_model.occurrence.bias.assign([20])
    payment_model.immediate_risk.bias.assign([-20])
    probability = payment_model(inputs)["payment_probability"].numpy()
    numpy.testing.assert_allclose(
        probability[:, critic.DEAL_IN_CELLS, 1], 0, atol=1e-6
    )
    numpy.testing.assert_allclose(probability.sum(-1), 1, atol=1e-6)


def test_joint_score_decoder_preserves_fu_exceptions_and_public_meld_fu():
    assert sum(han > 0 for han, _ in payment_scoring.SCORES) == 65
    limit_classes = []
    for han in range(5, 14):
        limit_classes.append(payment_scoring.score_class(han, 20, False, 0))
        assert all(
            payment_scoring.score_class(han, fu, False, 0) == limit_classes[-1]
            for fu in (25, *range(30, 141, 10))
        )
    limits = payment_scoring.masked_score_logits(
        logits=tf.zeros((2, 24, 3, len(payment_scoring.SCORES))),
        dealer=tf.constant([0, 1]),
        meld_fu=tf.constant([[0] * 4, [128] * 4]),
        scenarios=[(0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 4)],
    ).numpy()
    assert numpy.all(limits[:, :, :2, limit_classes] == 0)
    assert numpy.all(limits[:, :, 2, limit_classes] == -1e9)
    indices = [
        payment_scoring.score_class(han, fu, False, 0)
        for han, fu in ((1, 40), (1, 30), (2, 25))
    ]
    logits = tf.broadcast_to(
        200 * tf.one_hot(indices, len(payment_scoring.SCORES)) - 100,
        (2, 24, 3, len(payment_scoring.SCORES)),
    )
    _, probabilities = payment_scoring.payment_probabilities(
        logits,
        tf.constant([0, 1]),
        tf.zeros((2, 4), tf.int32),
        [(1, 0, 0, 0)] * 3,
    )
    cell = payment_bonus.CELLS.index((0, 1, 0))
    numpy.testing.assert_allclose(
        probabilities.numpy()[:, cell] @ outcomes.PAYMENT_VALUES,
        [[2000, 1500, 2400], [1300, 1000, 1600]],
        atol=1e-5,
    )
    numpy.testing.assert_allclose(probabilities.numpy().sum(-1), 1, atol=1e-6)
    constrained, _ = payment_scoring.payment_probabilities(
        logits, tf.constant([0, 1]), tf.fill((2, 4), 12), [(1, 0, 0, 0)] * 3
    )
    assert numpy.all(constrained.numpy()[:, cell, :, indices] == -1e9)
    opened, _ = payment_scoring.payment_probabilities(
        logits,
        tf.constant([0, 1], tf.int8),
        tf.zeros((2, 4), tf.int32),
        [(0, 0, 0, 0)] * 3,
    )
    assert numpy.all(
        opened.numpy()[
            ...,
            [
                payment_scoring.score_class(2, 20, False, 0),
                payment_scoring.score_class(2, 25, False, 0),
            ],
        ]
        == -1e9
    )
    known = payment_scoring.public_fu(
        {
            "meld_type": tf.constant([[[3], [2], [4], [1]]]),
            "meld_tile_ids": tf.constant([[[[4]], [[27]], [[8]], [[3]]]]),
            "meld_valid": tf.ones((1, 4, 1), tf.int32),
        }
    )
    numpy.testing.assert_array_equal(known, [[8, 4, 32, 0]])
    payment_model = critic.PaymentCritic()
    with tf.GradientTape() as tape:
        result = payment_model(
            {
                "state": tf.ones((2, 16)),
                "opened": tf.zeros((2, 4), tf.bool),
                "deposit_allowed": tf.zeros((2, 4), tf.bool),
                "immediate_allowed": tf.ones((2,), tf.bool),
                "immediate_safe": tf.zeros((2, 3), tf.bool),
                "dealer": tf.constant([0, 1]),
                "public_fu": tf.zeros((2, 4), tf.int32),
            }
        )
        loss = -tf.reduce_sum(
            tf.math.log(
                result["payment_probability"][
                    :, cell, int(outcomes.payment_classes(12000))
                ]
            )
        )
    gradients = tape.gradient(loss, payment_model.trainable_variables)
    assert all(
        g is not None and numpy.isfinite(g.numpy()).all() for g in gradients
    )
    assert float(tf.linalg.global_norm(gradients)) > 0
    numpy.testing.assert_allclose(
        result["payment_probability"].numpy().sum(-1), 1, atol=1e-6
    )


def test_same_hand_scoring_separates_bonuses_and_respects_limits():
    ctx = types.SimpleNamespace(
        tiles=mahjong.tile.TilesConverter.string_to_136_array(
            man="234", pin="234", sou="234678", honors="55"
        ),
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
                .with_name("fixtures")
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


def test_same_hand_dealer_90fu_dama_maps_to_payment_tensor():
    events = json.loads(
        pathlib.Path(__file__)
        .with_name("fixtures")
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
        .with_name("fixtures")
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


def test_payment_tensor_is_the_only_value_and_all_factors_learn():
    tf.keras.utils.set_random_seed(941)
    payment_model = critic.PaymentCritic()
    inputs = {
        "state": tf.random.normal((2, 16)),
        "opened": tf.constant([[False] * 4, [True] * 4]),
        "deposit_allowed": tf.constant([[True] * 4, [False] * 4]),
        "immediate_allowed": tf.constant([True, True]),
        "dealer": tf.zeros(2, tf.int32),
        "public_fu": tf.zeros((2, 4), tf.int32),
        "immediate_safe": tf.zeros((2, 3), tf.bool),
    }
    labels = {
        "payment": tf.ones((2, 40), tf.int32),
        "bonus": tf.fill((2, 24), -1),
        "scenario": tf.fill((2, 24, len(payment_bonus.SCENARIOS)), -1),
        "score": tf.fill((2, 24, len(payment_bonus.SCENARIOS)), -1),
        "points": tf.zeros(2),
    }
    cell = payment_bonus.CELLS.index((0, 1, 0))
    scenario = payment_bonus.SCENARIOS.index((2, 1, 1, 0))
    point_class = int(outcomes.payment_classes(8000))
    labels["payment"] = tf.tensor_scatter_nd_update(
        labels["payment"], [[0, cell]], [point_class]
    )
    labels["payment"] = tf.tensor_scatter_nd_update(
        labels["payment"],
        [[0, payment_bonus.CELLS.index((3, 0, 0))]],
        [int(outcomes.payment_classes(1000))],
    )
    labels["score"] = tf.tensor_scatter_nd_update(
        labels["score"],
        [[0, cell, scenario]],
        [payment_scoring.score_class(4, 40, False, 0)],
    )
    labels["bonus"] = tf.tensor_scatter_nd_update(
        labels["bonus"], [[0, cell]], [scenario]
    )
    labels["scenario"] = tf.tensor_scatter_nd_update(
        labels["scenario"], [[0, cell, scenario]], [point_class]
    )
    with tf.GradientTape() as tape:
        outputs = payment_model(inputs)
        loss, metrics = critic.loss_and_metrics(labels, outputs)
        loss += metrics["scenario_mse"]
    gradients = tape.gradient(loss, payment_model.trainable_variables)
    numpy.testing.assert_allclose(
        payment_model.distributions(inputs, actor_moments=True)[
            "actor_payment_moments"
        ],
        tf.gather(outputs["payment_probability"], critic.ACTOR_CELLS, axis=1)
        @ numpy.stack(
            [outcomes.PAYMENT_VALUES, outcomes.PAYMENT_VALUES**2], axis=-1
        ).astype(numpy.float32),
        rtol=2e-6,
        atol=1e-3,
    )
    assert all(
        g is not None and numpy.isfinite(g.numpy()).all() for g in gradients
    )
    assert all(
        float(
            tf.linalg.global_norm(
                [
                    gradient
                    for variable, gradient in zip(
                        payment_model.trainable_variables, gradients
                    )
                    if variable is layer.kernel
                ]
            )
        )
        > 0
        for layer in (
            payment_model.occurrence,
            payment_model.bonus,
            payment_model.amount,
            payment_model.hand_score,
        )
    )
    assert numpy.isfinite([float(v) for v in metrics.values()]).all()
    # Two different bonus events with equal added han have one amount law.
    numpy.testing.assert_array_equal(
        outputs["scenario_logits"][
            :, :, payment_bonus.SCENARIOS.index((2, 1, 0, 0))
        ],
        outputs["scenario_logits"][
            :, :, payment_bonus.SCENARIOS.index((3, 0, 0, 0))
        ],
    )
    numpy.testing.assert_allclose(
        metrics["scenario_mse"],
        (
            tf.nn.softmax(outputs["scenario_logits"])[0, cell, scenario].numpy()
            @ (outcomes.PAYMENT_VALUES / 10000)
            - 0.8
        )
        ** 2,
        rtol=1e-5,
    )
    numpy.testing.assert_allclose(
        outputs["terminal_payment_probability"].numpy().sum(-1), 1, atol=1e-6
    )
    payments = outcomes.expected_payments(
        outputs["terminal_payment_probability"].numpy()
    )
    numpy.testing.assert_allclose(
        outputs["expected_points"],
        [outcomes.settlement(row)[:4] / 10000 for row in payments],
        atol=1e-6,
    )
    probabilities = outputs["payment_probability"].numpy()
    assert numpy.all(probabilities[:, :36, 0] == 0)
    assert numpy.all(probabilities[1, 36:, 0] == 0)
    assert numpy.all(
        tf.nn.softmax(outputs["bonus_logits"]).numpy()[
            1,
            :,
            [i for i, s in enumerate(payment_bonus.SCENARIOS) if s[0] != 0],
        ]
        == 0
    )
    # A state baseline is derived from a policy over action distributions.
    numpy.testing.assert_allclose(
        outcomes.settlement(numpy.mean(payments, axis=0))[:4] / 10000,
        tf.reduce_mean(outputs["expected_points"], axis=0),
        atol=1e-6,
    )


def test_immediate_ron_suppresses_all_future_transfers_but_not_past_deposits():
    payment_model = critic.PaymentCritic()
    inputs = {
        "state": tf.zeros((1, 16)),
        "opened": tf.zeros((1, 4), tf.bool),
        "deposit_allowed": tf.ones((1, 4), tf.bool),
        "immediate_allowed": tf.constant([True]),
        "dealer": tf.zeros(1, tf.int32),
        "public_fu": tf.zeros((1, 4), tf.int32),
        "immediate_safe": tf.zeros((1, 3), tf.bool),
    }
    payment_model(inputs)
    payment_model.occurrence.kernel.assign(
        tf.zeros_like(payment_model.occurrence.kernel)
    )
    payment_model.occurrence.bias.assign([0])
    payment_model.immediate_risk.kernel.assign(
        tf.zeros_like(payment_model.immediate_risk.kernel)
    )
    payment_model.immediate_risk.bias.assign([numpy.log(0.2 / 0.8)])
    probability = payment_model(inputs)["payment_probability"].numpy()[0]
    expected = numpy.full(len(payment_bonus.CELLS), 0.5 * 0.8**3)
    expected[critic.DEAL_IN_CELLS] += [0.2, 0.2, 0.2]
    expected[[cell[0] == 3 for cell in payment_bonus.CELLS]] = 0.5
    numpy.testing.assert_allclose(1 - probability[:, 1], expected, atol=1e-6)
    numpy.testing.assert_allclose(probability.sum(-1), 1, atol=1e-6)

    inputs["immediate_safe"] = tf.constant([[True, False, False]])
    probability = payment_model(inputs)["payment_probability"].numpy()[0]
    expected = numpy.full(len(payment_bonus.CELLS), 0.5 * 0.8**2)
    expected[critic.DEAL_IN_CELLS] += [0, 0.2, 0.2]
    expected[[cell[0] == 3 for cell in payment_bonus.CELLS]] = 0.5
    numpy.testing.assert_allclose(1 - probability[:, 1], expected, atol=1e-6)
    inputs["immediate_allowed"] = tf.constant([False])
    probability = payment_model(inputs)["payment_probability"].numpy()[0]
    numpy.testing.assert_allclose(1 - probability[:, 1], 0.5, atol=1e-6)


def test_full_corpus_records_keep_visible_inputs_and_same_hand_targets():
    from log_dataset import afk, rebuild
    from model import critic_data, replay

    events = json.loads(
        pathlib.Path(__file__)
        .with_name("fixtures")
        .joinpath("payment_dealer_90fu.json")
        .read_text()
    )
    targets = payment_bonus.round_labels(events)
    observations = list(
        replay.Replay(
            "critic-roundtrip", afk.retained_hands(events)[0]
        ).observations()
    )
    raw = [
        critic_data.serialize(
            row, rebuild.retain_focused(row), targets[row.kyoku_id]
        )
        for row in observations
    ]
    features, labels, auxiliary = critic_data.parse_batch(tf.constant(raw))
    packing = critic_data.PackedBatch((features, labels, auxiliary))
    restored = packing.unpack(packing.pack(features, labels, auxiliary))
    for original, value in zip(
        tf.nest.flatten((features, labels, auxiliary)),
        tf.nest.flatten(restored),
    ):
        numpy.testing.assert_array_equal(original, value)
    for name in features:
        numpy.testing.assert_array_equal(
            features[name],
            numpy.stack([row.features[name] for row in observations]),
        )
    numpy.testing.assert_allclose(
        auxiliary["points"],
        [
            outcomes.settlement(outcomes.PAYMENT_VALUES[row.terminal_payment])[
                0
            ]
            / 10000
            for row in observations
        ],
    )
    assert numpy.any(auxiliary["score"].numpy() >= 0)
    table = payment_scoring.payment_amounts()
    valid = auxiliary["score"].numpy() >= 0
    mapped = table[
        auxiliary["dealer"].numpy()[:, None, None],
        numpy.arange(24)[None, :, None],
        numpy.maximum(auxiliary["score"].numpy(), 0),
    ]
    numpy.testing.assert_array_equal(
        mapped[valid],
        outcomes.PAYMENT_VALUES[auxiliary["scenario"].numpy()[valid]],
    )
    numpy.testing.assert_array_equal(
        labels["terminal_payment"],
        numpy.stack([row.terminal_payment for row in observations]),
    )


def test_joint_policy_restores_specialists_and_trains_one_shared_encoder(
    tmp_path,
):
    from log_dataset import afk, rebuild
    from model import (
        critic,
        critic_data,
        critic_full,
        critic_utility,
        decision_layers,
        replay,
    )

    events = json.loads(
        pathlib.Path(__file__)
        .with_name("fixtures")
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
                critic_data.serialize(
                    row, rebuild.retain_focused(row), targets[row.kyoku_id]
                )
                for row in observations
            ]
        )
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
                "correlation": numpy.eye(19).tolist(),
                "variance_scale": 1.0,
                "placement_scale": 4720.0,
                "temperature": 3.0,
            },
        )
        loss, metrics = critic_full.objective_metrics(
            inputs,
            outputs,
            utility={"advantages": advantages, "enabled": True, "weight": 0.1},
        )
    rates = critic_full.policy_rates(
        {name: float(value) for name, value in metrics.items()}
    )
    import feature_vector

    expected_top3 = []
    for name in feature_vector.TRAIN_POLICY_HEADS:
        valid = inputs[1][name].numpy() >= 0
        ranking = numpy.argsort(-outputs[name].numpy(), axis=-1)[:, :3]
        expected_top3.extend(
            numpy.any(
                ranking[valid] == inputs[1][name].numpy()[valid, None], axis=-1
            )
        )
    numpy.testing.assert_allclose(
        rates["imitation_top3_accuracy"], numpy.mean(expected_top3)
    )
    assert rates["imitation_top3_accuracy"] >= rates["imitation_accuracy"]
    base_loss, _ = critic_full.objective_metrics(inputs, outputs)
    numpy.testing.assert_allclose(
        loss - base_loss,
        (metrics["utility_loss"] - metrics["policy_loss"]) * (0.1 / 1.1),
        atol=2e-6,
    )
    inactive_loss, _ = critic_full.objective_metrics(
        inputs,
        outputs,
        utility={"advantages": advantages, "enabled": False, "weight": 2.0},
    )
    numpy.testing.assert_allclose(inactive_loss, base_loss, atol=2e-6)
    gradients = tape.gradient(loss, network.trainable_variables)
    assert all(value is not None for value in gradients), [
        variable.path
        for variable, gradient in zip(network.trainable_variables, gradients)
        if gradient is None
    ]
    assert all(
        gradient is None
        or not numpy.any(tf.convert_to_tensor(gradient).numpy())
        for gradient in tape.gradient(
            metrics["utility_loss"], network.critic.trainable_variables
        )
    )
    for name in (
        "policy_loss",
        "payment_auxiliary_loss",
        "final_placement_loss",
    ):
        gradient = tape.gradient(
            metrics[name], network.encoder.get_layer("board_projection").kernel
        )
        assert numpy.any(tf.convert_to_tensor(gradient).numpy() != 0)
    assert all(
        numpy.isfinite(tf.convert_to_tensor(value)).all() for value in gradients
    )
    for name in ("attack_residual", "defense_residual", "final_pass"):
        assert network.actor.get_layer(name).units == 256
    for value in (
        network.encoder.get_layer("hand_bank").tiles.kernel,
        network.encoder.get_layer("board_projection").kernel,
        network.actor.get_layer("attack_residual").kernel,
        network.actor.get_layer("defense_residual").kernel,
        network.actor.get_layer("final_pass").kernel,
        network.critic.state.kernel,
    ):
        index = next(
            index
            for index, variable in enumerate(network.trainable_variables)
            if variable is value
        )
        assert numpy.any(tf.convert_to_tensor(gradients[index]).numpy() != 0)
    assert network.actor.get_layer("hand_bank") is network.encoder.get_layer(
        "hand_bank"
    )
    assert {"terminal_payment_logits", "final_st3_value"}.isdisjoint(
        network.actor.output
    )
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
    saved = tmp_path / "joint.weights.h5"
    network.save_weights(saved)
    restored = critic.JointPolicyPayment()
    restored(inputs)
    restored.compile_training()
    restored(inputs)
    restored.load_weights(saved)
    for actual, expected in zip(restored.get_weights(), network.get_weights()):
        numpy.testing.assert_array_equal(actual, expected)
    for name in ("discard_policy_logits", "payment_probability"):
        numpy.testing.assert_allclose(
            restored(inputs)[name], outputs[name], atol=1e-7, rtol=1e-6
        )
    serving = export_policy.ServingModel(restored)
    expected = serving.serve(**inputs[0])
    tf.saved_model.save(
        serving,
        str(tmp_path / "saved_model"),
        signatures={
            "serving_default": serving.serve.get_concrete_function(**inputs[0])
        },
    )
    exported = tf.saved_model.load(str(tmp_path / "saved_model")).signatures[
        "serving_default"
    ](**inputs[0])
    for name in expected:
        numpy.testing.assert_allclose(
            exported[name], expected[name], atol=1e-6, rtol=1e-5
        )
