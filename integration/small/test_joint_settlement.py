"""Complete physical-hand settlements supervise one shared rank/payoff law."""

import numpy
import tensorflow as tf

from log_dataset import critic_data, payment_bonus
from model import joint_settlement, objectives, outcomes, payment_scoring


def expanded_terms(labels, outputs, mean=objectives.mean_valid):
    """Exact observed-payment likelihood and derived hand-end rank likelihood.

    Liability/nagashi allocations outside the ordinary support
    receive neither settlement nor rank supervision. Coverage is reported.
    """
    same_transfer = tf.reduce_all(
        outputs["joint_classes"]
        == tf.cast(labels["payment"][:, None, :36], tf.int64),
        -1,
    )
    actual_bank = tf.gather(
        joint_settlement.payment_values(labels),
        labels["payment"][:, 36:],
        axis=1,
        batch_dims=1,
    )
    matches = same_transfer[:, :, None] & tf.reduce_all(
        outputs["joint_bank"] == actual_bank[:, None, None], -1
    )
    matches &= outputs["joint_legal_deposits"]
    supported = tf.reduce_any(matches, axis=(1, 2))
    log_likelihood = tf.reduce_logsumexp(
        tf.where(matches, outputs["joint_log_probability"], -1e30), axis=(1, 2)
    )
    rank_log_probability = tf.gather(
        outputs["round_placement_logits"],
        tf.maximum(labels["round_placement"], 0),
        axis=2,
        batch_dims=2,
    )
    return {
        "payment_nll": mean(-log_likelihood, supported),
        "round_placement_loss": mean(
            -rank_log_probability,
            supported[:, None] & (labels["round_placement"] >= 0),
        ),
        "settlement_multi_ron_fraction": tf.reduce_mean(
            tf.cast(
                tf.reduce_sum(
                    tf.cast(
                        tf.linalg.matmul(
                            tf.cast(labels["payment"][:, :12] != 1, tf.float32),
                            tf.one_hot(
                                [cell[2] for cell in payment_bonus.CELLS[:12]],
                                4,
                            ),
                        )
                        > 0,
                        tf.int32,
                    ),
                    -1,
                )
                > 1,
                tf.float32,
            )
        ),
        "settlement_supported_fraction": tf.reduce_mean(
            tf.cast(supported, tf.float32)
        ),
    }


def inputs():
    return {
        "dealer": tf.constant([0]),
        "immediate_probability": tf.constant([[1.0] + [0.0] * 24]),
        "immediate_score_probability": tf.ones(
            (1, 3, len(payment_scoring.SCORES))
        )
        / len(payment_scoring.SCORES),
        "score_probability": tf.ones((1, 24, len(payment_scoring.SCORES)))
        / len(payment_scoring.SCORES),
        "event_logits": tf.zeros((1, len(joint_settlement.EVENTS))),
        "immediate_deposit_required": tf.zeros((1, 4), tf.bool),
        "deposit_required": tf.constant([[False] * 4]),
        "riichi_required": tf.zeros((1, 4), tf.bool),
        "pot": tf.constant([14000.0]),
        "honba": tf.constant([2.0]),
        "scores": tf.constant([[24000.0, 24000, 19000, 19000]]),
        "first_dealer": tf.constant([2]),
        "hand_number": tf.constant([7]),
        "deposit_allowed": tf.constant([[True, True, False, True]]),
        "deposit_logits": tf.zeros((1, len(joint_settlement.EVENTS), 16)),
    }


def test_joint_law_conserves_points_and_reconstructs_tsumo_bank_and_tie_order():
    public = inputs()
    result = joint_settlement.decode(public)
    probability = result["joint_probability"].numpy()[0]
    numpy.testing.assert_allclose(probability.sum(), 1, atol=1e-6)
    assert not probability[:, joint_settlement.DEPOSITS[:, 2] == 1].any()
    support = result["joint_support"].numpy()[0]
    # One winner receives a shared nondealer mangan: 4000/2000/2000.
    payment = numpy.zeros((4, 4, 4), numpy.int32)
    payment[1, [0, 2, 3], 1] = [4000, 2000, 2000]
    classes = outcomes.payment_classes(payment)[outcomes.PAYMENT_CELLS][:36]
    row = numpy.flatnonzero(
        numpy.all(result["joint_classes"][0] == classes, -1)
    )[0]
    # New riichi by winner and seat 3; existing deposits aren't paid twice.
    numpy.testing.assert_array_equal(
        result["joint_bank"][0, row, 10], [0, 15000, 0, -1000]
    )
    numpy.testing.assert_array_equal(
        result["joint_scores"][0, row, 10], [19800, 47600, 16800, 15800]
    )
    outstanding = numpy.where(
        support[:, 1, None] < 2,
        0,
        14000 + 1000 * joint_settlement.DEPOSITS.sum(-1)[None],
    )
    numpy.testing.assert_allclose(
        result["joint_scores"][0].numpy().sum(-1) + outstanding, 100000
    )
    for seat in range(4):
        numpy.testing.assert_allclose(
            tf.exp(result["round_placement_logits"])[0, seat].numpy().sum(),
            1,
            atol=2e-5,
        )
    zero = numpy.flatnonzero((support[:, 1] == 2) & (support[:, 3] == 0))[0]
    numpy.testing.assert_array_equal(
        result["joint_ranks"][0, zero, 0], [0, 1, 2, 3]
    )
    tied = joint_settlement.decode(
        {**public, "scores": tf.constant([[25000.0] * 4])},
    )
    numpy.testing.assert_array_equal(
        tied["joint_ranks"][0, zero, 0], [2, 3, 0, 1]
    )
    marginal = joint_settlement.marginals(result)
    numpy.testing.assert_allclose(tf.reduce_sum(marginal, -1), 1, atol=3e-5)
    cells = numpy.array(payment_bonus.CELLS)
    coefficient = numpy.array(
        [
            (cells[:, 2] == seat).astype(float)
            - (cells[:, 1] == seat)
            + ((cells[:, 0] == 3) & (cells[:, 1] == seat))
            for seat in range(4)
        ]
    ).T
    expected = (
        tf.reduce_sum(
            marginal * joint_settlement.payment_values(public)[:, None], -1
        ).numpy()
        @ coefficient
    )
    numpy.testing.assert_allclose(
        expected / 10000, result["expected_points"], atol=3e-5
    )

    # Concentrated legal score distributions must not create negative
    # zero-payment probability through float32 complement roundoff.
    concentrated = {
        name: tf.repeat(value, 10, 0) for name, value in public.items()
    }
    concentrated["immediate_probability"] = tf.one_hot([1] * 10, 25)
    concentrated["immediate_score_probability"] = tf.nn.softmax(
        tf.random.stateless_normal((10, 3, len(payment_scoring.SCORES)), [4, 6])
        * 3
    )
    marginal = joint_settlement.marginals(joint_settlement.decode(concentrated))
    assert tf.reduce_min(marginal) >= 0
    numpy.testing.assert_allclose(tf.reduce_sum(marginal, -1), 1, atol=3e-5)


def test_payment_and_round_rank_supervise_single_and_double_ron():
    public = {**inputs(), "pot": tf.zeros(1), "honba": tf.zeros(1)}
    factors = tf.Variable(tf.zeros((1, len(joint_settlement.EVENTS))))
    deposits = tf.Variable(public["deposit_logits"])
    payment = numpy.zeros((4, 4, 4), numpy.int32)
    payment[0, 1, 0] = 12000
    labels = {
        "payment": tf.constant(
            outcomes.payment_classes(payment)[outcomes.PAYMENT_CELLS][None],
            tf.int32,
        ),
        "round_placement": tf.constant([[0, 3, 1, 2]]),
        "pot": public["pot"],
    }
    with tf.GradientTape(persistent=True) as tape:
        result = joint_settlement.decode(
            {**public, "event_logits": factors, "deposit_logits": deposits}
        )
        terms = expanded_terms(labels, result)
        utility = joint_settlement.projected_st3(result, public)
    assert terms["settlement_supported_fraction"] == 1
    for loss in (terms["payment_nll"], terms["round_placement_loss"], utility):
        for gradient in tape.gradient(loss, [factors, deposits]):
            assert gradient is not None
            assert numpy.isfinite(gradient).all()
            assert numpy.any(gradient.numpy() != 0)
    payment[0, 1, 2] = 8000
    masked = expanded_terms(
        {
            **labels,
            "payment": tf.constant(
                outcomes.payment_classes(payment)[outcomes.PAYMENT_CELLS][None],
                tf.int32,
            ),
        },
        result,
    )
    assert masked["settlement_supported_fraction"] == 1
    assert masked["payment_nll"] > 0
    assert masked["round_placement_loss"] > 0


def test_terminal_st3_is_exact_and_draw_pot_goes_to_leader():
    public = {
        **inputs(),
        "honba": tf.zeros(1),
        "scores": tf.constant([[40000.0, 16000, 15000, 15000]]),
    }
    result = joint_settlement.decode(public)
    # South 4, zero-tenpai draw advances and ends. The bank goes to first.
    support = result["joint_support"].numpy()[0]
    row = numpy.flatnonzero((support[:, 1] == 2) & (support[:, 3] == 0))[0]
    exact = tf.scatter_nd(
        [[0, row, 0]], [1.0], tf.shape(result["joint_probability"])
    )
    payoff = joint_settlement.projected_st3(
        {**result, "joint_probability": exact}, public
    )
    numpy.testing.assert_array_equal(payoff, [154])
    branch_payoffs, final_ranks = joint_settlement.continuation_forecast(
        scores=result["joint_scores"],
        ranks=result["joint_ranks"],
        support=result["joint_support"],
        dealer=public["dealer"],
        hand_number=public["hand_number"],
        pot=public["pot"],
    )
    numpy.testing.assert_array_equal(final_ranks[0, row, 0], [1, 0, 0, 0])
    numpy.testing.assert_array_equal(branch_payoffs[0, row, 0], 154)
    numpy.testing.assert_allclose(tf.reduce_sum(final_ranks, -1), 1, atol=1e-6)
    assert numpy.all(final_ranks >= 0)
    # All-tenpai repeats, but the leading dealer can still end South 4.
    row = numpy.flatnonzero((support[:, 1] == 2) & (support[:, 3] == 15))[0]
    exact = tf.scatter_nd(
        [[0, row, 0]], [1.0], tf.shape(result["joint_probability"])
    )
    numpy.testing.assert_array_equal(
        joint_settlement.projected_st3(
            {**result, "joint_probability": exact}, public
        ),
        [154],
    )
    # Abortive draws repeat even when the dealer leads in South 4.
    row = numpy.flatnonzero(support[:, 1] == 3)[0]
    numpy.testing.assert_allclose(
        branch_payoffs[0, row, 0],
        joint_settlement.continuation_values(
            public["scores"], tf.constant([1.0]), tf.constant([[1.0, 0, 0, 0]])
        )[0][0],
        atol=1e-5,
    )
    # Identical zero payments do not imply identical continuation: nobody
    # tenpai advances; all-tenpai and abortive draws retain the dealer.
    public = {
        **public,
        "hand_number": tf.constant([6]),
        "scores": tf.constant([[28000.0, 29000, 22000, 21000]]),
    }
    result = joint_settlement.decode(public)
    for mask in (0, 15, 16):
        expected = joint_settlement.continuation_values(
            public["scores"],
            tf.constant([1.0 if mask == 0 else 2.0]),
            tf.constant([[0.0 if mask == 0 else 1.0, 1, 0, 0]]),
        )[0]
        row = numpy.flatnonzero((support[:, 1] >= 2) & (support[:, 3] == mask))[
            0
        ]
        exact = tf.scatter_nd(
            [[0, row, 0]], [1.0], tf.shape(result["joint_probability"])
        )
        numpy.testing.assert_allclose(
            joint_settlement.projected_st3(
                {**result, "joint_probability": exact}, public
            ),
            expected,
            atol=1e-5,
        )


def test_immediate_deal_in_ends_the_hand_before_future_deposits():
    public = {
        **inputs(),
        "immediate_probability": tf.constant([[0.0, 1, 0, 0] + [0.0] * 21]),
    }
    result = joint_settlement.decode(public)
    marginal = joint_settlement.marginals(result).numpy()[0]
    values = joint_settlement.payment_values(public).numpy()[0]
    outgoing = payment_bonus.CELLS.index((0, 0, 1))
    numpy.testing.assert_allclose(
        marginal[outgoing, values > 0].sum(), 1, atol=1e-6
    )
    assert not result["joint_probability"].numpy()[..., 1:].any()
    for cell in range(36):
        if cell != outgoing:
            numpy.testing.assert_allclose(marginal[cell, 1], 1, atol=1e-6)
    # Only the existing bank is collected; nobody pays an extra deposit.
    numpy.testing.assert_allclose(
        marginal[36:] @ values, [0, 14000, 0, 0], atol=0.01
    )
    baseline = joint_settlement.decode(inputs())
    mixture = joint_settlement.decode(
        {
            **inputs(),
            "immediate_probability": tf.constant(
                [[0.25, 0.75, 0, 0] + [0.0] * 21]
            ),
        },
    )
    numpy.testing.assert_allclose(
        mixture["joint_probability"],
        0.25 * baseline["joint_probability"]
        + 0.75 * result["joint_probability"],
        atol=1e-7,
    )


def test_terminal_draw_accepts_declaration_but_ron_and_triple_abort_do_not():
    public = {
        **inputs(),
        "deposit_required": tf.constant([[True, False, False, False]]),
    }
    future = joint_settlement.decode(public)
    assert (
        not future["joint_probability"]
        .numpy()[:, :, joint_settlement.DEPOSITS[:, 0] == 0]
        .any()
    )
    for event, deposit, kind, mask in (
        (1, 0, 0, 0),  # Declaration discard is ronned.
        (19, 1, 2, 15),  # All four tenpai: no noten transfer.
        (20, 1, 3, 16),  # Four-riichi abort accepts the fourth deposit.
        (21, 0, 3, 16),  # Triple ron abort precedes declaration acceptance.
    ):
        ended = joint_settlement.decode(
            {
                **public,
                "immediate_probability": tf.one_hot([event], 25),
            },
        )
        probability = ended["joint_probability"].numpy()[0]
        support = ended["joint_support"].numpy()[0]
        numpy.testing.assert_allclose(probability.sum(), 1, atol=1e-6)
        assert not probability[:, numpy.arange(16) != deposit].any()
        assert not probability[support[:, 1] != kind].any()
        if kind >= 2:
            assert not probability[support[:, 3] != mask].any()


def test_future_draw_readiness_agrees_with_existing_and_new_riichi():
    public = {
        **inputs(),
        "riichi_required": tf.constant([[False, False, True, False]]),
        "deposit_required": tf.constant([[True, False, False, False]]),
    }
    result = joint_settlement.decode(public)
    probability = result["joint_probability"].numpy()[0]
    support = result["joint_support"].numpy()[0]
    draws = support[:, 1] == 2
    ready = (support[:, 3, None] & (1 << numpy.arange(4))) > 0
    required = numpy.array([True, False, True, False])
    incompatible = draws[:, None] & numpy.any(
        ~ready[:, None] & (required | (joint_settlement.DEPOSITS > 0))[None], -1
    )
    assert not probability[incompatible].any()
    numpy.testing.assert_allclose(probability.sum(), 1, atol=1e-6)
    assert probability[draws & (support[:, 3] == 15)].sum() > 0
    assert probability[draws & (support[:, 3] == 5)].sum() > 0
    assert not result["joint_legal_deposits"].numpy()[0][incompatible].any()


def test_double_ron_awards_honba_and_bank_once_and_keeps_dealer_repeat():
    public = {
        **inputs(),
        "dealer": tf.constant([2]),
        "immediate_probability": tf.one_hot([22], 25),
        "immediate_score_probability": tf.broadcast_to(
            tf.one_hot(
                payment_scoring.SCORES.index((5, 0)),
                len(payment_scoring.SCORES),
            ),
            (1, 3, len(payment_scoring.SCORES)),
        ),
        "deposit_required": tf.constant([[False, False, False, True]]),
        "immediate_deposit_required": tf.constant(
            [[False, False, False, True]]
        ),
    }
    result = joint_settlement.decode(public)
    probability = result["joint_probability"].numpy()[0]
    row, deposit = numpy.argwhere(probability > 0)[0]
    assert probability[row, deposit] == 1
    assert deposit == 8
    # Seat 1 takes the pot/honba, while seat 2's dealer win still repeats.
    numpy.testing.assert_array_equal(
        result["joint_bank"][0, row, deposit], [0, 15000, 0, -1000]
    )
    numpy.testing.assert_array_equal(
        result["joint_scores"][0, row, deposit], [3400, 47600, 31000, 18000]
    )
    assert result["joint_support"][0, row, 4] == 6
    triple = joint_settlement.decode(
        {**public, "immediate_probability": tf.one_hot([21], 25)}
    )
    numpy.testing.assert_allclose(
        tf.reduce_sum(
            triple["joint_probability"][..., None] * triple["joint_bank"],
            (1, 2),
        ),
        [[0, 0, 0, -1000]],
        atol=1e-5,
    )


def test_streamed_settlement_gradients_equal_full_batch(monkeypatch):
    from model import critic

    # Exercise a partial final chunk without inflating the CPU test batch.
    monkeypatch.setattr(critic, "STATISTICS_CHUNK", 2)
    public = {
        **inputs(),
        "state": tf.ones((1, 64)),
        "immediate_wait": tf.ones((1, 3)) * 0.3,
        "immediate_safe": tf.zeros((1, 3), tf.bool),
        "immediate_allowed": tf.ones(1, tf.bool),
        "immediate_ankan": tf.zeros(1, tf.bool),
        "has_meld": tf.zeros((1, 4), tf.bool),
        "opened": tf.zeros((1, 4), tf.bool),
        "riichi_required": tf.zeros((1, 4), tf.bool),
        "ippatsu_allowed": tf.ones((1, 4), tf.bool),
        "immediate_ippatsu": tf.ones((1, 4), tf.bool),
        "public_fu": tf.zeros((1, 4), tf.int32),
        "immediate_draw_kind": tf.zeros(1, tf.int32),
        "immediate_tenpai": tf.fill((1, 4), -1),
    }
    public = {
        name: tf.repeat(value, 3, axis=0) for name, value in public.items()
    }
    labels = {
        "payment": tf.ones((3, 40), tf.int32),
        "round_placement": tf.constant([[0, 1, 2, 3]] * 3),
        "pot": public["pot"],
        "points": tf.zeros(3),
        "score": tf.fill((3, 24, len(payment_bonus.SCENARIOS)), -1),
    }
    # Sparse scoring labels keep their valid counts across chunks.
    labels["score"] = tf.where(
        tf.range(3)[:, None, None] == 0,
        tf.fill(
            (3, 24, len(payment_bonus.SCENARIOS)),
            payment_scoring.SCORES.index((5, 0)),
        ),
        -1,
    )
    labels.update(critic_data.score_targets(labels.pop("score")))
    predictor = critic.PaymentCritic()
    predictor(public)
    statistics, gradients = [], []
    for method in (predictor.chunk_statistics, predictor.supervised_statistics):
        with tf.GradientTape() as tape:
            tape.watch([public["state"], public["immediate_wait"]])
            result = method(public, labels)
            loss = tf.reduce_sum(result[:, 0])
        statistics.append(result)
        gradients.append(
            tape.gradient(
                loss,
                [
                    public["state"],
                    public["immediate_wait"],
                    *predictor.trainable_variables,
                ],
            )
        )
    numpy.testing.assert_allclose(statistics[0], statistics[1], atol=1e-4)
    for full, streamed in zip(*gradients):
        assert full is not None and streamed is not None
        numpy.testing.assert_allclose(full, streamed, atol=3e-4, rtol=1e-4)


def test_observed_categories_match_enumeration_without_score_cross_products():
    public = {
        **inputs(),
        "immediate_probability": tf.fill((1, 25), 1 / 25),
        "immediate_deposit_required": tf.constant(
            [[False, True, False, False]]
        ),
        "deposit_required": tf.constant([[True, True, False, False]]),
    }
    factors = tf.Variable(
        tf.random.stateless_normal(
            (1, len(joint_settlement.EVENTS)), seed=[41, 8]
        )
    )
    deposits = tf.Variable(public["deposit_logits"])
    score = tf.Variable(tf.math.log(public["score_probability"]))
    # Singles, shared-score tsumo, double ron, every exhaustive mask and
    # abort include the ambiguous zero-transfer label and bank priority.
    reference = joint_settlement.decode(public)
    support = reference["joint_support"].numpy()[0]
    for event in [0, 2, 6, 28, 29, *range(32, 49)]:
        row = numpy.flatnonzero(support[:, 5] == event)[-1]
        for deposit in (0, 2, 3, 10):
            payment = tf.concat(
                [
                    reference["joint_classes"][:, row],
                    tf.constant(
                        numpy.argmax(
                            reference["joint_bank"][
                                :, row, deposit, :, None
                            ].numpy()
                            == joint_settlement.payment_values(public)[
                                :, None
                            ].numpy(),
                            -1,
                        )
                    ),
                ],
                -1,
            )
            labels = {
                "payment": tf.cast(payment, tf.int32),
                "pot": public["pot"],
                "round_placement": reference["joint_ranks"][:, row, deposit],
            }
            with tf.GradientTape(persistent=True) as tape:
                values = {
                    **public,
                    "event_logits": factors,
                    "deposit_logits": deposits,
                    "score_probability": tf.nn.softmax(score),
                }
                direct = joint_settlement.observed_terms(labels, values)
                expanded = expanded_terms(
                    labels, joint_settlement.decode(values)
                )
            numpy.testing.assert_allclose(
                direct["payment_nll"], expanded["payment_nll"], atol=3e-5
            )
            numpy.testing.assert_equal(
                direct["settlement_supported_fraction"],
                expanded["settlement_supported_fraction"],
            )
            numpy.testing.assert_equal(
                direct["settlement_multi_ron_fraction"],
                expanded["settlement_multi_ron_fraction"],
            )
            for compact, full in zip(
                tape.gradient(
                    direct["payment_nll"], [factors, deposits, score]
                ),
                tape.gradient(
                    expanded["payment_nll"], [factors, deposits, score]
                ),
            ):
                numpy.testing.assert_allclose(compact, full, atol=3e-5)


def test_payoff_contexts_share_only_identical_public_settlements():
    from model import critic_utility

    public = {
        name: inputs()[name]
        for name in (
            "scores",
            "dealer",
            "pot",
            "honba",
            "first_dealer",
            "hand_number",
        )
    }
    # Duplicate contexts share a table. Changing any rule input must split
    # it, including first-dealer tie priority and terminal round thresholds.
    rows = [public, public]
    for name in public:
        rows.append(
            {
                **public,
                name: public[name] + (1000 if name in ("scores", "pot") else 1),
            }
        )
    batch = {name: tf.concat([row[name] for row in rows], 0) for name in public}
    contexts, mapping = critic_utility.payoff_contexts(batch)
    numpy.testing.assert_array_equal(mapping, [0, 0, 1, 2, 3, 4, 5, 6])
    for name in public:
        numpy.testing.assert_array_equal(
            tf.gather(contexts[name], mapping), batch[name]
        )
    expected = critic_utility.position_payoffs(batch)
    actual = tf.gather(critic_utility.position_payoffs(contexts), mapping)
    numpy.testing.assert_array_equal(actual, expected)
