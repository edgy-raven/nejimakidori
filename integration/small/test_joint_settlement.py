"""Complete physical-hand settlements supervise one shared rank/payoff law."""

import numpy
import tensorflow as tf

from integration import settlement_reference
from log_dataset import scoring
from model import joint_settlement, losses, outcomes, payment_scoring


def expanded_terms(labels, outputs, mean=losses.mean_valid):
    """Exact observed-payment likelihood and derived hand-end rank likelihood.

    Liability/nagashi allocations outside the ordinary support
    receive neither settlement nor rank supervision. Coverage is reported.
    """
    same_transfer = tf.reduce_all(
        (
            outputs["joint_classes"]
            == tf.cast(labels["payment"][:, None, :36], tf.int64)
        )
        | (
            (labels["payment"][:, None, :36] < 0)
            & (outputs["joint_classes"] != 1)
        ),
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
    if "event" in labels:
        matches &= (
            outputs["joint_support"][:, :, 5]
            == tf.cast(labels["event"][:, None], tf.int64)
        )[..., None]
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
                                [cell[2] for cell in scoring.CELLS[:12]],
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
        "score_probability": tf.ones((1, 24, len(payment_scoring.SCORES)))
        / len(payment_scoring.SCORES),
        "event_logits": tf.zeros((1, len(joint_settlement.EVENTS))),
        "ending_draw_kind": tf.zeros(1, tf.int32),
        "ending_tenpai": tf.fill((1, 4), -1),
        "ending_safe": tf.zeros((1, 3), tf.bool),
        "deposit_required": tf.constant([[False] * 4]),
        "riichi_required": tf.zeros((1, 4), tf.bool),
        "tenpai_required": tf.zeros((1, 4), tf.bool),
        "action_tile": tf.fill((1,), 37),
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
    # GPU execution must retain exact integer transfers, including 8,700.
    compiled = tf.function(joint_settlement.settlement_states, jit_compile=True)
    indices = joint_settlement.dealer_indices(public["dealer"]).numpy()
    integer_delta = (
        joint_settlement.TRANSFERS[indices].astype(numpy.int64)
        + public["honba"].numpy().astype(numpy.int64)[:, None, None]
        * joint_settlement.HONBA[indices].astype(numpy.int64)
    ) @ joint_settlement.COEFFICIENTS.astype(numpy.int64)
    numpy.testing.assert_array_equal(
        compiled(public)["joint_scores"],
        public["scores"].numpy()[:, None, None]
        + integer_delta[:, :, None]
        + result["joint_bank"].numpy(),
    )
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
    cells = numpy.array(scoring.CELLS)
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
    concentrated["event_logits"] = tf.one_hot(
        [0] * 10, len(joint_settlement.EVENTS), 0.0, -1e9
    )
    concentrated["score_probability"] = tf.nn.softmax(
        tf.random.stateless_normal(
            (10, 24, len(payment_scoring.SCORES)), [4, 6]
        )
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
        utility = settlement_reference.projected_st3(result, public)
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


def test_policy_objective_adds_rank_bonuses_and_preserves_raw_st3():
    scores = numpy.array(
        [numpy.roll([40000.0, 30000, 20000, 10000], -seat) for seat in range(4)]
    )
    ranks = numpy.argsort(numpy.argsort(-scores, axis=1), axis=1)
    st3, probability = joint_settlement.post_hand_forecast(
        scores=tf.constant(scores, tf.float32),
        ranks=tf.constant(ranks, tf.int64),
        next_dealer=tf.zeros(4, tf.int64),
        next_hand=tf.fill([4], 8),
        terminal=tf.ones(4, tf.bool),
        pot=tf.zeros(4),
    )
    numpy.testing.assert_array_equal(st3, [140, 65, -10, -270])
    objective = joint_settlement.policy_objective(st3, probability)
    numpy.testing.assert_allclose(objective, [2.4, 0.65, -0.1, -3.7], atol=1e-6)
    # A forecast mixes terminal utilities; neither normalization nor bonuses
    # may be applied to a hard predicted rank instead of the rank distribution.
    weight = tf.constant([0.1, 0.2, 0.3, 0.4])
    numpy.testing.assert_allclose(
        joint_settlement.policy_objective(tf.reduce_sum(weight * st3), weight),
        -1.14,
        atol=1e-6,
    )
    numpy.testing.assert_allclose(
        tf.reduce_sum(weight * objective), -1.14, atol=1e-6
    )


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
    payoff = settlement_reference.projected_st3(
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
        settlement_reference.projected_st3(
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
            settlement_reference.projected_st3(
                {**result, "joint_probability": exact}, public
            ),
            expected,
            rtol=1e-6,
            atol=1e-5,
        )


def test_terminal_ron_draw_and_abort_obey_deposit_and_readiness_rules():
    public = {
        **inputs(),
        "deposit_required": tf.constant([[True, False, False, False]]),
    }
    # Before the final boundary, an actor-payer ron can occur before or after
    # declaration acceptance. Other endings require its deposit.
    ordinary, _, legal = joint_settlement.event_probabilities(public)
    assert legal[0, 0, 0] and legal[0, 0, 1]
    assert not legal[0, 28, 0] and legal[0, 28, 1]
    numpy.testing.assert_allclose(tf.reduce_sum(ordinary), 1, atol=1e-6)
    for event, boundary, deposit, kind, mask in (
        (0, 1, 0, 0, 0),  # Last declaration discard is ronned.
        (47, 1, 1, 2, 15),  # All tenpai: no noten transfer.
        (48, 2, 1, 3, 16),  # Four-riichi abort accepts the fourth stick.
        (6, 2, 0, 3, 16),  # Triple ron abort precedes acceptance.
    ):
        ended = joint_settlement.decode(
            {
                **public,
                "ending_draw_kind": tf.constant([boundary]),
                "event_logits": tf.one_hot(
                    [event], len(joint_settlement.EVENTS), 0.0, -1e9
                ),
            }
        )
        probability = ended["joint_probability"].numpy()[0]
        support = ended["joint_support"].numpy()[0]
        numpy.testing.assert_allclose(probability.sum(), 1, atol=1e-6)
        assert not probability[:, numpy.arange(16) != deposit].any()
        assert not probability[support[:, 1] != kind].any()
        if kind >= 2:
            assert not probability[support[:, 3] != mask].any()
    boundary = {
        **public,
        "ending_draw_kind": tf.constant([1]),
        "ending_tenpai": tf.constant([[1, 1, -1, -1]]),
        "ending_safe": tf.ones((1, 3), tf.bool),
    }
    event, _, legal = joint_settlement.event_probabilities(boundary)
    kinds = numpy.array([row[0] for row in joint_settlement.EVENTS])
    masks = numpy.array([row[3] for row in joint_settlement.EVENTS])
    assert not event.numpy()[0, kinds != 2].any()
    assert not event.numpy()[0, (masks & 3) != 3].any()
    assert not legal.numpy()[0, kinds != 2].any()
    # Safety at this action must not prohibit a later deal-in.
    future, _, _ = joint_settlement.event_probabilities(
        {**boundary, "ending_draw_kind": tf.zeros(1, tf.int32)}
    )
    assert future.numpy()[0, 0] > 0


def test_future_draw_readiness_requires_riichi_and_hadaka():
    public = {
        **inputs(),
        "riichi_required": tf.constant([[False, False, True, False]]),
        "tenpai_required": tf.constant([[False, False, True, True]]),
        "deposit_required": tf.constant([[True, False, False, False]]),
    }
    result = joint_settlement.decode(public)
    probability = result["joint_probability"].numpy()[0]
    support = result["joint_support"].numpy()[0]
    draws = support[:, 1] == 2
    ready = (support[:, 3, None] & (1 << numpy.arange(4))) > 0
    required = numpy.array([True, False, True, True])
    incompatible = draws[:, None] & numpy.any(
        ~ready[:, None] & (required | (joint_settlement.DEPOSITS > 0))[None], -1
    )
    assert not probability[incompatible].any()
    numpy.testing.assert_allclose(probability.sum(), 1, atol=1e-6)
    assert probability[draws & (support[:, 3] == 15)].sum() > 0
    assert probability[draws & (support[:, 3] == 13)].sum() > 0
    # Four melds require tenpai even without riichi; they do not force a stick.
    assert probability[draws & (support[:, 3] == 13), 1].sum() > 0
    assert not public["riichi_required"][0, 3]
    assert not result["joint_legal_deposits"].numpy()[0][incompatible].any()


def test_double_ron_awards_honba_and_bank_once_and_keeps_dealer_repeat():
    public = {
        **inputs(),
        "dealer": tf.constant([2]),
        "event_logits": tf.one_hot(
            [2], len(joint_settlement.EVENTS), 0.0, -1e9
        ),
        "score_probability": tf.broadcast_to(
            tf.one_hot(
                payment_scoring.SCORES.index((5, 0)),
                len(payment_scoring.SCORES),
            ),
            (1, 24, len(payment_scoring.SCORES)),
        ),
        "deposit_required": tf.constant([[False, False, False, True]]),
        "deposit_logits": tf.broadcast_to(
            tf.one_hot([8], 16, 0.0, -1e9)[:, None],
            (1, len(joint_settlement.EVENTS), 16),
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
        {
            **public,
            "event_logits": tf.one_hot(
                [6], len(joint_settlement.EVENTS), 0.0, -1e9
            ),
        }
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
        "state": tf.concat(
            [tf.fill((1, 12), 0.25), tf.fill((1, 408), 0.5), tf.ones((1, 92))],
            -1,
        ),
        "action_tile": tf.zeros(1, tf.int32),
        "has_meld": tf.zeros((1, 4), tf.bool),
        "opened": tf.zeros((1, 4), tf.bool),
        "riichi_required": tf.zeros((1, 4), tf.bool),
        "ippatsu_allowed": tf.ones((1, 4), tf.bool),
        "public_fu": tf.zeros((1, 4), tf.int32),
        "public_bonus": tf.zeros((1, 4), tf.int32),
        "yakuman_allowed": tf.ones((1, 4), tf.bool),
    }
    public = {
        name: tf.repeat(value, 3, axis=0) for name, value in public.items()
    }
    labels = {
        "payment": tf.ones((3, 40), tf.int32),
        "round_placement": tf.constant([[0, 1, 2, 3]] * 3),
        "pot": public["pot"],
        "points": tf.zeros(3),
        "score": tf.fill((3, 24, len(scoring.BONUS_HAN)), -1),
        "bonus": tf.broadcast_to(tf.constant([[2], [-1], [2]]), (3, 24)),
    }
    # Sparse scoring labels keep their valid counts across chunks.
    labels["score"] = tf.where(
        tf.range(3)[:, None, None] == 0,
        tf.fill(
            (3, 24, len(scoring.BONUS_HAN)),
            payment_scoring.SCORES.index((5, 0)),
        ),
        -1,
    )
    predictor = critic.PaymentCritic(belief_offset=0, player_width=8)
    predictor(public)
    # Partial chunks have a dynamic batch dimension. Their compiled loss
    # must stay stateless so replicas can execute the backward pass together.
    compiled = tf.function(predictor.chunk_statistics).get_concrete_function(
        *tf.nest.map_structure(
            lambda value: tf.TensorSpec((None, *value.shape[1:]), value.dtype),
            (public, {**labels, "valid": tf.ones(3)}),
        ),
        weights=[
            tf.TensorSpec(value.shape, value.dtype)
            for value in predictor.trainable_variables
        ],
    )
    assert not compiled.function_def.signature.is_stateful
    statistics, gradients = [], []
    for method in (predictor.chunk_statistics, predictor.supervised_statistics):
        with tf.GradientTape() as tape:
            tape.watch(public["state"])
            result = method(public, {**labels, "valid": tf.ones(3)})
            loss = tf.reduce_sum(result[:, 0])
        statistics.append(result)
        gradients.append(
            tape.gradient(
                loss,
                [
                    public["state"],
                    *predictor.trainable_variables,
                ],
            )
        )
    numpy.testing.assert_allclose(statistics[0], statistics[1], atol=1e-4)
    for full, streamed in zip(*gradients):
        assert full is not None and streamed is not None
        numpy.testing.assert_allclose(
            tf.convert_to_tensor(full),
            tf.convert_to_tensor(streamed),
            atol=3e-4,
            rtol=1e-4,
        )

    # Flattening must preserve the original categorical loss and derivative,
    # including unknown labels, while avoiding its graph-level shape assert.
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(public["state"])
        reference = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.maximum(labels["bonus"], 0),
                logits=predictor.distributions(public)["bonus_logits"],
            )
            * tf.cast(labels["bonus"] >= 0, tf.float32)
        )
        actual = predictor.chunk_statistics(
            public, {**labels, "valid": tf.ones(3)}
        )[critic.CRITIC_METRICS.index("bonus_han_nll"), 0]
    numpy.testing.assert_allclose(actual, reference, rtol=1e-6)
    variables = [public["state"], *predictor.bonus.trainable_variables]
    for actual, expected in zip(
        tape.gradient(actual, variables), tape.gradient(reference, variables)
    ):
        numpy.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    # Fresh-game labels have only transfers. Padding contributes neither
    # gradients nor support/count normalization mass.
    fresh = {name: labels[name] for name in ("payment", "pot")}
    for valid in ([False, True, False], [False] * 3):
        with tf.GradientTape() as tape:
            tape.watch(public["state"])
            result = predictor.supervised_statistics(
                public, fresh, valid=tf.constant(valid)
            )
            loss = sum(
                critic.settlement_metrics(result)[name]
                for name in ("payment_nll", "critic_constraint_loss")
            )
        gradients = tape.gradient(
            loss,
            [
                public["state"],
                *predictor.trainable_variables,
            ],
        )
        numpy.testing.assert_array_equal(
            result[:, 1],
            [sum(valid), 0, 0, *([sum(valid)] * 5)],
        )
        for gradient in gradients[:1]:
            numpy.testing.assert_array_equal(
                tf.convert_to_tensor(gradient).numpy()[~numpy.array(valid)], 0
            )
        if any(valid):
            expected = predictor.supervised_statistics(
                {name: value[1:2] for name, value in public.items()},
                {name: value[1:2] for name, value in fresh.items()},
            )
            numpy.testing.assert_allclose(result, expected, atol=1e-5)
            assert float(tf.linalg.global_norm(gradients)) > 0
        else:
            assert float(loss) == float(tf.linalg.global_norm(gradients)) == 0


def test_observed_categories_match_enumeration_without_score_cross_products():
    public = {
        **inputs(),
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


def test_partial_ron_amount_marginalizes_scores_but_preserves_exact_event():
    public = inputs()
    event_logits = tf.Variable(public["event_logits"])
    score_logits = tf.Variable(tf.math.log(public["score_probability"]))
    reference = joint_settlement.decode(public)
    # An unseen ura indicator hides a winning amount, while the event and
    # bank remain known. Score uncertainty must not erase the ron target.
    row = numpy.flatnonzero(reference["joint_support"].numpy()[0, :, 5] == 0)[0]
    bank = numpy.argmax(
        reference["joint_bank"][:, row, 0, :, None].numpy()
        == joint_settlement.payment_values(public)[:, None].numpy(),
        -1,
    )
    payment = tf.concat(
        [tf.cast(reference["joint_classes"][:, row], tf.int32), bank], -1
    )
    payment = tf.where((tf.range(40)[None] < 24) & (payment != 1), -1, payment)
    labels = {
        "payment": payment,
        "event": tf.constant([0]),
        "pot": public["pot"],
        "round_placement": tf.fill((1, 4), -1),
    }
    with tf.GradientTape(persistent=True) as tape:
        values = {
            **public,
            "event_logits": event_logits,
            "score_probability": tf.nn.softmax(score_logits),
        }
        actual = joint_settlement.observed_terms(labels, values)
        expected = expanded_terms(labels, joint_settlement.decode(values))
    numpy.testing.assert_allclose(
        actual["payment_nll"], expected["payment_nll"], atol=1e-6
    )
    assert actual["settlement_supported_fraction"] == 1
    for variable in (event_logits, score_logits):
        numpy.testing.assert_allclose(
            tape.gradient(actual["payment_nll"], variable),
            tape.gradient(expected["payment_nll"], variable),
            atol=1e-6,
        )
    numpy.testing.assert_allclose(
        tape.gradient(actual["payment_nll"], score_logits), 0, atol=1e-6
    )
    assert (
        float(
            tf.linalg.global_norm(
                [tape.gradient(actual["payment_nll"], event_logits)]
            )
        )
        > 0
    )


def test_observed_hand_target_matches_critic_without_future_outcomes():
    events = [
        dict(
            type="start_kyoku",
            bakaze="S",
            kyoku=4,
            oya=0,
            scores=[28000, 29000, 22000, 19000],
            kyotaku=2,
        ),
        dict(type="reach_accepted", actor=1),
        dict(type="ryukyoku", deltas=[1500, 1500, -1500, -1500]),
        dict(type="end_kyoku"),
        dict(
            type="start_kyoku",
            bakaze="S",
            kyoku=4,
            oya=0,
            scores=[29500, 29500, 20500, 17500],
            kyotaku=3,
        ),
        dict(type="hora", deltas=[-8300, 0, 11300, 0]),
        dict(type="end_kyoku"),
        dict(type="end_game"),
    ]
    states = outcomes.hand_continuation_states(events)
    numpy.testing.assert_array_equal(
        states[0, 0],
        [29500, 29500, 20500, 17500, 1, 0, 2, 3, 0, 7, 0, 3000],
    )
    flat = states.reshape(-1, 12)
    observed = (
        joint_settlement.post_hand_forecast(
            scores=tf.constant(flat[:, :4], tf.float32),
            ranks=tf.constant(flat[:, 4:8], tf.int64),
            next_dealer=tf.constant(flat[:, 8], tf.int64),
            next_hand=tf.constant(flat[:, 9], tf.int32),
            terminal=tf.cast(flat[:, 10], tf.bool),
            pot=tf.constant(flat[:, 11], tf.float32),
        )[0]
        .numpy()
        .reshape(2, 4)
    )
    public = {
        **inputs(),
        "scores": tf.constant([[28000.0, 29000, 22000, 19000]]),
        "pot": tf.constant([2000.0]),
        "honba": tf.constant([0.0]),
        "first_dealer": tf.constant([1]),
    }
    predicted = joint_settlement.decode(public)
    support = predicted["joint_support"].numpy()[0]
    row = numpy.flatnonzero((support[:, 1] == 2) & (support[:, 3] == 3))[0]
    payoffs = joint_settlement.continuation_payoffs(
        scores=predicted["joint_scores"],
        ranks=predicted["joint_ranks"],
        support=predicted["joint_support"],
        dealer=public["dealer"],
        hand_number=public["hand_number"],
        pot=public["pot"],
    )
    numpy.testing.assert_allclose(
        observed[0, 0], payoffs[0, row, 2], rtol=1e-6, atol=1e-5
    )
    numpy.testing.assert_allclose(
        joint_settlement.observed_payoffs(tf.constant(flat)),
        observed.reshape(-1),
        atol=1e-5,
    )
    final_scores = [21200, 29500, 31800, 17500]
    numpy.testing.assert_array_equal(
        observed[1],
        outcomes.saint3_jade_south(
            final_scores, outcomes.score_ranks(final_scores, 1)
        ),
    )
    events[5]["deltas"] = [0, -8300, 11300, 0]
    changed = outcomes.hand_continuation_states(events)
    numpy.testing.assert_array_equal(changed[0], states[0])
    numpy.testing.assert_array_equal(
        joint_settlement.observed_payoffs(tf.constant(changed[0])),
        joint_settlement.observed_payoffs(tf.constant(states[0])),
    )
    assert not numpy.array_equal(changed[1], states[1])
