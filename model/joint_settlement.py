"""Scored hand endings and bank transitions, with coexisting ron claims.

The learned tensor describes events and hand scores. Monetary transfers are
computed by rules, including double ron, dealer repeats and deposit priority.
"""

import itertools

import numpy
import tensorflow as tf

from log_dataset import scoring
from model import losses, outcomes, payment_scoring

DEPOSITS = numpy.array(
    [[(mask >> seat) & 1 for seat in range(4)] for mask in range(16)],
    numpy.float32,
)
# kind, discarder (or tsumo winner), absolute winner bits, draw mask.
# A ron group has three distinct claimant slots; mask 7 is a triple abort.
EVENTS = (
    tuple(
        (
            0,
            payer,
            sum(
                1 << ((payer + offset + 1) % 4)
                for offset in range(3)
                if mask & (1 << offset)
            ),
            0,
        )
        for payer in range(4)
        for mask in range(1, 8)
    )
    + tuple((1, seat, 1 << seat, 0) for seat in range(4))
    + tuple((2, -1, 0, mask) for mask in range(16))
    + ((3, -1, 0, 16),)
)
BASE_VALUES = numpy.array(
    sorted(
        {
            8000 * fu if han == 0 else outcomes.basic_points(han, fu)
            for han, fu in payment_scoring.SCORES
        }
    ),
    numpy.int32,
)
SCORE_TO_BASE = numpy.array(
    [
        BASE_VALUES
        == (8000 * fu if han == 0 else outcomes.basic_points(han, fu))
        for han, fu in payment_scoring.SCORES
    ],
    numpy.float32,
)


def settlement_support():
    """Exactly enumerate scored wins; deposits remain a separate 16-way axis."""
    rows, transfers, selectors, honba = [], [], [], []
    for dealer in range(4):
        for event_id, (kind, payer, mask, draw) in enumerate(EVENTS):
            winners = [seat for seat in range(4) if mask & (1 << seat)]
            triple = len(winners) == 3
            collector = (
                min(winners, key=lambda seat: (seat - payer) % 4)
                if winners and not triple
                else -1
            )
            scored = winners if not triple else []
            for values in itertools.product(
                range(len(BASE_VALUES)), repeat=len(scored)
            ):
                payment = numpy.zeros(36, numpy.int32)
                bonus = numpy.zeros(36, numpy.int32)
                heads = []
                for winner, value in zip(scored, values):
                    cells = [
                        i
                        for i, cell in enumerate(scoring.CELLS[:24])
                        if cell[0] == kind
                        and cell[2] == winner
                        and (kind == 1 or cell[1] == payer)
                    ]
                    heads.append((cells[0], value))
                    for cell in cells:
                        source = scoring.CELLS[cell][1]
                        factor = (
                            (6 if winner == dealer else 4)
                            if kind == 0
                            else (
                                2 if winner == dealer or source == dealer else 1
                            )
                        )
                        payment[cell] = 100 * (
                            (factor * BASE_VALUES[value] + 99) // 100
                        )
                        # Native multiple-ron rules award honba and pot to
                        # the closest winner only, rather than to every claim.
                        bonus[cell] = (
                            300 * (winner == collector) if kind == 0 else 100
                        )
                if kind == 2:
                    count = int(DEPOSITS[draw].sum())
                    if 0 < count < 4:
                        for cell, (mode, source, winner) in enumerate(
                            scoring.CELLS[:36]
                        ):
                            if (
                                mode == 2
                                and not DEPOSITS[draw, source]
                                and DEPOSITS[draw, winner]
                            ):
                                payment[cell] = 3000 // (count * (4 - count))
                rows.append(
                    (
                        dealer,
                        3 if triple else kind,
                        collector,
                        16 if triple else draw,
                        0 if triple else mask,
                        event_id,
                    )
                )
                transfers.append(payment)
                honba.append(bonus)
                selectors.append(heads + [(-1, 0)] * (2 - len(heads)))
    return (
        numpy.array(rows, numpy.int32),
        numpy.array(transfers),
        numpy.array(selectors),
        numpy.array(honba),
    )


# Support columns: dealer, kind, pot collector, draw mask, winner bits, event.
# Transfers/honba: [settlement, 36 directed payment cells].
# Selectors: [settlement, 2 claims, (score head, base-value index)].
SUPPORT, TRANSFERS, SELECTORS, HONBA = settlement_support()
CLASSES = outcomes.payment_classes(TRANSFERS)
COEFFICIENTS = numpy.array(
    [
        [int(recipient == seat) - int(payer == seat) for seat in range(4)]
        for _, payer, recipient in scoring.CELLS[:36]
    ],
    numpy.float32,
)
# Existing immediate columns: survive, singles, exhaustive masks, abort,
# triple abort, then pairs (1+2, 1+3, 2+3).
IMMEDIATE_EVENTS = numpy.array(
    [
        (
            {2: 1, 4: 2, 8: 3, 6: 22, 10: 23, 12: 24, 14: 21}[mask]
            if kind == 0 and payer == 0
            else 4 + draw if kind == 2 else 20 if kind == 3 else -1
        )
        for kind, payer, mask, draw in EVENTS
    ]
)

# Tsumo shares one score across three payers; ron has one head per claim.
TSUMO_CELLS = tuple(
    tuple(
        index
        for index, cell in enumerate(scoring.CELLS[:24])
        if cell[0] == 1 and cell[2] == winner
    )
    for winner in range(4)
)
SCORE_HEADS = tuple(range(12)) + tuple(cells[0] for cells in TSUMO_CELLS)
EVENT_ROWS = numpy.array(
    [
        numpy.flatnonzero((SUPPORT[:, 0] == 0) & (SUPPORT[:, 5] == event))[0]
        for event in range(len(EVENTS))
    ]
)


def dealer_indices(dealer):
    """Select the contiguous settlement block for each position's dealer."""
    return (
        tf.cast(dealer, tf.int32)[:, None] * (len(SUPPORT) // 4)
        + tf.range(len(SUPPORT) // 4)[None]
    )


def bank_transfers(collectors, pot):
    """Bank movement for every position, settlement and deposit mask."""
    return (
        tf.one_hot(collectors, 4)[:, :, None]
        * (
            pot[:, None, None, None]
            + 1000
            * tf.reduce_sum(tf.constant(DEPOSITS), -1)[None, None, :, None]
        )
        - 1000 * tf.constant(DEPOSITS)[None, None]
    )


def score_likelihood(score_probability, heads, values):
    """Multiply active claims after combining score states with equal payouts."""
    base_probability = tf.reduce_sum(
        tf.where(
            tf.constant(SCORE_TO_BASE, tf.bool),
            score_probability[..., None],
            0.0,
        ),
        axis=-2,
    )
    likelihood = tf.gather(
        tf.reshape(base_probability, (tf.shape(base_probability)[0], -1)),
        tf.maximum(heads, 0) * len(BASE_VALUES) + values,
        axis=1,
        batch_dims=1,
    )
    return tf.reduce_prod(tf.where(heads >= 0, likelihood, 1.0), -1)


def immediate_deposits(inputs, events):
    """Draws accept the declaration; ron and triple abort precede its deposit."""
    before = tf.reduce_sum(
        tf.cast(inputs["immediate_deposit_required"], tf.int32)
        * tf.constant([1, 2, 4, 8]),
        -1,
    )
    after = tf.reduce_sum(
        tf.cast(inputs["deposit_required"], tf.int32)
        * tf.constant([1, 2, 4, 8]),
        -1,
    )
    return tf.one_hot(
        tf.where(
            (events >= 4) & (events <= 20), after[:, None], before[:, None]
        ),
        16,
    )


def public_inputs(features):
    return {
        "scores": tf.round(
            tf.cast(features["seat_context"][:, :, 0], tf.float32) * 100000
        ),
        "honba": tf.cast(features["honba"], tf.float32),
        "pot": tf.cast(features["riichi_sticks"], tf.float32) * 1000,
        "first_dealer": (
            tf.cast(features["dealer_relative_seat"], tf.int32)
            - tf.cast(features["kyoku_number"], tf.int32)
        )
        % 4,
        "hand_number": 4 * tf.cast(features["prevailing_wind"], tf.int32)
        + tf.cast(features["kyoku_number"], tf.int32),
    }


def settlement_states(inputs):
    """Rule-derived outcomes depend only on the position, not its actions."""
    indices = dealer_indices(inputs["dealer"])
    classes = tf.gather(tf.constant(CLASSES, tf.int64), indices)
    transfers = tf.gather(tf.constant(TRANSFERS, tf.float32), indices)
    support = tf.gather(tf.constant(SUPPORT, tf.int64), indices)
    bank = bank_transfers(support[:, :, 2], inputs["pot"])
    delta = tf.linalg.matmul(transfers, tf.constant(COEFFICIENTS))
    delta += inputs["honba"][:, None, None] * tf.linalg.matmul(
        tf.gather(tf.constant(HONBA, tf.float32), indices),
        tf.constant(COEFFICIENTS),
    )
    scores = inputs["scores"][:, None, None] + delta[:, :, None] + bank
    priority = (tf.range(4)[None] - inputs["first_dealer"][:, None]) % 4
    ranks = tf.reduce_sum(
        tf.cast(
            (scores[..., None, :] > scores[..., :, None])
            | (
                (scores[..., None, :] == scores[..., :, None])
                & (
                    priority[:, None, None, None, :]
                    < priority[:, None, None, :, None]
                )
            ),
            tf.int64,
        ),
        -1,
    )
    return {
        "joint_classes": classes,
        "joint_bank": bank,
        "joint_scores": scores,
        "joint_ranks": ranks,
        "joint_support": support,
    }


def event_probabilities(inputs):
    legal = tf.reduce_all(
        (
            ~tf.constant(DEPOSITS, tf.bool)[None]
            | inputs["deposit_allowed"][:, None]
        )
        & (
            tf.constant(DEPOSITS, tf.bool)[None]
            | ~inputs["deposit_required"][:, None]
        ),
        -1,
    )
    # An accepted riichi guarantees tenpai on an exhaustive draw. This
    # includes both prior declarations and deposits predicted in this hand.
    ready = tf.constant(
        [
            [bool(mask & (1 << seat)) for seat in range(4)]
            for _, _, _, mask in EVENTS
        ]
    )
    legal = legal[:, None] & tf.reduce_all(
        (tf.constant([event[0] for event in EVENTS]) != 2)[None, :, None, None]
        | ready[None, :, None]
        | ~(
            inputs["riichi_required"][:, None, None]
            | tf.constant(DEPOSITS, tf.bool)[None, None]
        ),
        -1,
    )
    deposit = tf.nn.softmax(
        tf.where(legal, inputs["deposit_logits"], -1e9)
    ) * tf.cast(legal, tf.float32)
    event = tf.nn.softmax(
        tf.where(tf.reduce_any(legal, -1), inputs["event_logits"], -1e9)
    )
    return event, deposit, legal


def probabilities(inputs):
    """Exact categorical law, without ranks or payment marginals."""
    indices = dealer_indices(inputs["dealer"])
    support = tf.gather(tf.constant(SUPPORT, tf.int64), indices)
    selectors = tf.gather(tf.constant(SELECTORS, tf.int32), indices)
    amount = score_likelihood(
        inputs["score_probability"], selectors[..., 0], selectors[..., 1]
    )
    event, deposit, legal = event_probabilities(inputs)
    probability = (
        tf.gather(event, support[:, :, 5], axis=1, batch_dims=1)
        * amount
        * inputs["immediate_probability"][:, :1]
    )[..., None] * tf.gather(deposit, support[:, :, 5], axis=1, batch_dims=1)

    immediate_event = tf.gather(tf.constant(IMMEDIATE_EVENTS), support[:, :, 5])
    immediate = tf.gather(
        inputs["immediate_probability"],
        tf.maximum(immediate_event, 0),
        axis=1,
        batch_dims=1,
    )
    immediate = tf.where(immediate_event >= 0, immediate, 0.0)
    # For actor-payer ron, the canonical ron cell IDs 0..2 are opponent slots.
    amount = score_likelihood(
        inputs["immediate_score_probability"],
        tf.minimum(selectors[..., 0], 2),
        selectors[..., 1],
    )
    immediate_probability = (immediate * amount)[..., None] * (
        immediate_deposits(inputs, immediate_event)
    )
    probability += immediate_probability
    legal = tf.gather(legal, support[:, :, 5], axis=1, batch_dims=1)
    legal |= immediate_probability > 0
    return probability, legal


def decode(inputs):
    """Rule-derived transfers, ranks and diagnostics for serving."""
    probability, legal = probabilities(inputs)
    log_probability = tf.where(
        probability > 0, tf.math.log(tf.maximum(probability, 1e-30)), -1e30
    )
    indices = dealer_indices(inputs["dealer"])
    transfers = tf.gather(tf.constant(TRANSFERS, tf.float32), indices)
    states = settlement_states(inputs)
    rank_probability = tf.reduce_sum(
        tf.reduce_sum(
            probability[..., None, None] * tf.one_hot(states["joint_ranks"], 4),
            axis=2,
        ),
        axis=1,
    )
    rank_probability /= tf.reduce_sum(rank_probability, axis=-1, keepdims=True)
    return {
        "joint_probability": probability,
        "joint_log_probability": log_probability,
        "joint_legal_deposits": legal,
        **states,
        "round_placement_logits": tf.math.log(
            tf.maximum(rank_probability, 1e-30)
        ),
        "expected_points": expectation(
            probability,
            tf.linalg.matmul(transfers, tf.constant(COEFFICIENTS))[:, :, None]
            + states["joint_bank"],
        )
        / 10000,
    }


def observed_terms(
    labels,
    inputs,
    mean=losses.mean_valid,
    coverage_mean=losses.mean_valid,
):
    """Observed categorical likelihood without a score Cartesian product.

    Ambiguous zero-transfer draws remain marginalized: the stored payment
    label does not identify nobody/all-tenpai or an abort. Unsupported
    liability allocations remain excluded, as in the enumerated decoder.
    """
    event, deposit, legal = event_probabilities(inputs)
    same_score = (
        tf.gather(
            tf.constant(payment_scoring.payment_amounts(), tf.int32),
            tf.cast(inputs["dealer"], tf.int32),
        )
        == tf.gather(
            tf.constant(outcomes.PAYMENT_VALUES, tf.int32),
            labels["payment"][:, :24],
        )[..., None]
    )
    score_match = tf.concat(
        [
            same_score[:, :12],
            tf.reduce_all(tf.gather(same_score, TSUMO_CELLS, axis=1), axis=2),
        ],
        axis=1,
    )
    score = tf.reduce_sum(
        tf.gather(inputs["score_probability"], SCORE_HEADS, axis=1)
        * tf.cast(score_match, tf.float32),
        -1,
    )
    # One representative per event supplies only its active cells/draw
    # transfers; no winning score values are enumerated here.
    active = TRANSFERS[EVENT_ROWS, :24] != 0
    selected = active[:, SCORE_HEADS]
    event_score_likelihood = tf.reduce_prod(
        tf.where(tf.constant(selected)[None], score[:, None], 1.0), -1
    )
    compatible = tf.reduce_all(
        (labels["payment"][:, None, :24] != 1) == tf.constant(active)[None], -1
    )
    compatible &= tf.reduce_all(
        labels["payment"][:, None, 24:36]
        == tf.constant(CLASSES[EVENT_ROWS, 24:36], tf.int32)[None],
        -1,
    )
    compatible &= tf.reduce_all(
        ~tf.constant(selected)[None] | tf.reduce_any(score_match, -1)[:, None],
        -1,
    )
    bank = bank_transfers(
        tf.constant(SUPPORT[EVENT_ROWS, 2])[None], inputs["pot"]
    )
    actual_bank = tf.gather(
        payment_values(labels), labels["payment"][:, 36:], axis=1, batch_dims=1
    )
    bank_match = tf.reduce_all(bank == actual_bank[:, None, None], -1)
    future = (
        event * event_score_likelihood * inputs["immediate_probability"][:, :1]
    )[..., None] * deposit
    immediate = tf.gather(
        inputs["immediate_probability"],
        tf.maximum(tf.constant(IMMEDIATE_EVENTS), 0),
        axis=1,
    )
    immediate *= tf.cast(tf.constant(IMMEDIATE_EVENTS >= 0), tf.float32)
    immediate_score = tf.reduce_sum(
        inputs["immediate_score_probability"]
        * tf.cast(score_match[:, :3], tf.float32),
        -1,
    )
    immediate *= tf.reduce_prod(
        tf.where(
            tf.constant(selected[:, :3])[None], immediate_score[:, None], 1.0
        ),
        -1,
    )
    immediate = immediate[..., None] * immediate_deposits(
        inputs, tf.constant(IMMEDIATE_EVENTS)[None]
    )
    matches = compatible[..., None] & bank_match & (legal | (immediate > 0))
    supported = tf.reduce_any(matches, axis=(1, 2))
    likelihood = tf.reduce_sum(
        tf.where(matches, future + immediate, 0.0), (1, 2)
    )
    return {
        "payment_nll": mean(
            -tf.math.log(tf.maximum(likelihood, 1e-30)), supported
        ),
        "settlement_supported_fraction": coverage_mean(
            tf.cast(supported, tf.float32), tf.ones_like(supported)
        ),
        "settlement_multi_ron_fraction": coverage_mean(
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
            ),
            tf.ones_like(supported),
        ),
    }


def expectation(probability, values):
    """Sum deposits before scored events to avoid long FP32 channel sums."""
    if values.shape.rank == 4:
        probability = probability[..., None]
    return tf.reduce_sum(tf.reduce_sum(probability * values, axis=2), axis=1)


@tf.function(jit_compile=True)
def continuation_payoffs(scores, ranks, support, dealer, hand_number, pot):
    return continuation_forecast(
        scores=scores,
        ranks=ranks,
        support=support,
        dealer=dealer,
        hand_number=hand_number,
        pot=pot,
    )[0]


@tf.function(jit_compile=True)
def continuation_forecast(scores, ranks, support, dealer, hand_number, pot):
    """Public-state payoff of each settlement, shared by every action."""
    dealer = tf.cast(dealer, tf.int64)
    tenpai = tf.gather(
        tf.constant(DEPOSITS, tf.bool), tf.minimum(support[:, :, 3], 15)
    )
    repeat = (
        (
            tf.bitwise.bitwise_and(
                support[:, :, 4],
                tf.bitwise.left_shift(tf.ones_like(dealer), dealer)[:, None],
            )
            > 0
        )
        | (support[:, :, 1] == 3)
        | (
            (support[:, :, 1] == 2)
            & tf.gather(tenpai, dealer, axis=2, batch_dims=1)
        )
    )
    dealer_rank = tf.gather(ranks, dealer, axis=3, batch_dims=1)
    terminal = (
        tf.reduce_any(scores < 0, -1)
        | (
            (hand_number[:, None, None] >= 7)
            & (support[:, :, 1, None] != 3)
            & (tf.reduce_max(scores, -1) >= 30000)
            & (~repeat[:, :, None] | (dealer_rank == 0))
        )
        | ((hand_number[:, None, None] >= 11) & ~repeat[:, :, None])
    )
    phase = hand_number[:, None] + tf.cast(~repeat, tf.int32)
    next_dealer = (dealer[:, None] + tf.cast(~repeat, tf.int64)) % 4
    terminal_bank = tf.where(
        support[:, :, 1, None] < 2,
        0.0,
        pot[:, None, None]
        + 1000 * tf.reduce_sum(tf.constant(DEPOSITS), -1)[None, None],
    )
    return post_hand_forecast(
        scores=scores,
        ranks=ranks,
        next_dealer=next_dealer[:, :, None],
        next_hand=phase[:, :, None],
        terminal=terminal,
        pot=terminal_bank,
    )


def post_hand_forecast(scores, ranks, next_dealer, next_hand, terminal, pot):
    """Common observed/predicted hand settlement plus fixed continuation."""
    rounds = tf.cast(tf.maximum(8 - next_hand, 0), tf.float32)
    future = tf.range(8, dtype=tf.int64)
    dealerships = tf.reduce_sum(
        tf.cast(
            (tf.cast(future, tf.float32) < rounds[..., None])[..., None]
            & (
                ((next_dealer[..., None] + future) % 4)[..., None]
                == tf.range(4, dtype=tf.int64)
            ),
            tf.float32,
        ),
        axis=-2,
    )
    smooth, probability = continuation_values(scores, rounds, dealerships)
    offsets = tf.constant(outcomes.SAINT3_JADE_SOUTH_OFFSETS, tf.float32)
    terminal_points = scores[..., 0] + tf.where(ranks[..., 0] == 0, pot, 0.0)
    exact = tf.math.ceil(
        (terminal_points - 25000) / 1000 + tf.gather(offsets, ranks[..., 0])
    )
    return tf.where(terminal, exact, smooth), tf.where(
        terminal[..., None], tf.one_hot(ranks[..., 0], 4), probability
    )


def continuation_values(scores, rounds, dealerships):
    """Frozen seed-11 calibration from st3-continuation-20261003.

    Score location and diffusion were calibrated in 10,000-point units.
    Positive dealership drift and a + cT + dT² variance impose the agreed
    monotonicity constraints. Constants are never training variables.
    """
    coefficient = tf.nn.softplus(
        tf.constant([-3.6169434, -2.544114, -1.7547616, -5.1666236], tf.float32)
    )
    location = scores / 10000 + coefficient[0] * (
        dealerships - tf.reduce_mean(dealerships, -1, keepdims=True)
    )
    noise = tf.sqrt(
        coefficient[1]
        + coefficient[2] * rounds
        + coefficient[3] * tf.square(rounds)
    )
    probability = placement_prior(location, 0, noise[..., None])
    return (
        location[..., 0] * 10
        - 25
        + 0.5
        + tf.reduce_sum(
            probability
            * tf.constant(outcomes.SAINT3_JADE_SOUTH_OFFSETS, tf.float32),
            -1,
        )
    ), probability


def marginals(outputs):
    """Class marginals; four state-dependent atoms encode the bank exactly."""
    probability = outputs["joint_probability"]
    batch = tf.shape(probability)[0]
    count = len(outcomes.PAYMENT_VALUES) + 4
    base = tf.reduce_sum(probability, -1)
    transfers = tf.reshape(
        tf.math.unsorted_segment_sum(
            tf.reshape(
                tf.broadcast_to(
                    base[..., None], tf.shape(outputs["joint_classes"])
                ),
                [-1],
            ),
            tf.reshape(
                tf.cast(
                    tf.range(batch)[:, None, None] * 36
                    + tf.range(36)[None, None],
                    tf.int64,
                )
                * count
                + outputs["joint_classes"],
                [-1],
            ),
            batch * 36 * count,
        ),
        (batch, 36, count),
    )
    # Sum by pot collector first, avoiding a long scatter-add for every
    # score combination that has the same bank movement.
    # Tree reductions avoid the long float32 dot-product accumulation that
    # can move bank probability mass outside the export tolerance on GPU.
    bank_probability = tf.stack(
        [
            tf.reduce_sum(
                tf.where(
                    outputs["joint_support"][:, :, 2, None] == seat,
                    probability,
                    0.0,
                ),
                axis=1,
            )
            for seat in range(-1, 4)
        ],
        axis=1,
    )
    winner = tf.one_hot(tf.range(5) - 1, 4, dtype=tf.int32)
    deposits = tf.constant(DEPOSITS, tf.int32)
    bank_classes = tf.where(
        winner[:, None] > 0,
        len(outcomes.PAYMENT_VALUES)
        + tf.reduce_sum(deposits, -1)[None, :, None]
        - deposits[None],
        1 - deposits[None],
    )
    bank = tf.reduce_sum(
        tf.where(
            bank_classes[..., None] == tf.range(count),
            bank_probability[..., None, None],
            0.0,
        ),
        axis=(1, 2),
    )
    # Almost every transfer is zero. Recover its complement instead of
    # accumulating thousands of tiny zero-class masses in arbitrary order.
    nonzero = transfers * (1 - tf.one_hot(1, count))
    transfers = nonzero + tf.maximum(
        0.0, 1 - tf.reduce_sum(nonzero, -1, keepdims=True)
    ) * tf.one_hot(1, count)
    return tf.concat([transfers, bank], axis=1)


def payment_values(inputs):
    return tf.concat(
        [
            tf.broadcast_to(
                tf.constant(outcomes.PAYMENT_VALUES, tf.float32),
                [tf.shape(inputs["pot"])[0], len(outcomes.PAYMENT_VALUES)],
            ),
            inputs["pot"][:, None] + 1000 * tf.range(4, dtype=tf.float32)[None],
        ],
        -1,
    )


def payment_labels(points, public):
    """Map raw remaining transfers onto the shared public-pot alphabet."""
    matches = (
        tf.cast(points[..., None], tf.float32)
        == payment_values(public)[:, None]
    )
    tf.debugging.Assert(
        tf.reduce_all(tf.reduce_any(matches, -1)),
        [
            "Unsupported payment targets (row, cell), points, public pots:",
            tf.where(~tf.reduce_any(matches, -1)),
            tf.boolean_mask(points, ~tf.reduce_any(matches, -1)),
            tf.boolean_mask(
                public["pot"], ~tf.reduce_all(tf.reduce_any(matches, -1), -1)
            ),
        ],
    )
    return {
        "payment": tf.argmax(
            tf.cast(matches, tf.int32), -1, output_type=tf.int32
        ),
        "pot": public["pot"],
    }


def settlement_moments(outputs):
    """Exact honba-free player/bank moments, retaining transfer covariance."""
    net = (
        tf.linalg.matmul(
            tf.gather(
                tf.constant(outcomes.PAYMENT_VALUES, tf.float32),
                outputs["joint_classes"],
            ),
            tf.constant(COEFFICIENTS),
        )[:, :, None]
        + outputs["joint_bank"]
    )
    net = tf.concat([net, -tf.reduce_sum(net, -1, keepdims=True)], -1)
    mean = expectation(outputs["joint_probability"], net)
    second = expectation(outputs["joint_probability"], net**2)
    return mean, second


def placement_prior(scores, remaining, scale=0.0472):
    """Coherent actor rank probabilities from a Gumbel future-score prior.

    Scores and scale must share units; leading batch dimensions broadcast.
    Enumerate only opponent orders, avoiding the 24-permutation score tensor.
    """
    remaining = tf.clip_by_value(tf.cast(remaining, tf.float32), 0.0, 7.0)
    scores = tf.cast(scores, tf.float32)
    location = (scores - scores[..., :1]) / (
        scale * tf.sqrt(remaining[..., None] + 1)
    )
    total = tf.reduce_logsumexp(location, axis=-1)
    first = tf.exp(-total)
    second = tf.zeros_like(first)
    fourth = tf.zeros_like(first)
    for seat in (1, 2, 3):
        after_first = tf.reduce_logsumexp(
            tf.gather(location, [i for i in range(4) if i != seat], axis=-1),
            axis=-1,
        )
        second += tf.exp(location[..., seat] - total - after_first)
        for other in (i for i in (1, 2, 3) if i != seat):
            last = 6 - seat - other
            fourth += tf.exp(
                tf.reduce_sum(location[..., 1:], axis=-1)
                - total
                - after_first
                - tf.nn.softplus(location[..., last])
            )
    probability = tf.stack(
        [first, second, tf.maximum(1 - first - second - fourth, 0), fourth],
        axis=-1,
    )
    return probability / tf.reduce_sum(probability, axis=-1, keepdims=True)
