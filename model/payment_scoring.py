"""Joint han/fu states settle directly into the existing payment classes."""

import itertools

import numpy
import tensorflow as tf

from . import outcomes

# Ordinary han is capped at kazoe yakuman. True yakuman uses its actual
# transfer amount, preserving multiple yakuman and liability allocations.
SCORES = (
    tuple(itertools.product(range(1, 5), (20, 25, *range(30, 141, 10))))
    + tuple((han, 0) for han in range(5, 14))
    + tuple(
        (0, int(value))
        for value in outcomes.PAYMENT_VALUES
        if value > 0 and value % 8000 == 0
    )
)


def basic_points(han, fu):
    if han >= 13:
        return 8000
    if han >= 11:
        return 6000
    if han >= 8:
        return 4000
    if han >= 6:
        return 3000
    if han >= 5:
        return 2000
    return min(2000, fu * 2 ** (han + 2))


def score_class(han, fu, yakuman, payment):
    return SCORES.index(
        (0, payment) if yakuman else (min(han, 13), 0 if han >= 5 else fu)
    )


def public_fu(inputs):
    tiles = inputs["meld_tile_ids"][:, :, :, 0]
    return tf.reduce_sum(
        tf.gather(
            tf.constant([0, 0, 2, 8, 16, 8]),
            tf.cast(inputs["meld_type"], tf.int32),
        )
        * (
            1
            + tf.cast(
                (tiles >= 27) | (tiles % 9 == 0) | (tiles % 9 == 8), tf.int32
            )
        )
        * tf.cast(inputs["meld_valid"], tf.int32),
        axis=-1,
    )


# Settlement factors: nondealer tsumo share, dealer tsumo share, ordinary
# ron, dealer ron. True yakuman states already encode allocated payments.
CELLS = numpy.argwhere(outcomes.PAYMENT_CELLS)[:24]
PAYMENT_ENVIRONMENTS = numpy.where(
    CELLS[:, 0] == 0,
    2 + (CELLS[:, 2] == numpy.arange(4)[:, None]),
    (CELLS[:, 1] == numpy.arange(4)[:, None])
    | (CELLS[:, 2] == numpy.arange(4)[:, None]),
)
SCORE_HAN, SCORE_FU = numpy.array(SCORES).T
PAYMENTS_BY_FACTOR = numpy.where(
    SCORE_HAN == 0,
    SCORE_FU,
    100
    * (
        (
            numpy.array([1, 2, 4, 6])[:, None]
            * numpy.array([basic_points(han, fu) for han, fu in SCORES])
            + 99
        )
        // 100
    ),
)

# Fu is bounded by four exposed concealed terminal/honor kans (4 * 32).
# Axis order: settlement environment, public fu, scoring mode, score state.
# Modes are open, closed, and true-yakuman-only.
PUBLIC_FU = numpy.arange(129)[None, :, None, None]
RON = numpy.array([False, False, True, True])[:, None, None, None]
CLOSED = numpy.array([False, True, False])[None, None, :, None]
LEGAL_SCORE_STATES = (
    (
        (
            (
                (SCORE_HAN >= 5)
                | (
                    (SCORE_FU >= 30)
                    & (
                        SCORE_FU
                        >= 20 + PUBLIC_FU + 10 * (RON & CLOSED) + 2 * ~RON
                    )
                )
                | (
                    (SCORE_FU == 25)
                    & CLOSED
                    & (PUBLIC_FU == 0)
                    & (SCORE_HAN >= numpy.where(RON, 2, 3))
                )
                | (
                    (SCORE_FU == 20)
                    & ~RON
                    & CLOSED
                    & (PUBLIC_FU == 0)
                    & (SCORE_HAN >= 2)
                )
            )
            & (numpy.arange(3)[None, None, :, None] != 2)
        )
        | (SCORE_HAN == 0)
    )
    & (PAYMENTS_BY_FACTOR[..., None] == outcomes.PAYMENT_VALUES).any(-1)[
        :, None, None
    ]
).reshape(-1, 3, len(SCORES))


def payment_amounts():
    return PAYMENTS_BY_FACTOR[PAYMENT_ENVIRONMENTS]


def masked_score_logits(logits, dealer, meld_fu, scenarios, cell_ids=None):
    cells = numpy.arange(24) if cell_ids is None else list(cell_ids)
    indices = tf.gather(
        tf.constant(PAYMENT_ENVIRONMENTS[:, cells], tf.int32),
        tf.cast(dealer, tf.int32),
    ) * 129 + tf.gather(meld_fu, CELLS[cells, 2], axis=1)
    valid = tf.gather(tf.constant(LEGAL_SCORE_STATES), indices)
    return tf.where(
        tf.gather(
            valid,
            [2 if s[3] == 4 else int(s[0] != 0) for s in scenarios],
            axis=2,
        ),
        logits,
        -1e9,
    )


def payment_moments(logits, dealer, meld_fu, scenarios, cell_ids):
    """Exact first and second raw moments without a payment histogram."""
    logits = masked_score_logits(
        logits=logits,
        dealer=dealer,
        meld_fu=meld_fu,
        scenarios=scenarios,
        cell_ids=cell_ids,
    )
    amounts = tf.gather(
        tf.constant(payment_amounts()[:, list(cell_ids)], tf.float32),
        tf.cast(dealer, tf.int32),
    )
    weights = tf.exp(logits - tf.reduce_max(logits, axis=-1, keepdims=True))
    totals = tf.reduce_sum(
        weights[..., None]
        * tf.stack(
            [tf.ones_like(amounts), amounts, tf.square(amounts)], axis=-1
        )[:, :, None],
        axis=-2,
    )
    return logits, totals[..., 1:] / totals[..., :1]


def payment_probabilities(logits, dealer, meld_fu, scenarios):
    logits = masked_score_logits(logits, dealer, meld_fu, scenarios)
    matches = payment_amounts()[..., None] == outcomes.PAYMENT_VALUES
    mapped = tf.gather(
        tf.constant(matches.argmax(-1), tf.int32), tf.cast(dealer, tf.int32)
    )
    probability = tf.nn.softmax(logits)
    # Sparse pushforward avoids a dense score-state x payment-class matrix
    # multiplication for every scoring scenario.
    count = tf.shape(probability)[0] * 24
    indices = tf.reshape(mapped, (count, len(SCORES))) + tf.range(count)[
        :, None
    ] * len(outcomes.PAYMENT_VALUES)
    payments = tf.math.unsorted_segment_sum(
        tf.reshape(
            tf.transpose(probability, [0, 1, 3, 2]),
            (-1, tf.shape(probability)[2]),
        ),
        tf.reshape(indices, (-1,)),
        count * len(outcomes.PAYMENT_VALUES),
    )
    payments = tf.transpose(
        tf.reshape(
            payments,
            (-1, 24, len(outcomes.PAYMENT_VALUES), tf.shape(probability)[2]),
        ),
        [0, 1, 3, 2],
    )
    return logits, payments
