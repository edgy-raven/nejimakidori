"""Joint han/fu states settle directly into the existing payment classes."""

import numpy
import tensorflow as tf

from model import outcomes

SCORES = outcomes.SCORE_STATES


def score_class(han, fu, yakuman):
    return SCORES.index(
        (0, han // 13)
        if yakuman
        else (
            (max(tier for tier in (5, 6, 8, 11, 13) if tier <= han), 0)
            if han >= 5
            else (han, fu)
        )
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
# ron, dealer ron. True yakuman states encode their multiplicity.
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
    8000 * SCORE_FU * numpy.array([1, 2, 4, 6])[:, None],
    100
    * (
        (
            numpy.array([1, 2, 4, 6])[:, None]
            * numpy.array(
                [outcomes.basic_points(han, fu) for han, fu in SCORES]
            )
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


def minimum_han_mask(minimum):
    # Tier states represent intervals: e.g. haneman includes 6 and 7 han.
    maximum = numpy.array(
        [
            {6: 7, 8: 10, 11: 12, 13: numpy.inf}.get(han, han)
            for han in SCORE_HAN
        ]
    )
    return (tf.constant(SCORE_HAN) == 0) | (
        tf.constant(maximum, tf.float32)
        >= tf.cast(minimum[..., None], tf.float32)
        + tf.constant(
            numpy.where(SCORE_FU == 25, 2, SCORE_FU == 20), tf.float32
        )
    )


def masked_score_logits(logits, dealer, meld_fu, scenarios):
    indices = tf.gather(
        tf.constant(PAYMENT_ENVIRONMENTS, tf.int32),
        tf.cast(dealer, tf.int32),
    ) * 129 + tf.gather(meld_fu, CELLS[:, 2], axis=1)
    valid = tf.gather(tf.constant(LEGAL_SCORE_STATES), indices)
    return tf.where(
        tf.gather(
            valid,
            [2 if s[3] == 4 else int(s[0] != 0) for s in scenarios],
            axis=2,
        )
        & minimum_han_mask(
            tf.constant(
                [
                    [
                        max(mode - 1, 0)
                        + ura
                        + ippatsu
                        + int(special in (1, 2, 3))
                        + int(kind == 1 and mode != 0)
                        for mode, ura, ippatsu, special in scenarios
                    ]
                    for kind, _, _ in CELLS
                ]
            )
        )[None],
        logits,
        -1e9,
    )


def payment_probabilities(logits, dealer, meld_fu, scenarios):
    logits = masked_score_logits(logits, dealer, meld_fu, scenarios)
    return logits, score_payments(tf.nn.softmax(logits), dealer)


def score_payments(probability, dealer):
    """Push legal score probabilities for leading canonical payment cells."""
    matches = (
        payment_amounts()[:, : probability.shape[1], :, None]
        == outcomes.PAYMENT_VALUES
    )
    mapped = tf.gather(
        tf.constant(matches.argmax(-1), tf.int32), tf.cast(dealer, tf.int32)
    )
    # Sparse pushforward avoids a dense score-state x payment-class matrix
    # multiplication for every scoring scenario.
    count = tf.shape(probability)[0] * tf.shape(probability)[1]
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
    return tf.transpose(
        tf.reshape(
            payments,
            (
                tf.shape(probability)[0],
                tf.shape(probability)[1],
                len(outcomes.PAYMENT_VALUES),
                tf.shape(probability)[2],
            ),
        ),
        [0, 1, 3, 2],
    )


class CompletionScoreMask(tf.keras.layers.Layer):
    def call(self, values):
        logits, inputs = values
        fu = public_fu(inputs)
        opened = tf.reduce_any(
            (inputs["meld_valid"] > 0) & (inputs["meld_type"] != 4), -1
        )
        required = inputs["riichi_state"] > 0
        table = tf.constant(LEGAL_SCORE_STATES.reshape(4, 129, 3, -1))
        modes = []
        for environment in (2, 0):
            allowed = tf.gather(table[environment], fu)
            modes.append(
                (allowed[..., 0, :] & ~required[..., None])
                | (
                    allowed[..., 1, :]
                    & ~opened[..., None]
                    & minimum_han_mask(
                        tf.cast(required, tf.int32) + int(environment == 0)
                    )
                )
            )
        allowed = tf.stack(modes, axis=2)
        return tf.where(allowed, logits, -1e9)
