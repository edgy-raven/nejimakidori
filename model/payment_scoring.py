"""Public yaku constraints and joint han/fu payment scoring."""

import numpy
import tensorflow as tf

from log_dataset import scoring
from model import outcomes

# Unlisted pairs may coexist; yakuman replaces ordinary scored yaku.
EXCLUSIVE = numpy.zeros((len(scoring.NAMES), len(scoring.NAMES)), bool)
for name, others in {
    "pinfu": (
        "haku hatsu chun seat_wind round_wind honroutou toitoi sanankou "
        "sankantsu sanshoku_doukou chiitoitsu shousangen"
    ),
    "tanyao": (
        "haku hatsu chun seat_wind round_wind chanta honroutou shousangen "
        "honitsu junchan"
    ),
    "iipeikou": (
        "toitoi sanankou sankantsu sanshoku_doukou chiitoitsu ryanpeikou "
        "honroutou"
    ),
    "sanshoku_doujun": (
        "ittsuu honroutou toitoi sanankou sankantsu sanshoku_doukou "
        "chiitoitsu shousangen honitsu chinitsu ryanpeikou"
    ),
    "ittsuu": (
        "chanta honroutou toitoi sanankou sankantsu sanshoku_doukou "
        "chiitoitsu shousangen junchan ryanpeikou"
    ),
    "chanta": ("honroutou toitoi chiitoitsu junchan chinitsu"),
    "honroutou": "junchan ryanpeikou chinitsu",
    "toitoi": "chiitoitsu junchan ryanpeikou",
    "sanankou": "chiitoitsu ryanpeikou",
    "sankantsu": "chiitoitsu ryanpeikou",
    "sanshoku_doukou": "chiitoitsu shousangen honitsu chinitsu ryanpeikou",
    "chiitoitsu": "shousangen ryanpeikou haku hatsu chun seat_wind round_wind",
    "shousangen": "junchan chinitsu ryanpeikou",
    "honitsu": "junchan chinitsu",
    "junchan": "haku hatsu chun seat_wind round_wind",
    "ryanpeikou": "haku hatsu chun seat_wind round_wind",
    "chinitsu": "haku hatsu chun seat_wind round_wind",
    "yakuman": " ".join(scoring.NAMES[:-1]),
}.items():
    for other in others.split():
        EXCLUSIVE[scoring.NAMES.index(name), scoring.NAMES.index(other)] = True
        EXCLUSIVE[scoring.NAMES.index(other), scoring.NAMES.index(name)] = True


def exclusive_cliques(
    group=(),
    candidates=frozenset(range(len(scoring.NAMES))),
    excluded=frozenset(),
):
    """Maximal mutually exclusive sets, built once from the scored-yaku graph."""
    if not candidates and not excluded:
        yield group
    while candidates:
        vertex = min(candidates)
        neighbors = frozenset(numpy.flatnonzero(EXCLUSIVE[vertex]))
        yield from exclusive_cliques(
            group + (vertex,), candidates & neighbors, excluded & neighbors
        )
        candidates = candidates - {vertex}
        excluded = excluded | {vertex}


EXCLUSIVE_CLIQUES = numpy.array(
    [
        [index in group for index in range(len(scoring.NAMES))]
        for group in exclusive_cliques()
    ],
    numpy.float32,
)


def support(melds):
    """Twenty-three named-yaku masks; concealed future tiles stay unknown."""
    melds = tf.cast(melds, tf.float32)
    tiles = tf.range(34)
    sequence = melds[..., 0] > 0
    triplet = tf.reduce_any(melds[..., 1:5] > 0, axis=-1)
    used = sequence | triplet
    any_meld = tf.reduce_any(used, axis=-1)
    opened = tf.reduce_any(
        tf.gather(melds, [0, 1, 2, 4], axis=-1) > 0, axis=(-1, -2)
    )
    no_sequence = ~tf.reduce_any(sequence, axis=-1)
    terminal = (tiles >= 27) | (tiles % 9 == 0) | (tiles % 9 == 8)
    simple = ~tf.reduce_any(
        (triplet & terminal)
        | (sequence & ((tiles % 9 == 0) | (tiles % 9 == 6))),
        axis=-1,
    )
    outside = ~tf.reduce_any(
        (triplet & ~terminal)
        | (sequence & (tiles % 9 != 0) & (tiles % 9 != 6)),
        axis=-1,
    )
    honor = tf.reduce_any(used[..., 27:], axis=-1)
    suits = tf.reduce_sum(
        tf.cast(
            tf.reduce_any(
                tf.reshape(
                    used[..., :27], tf.concat([tf.shape(used)[:-1], [3, 9]], 0)
                ),
                axis=-1,
            ),
            tf.int32,
        ),
        axis=-1,
    )
    flush = suits <= 1
    groups = tf.reduce_sum(melds[..., :5], axis=(-1, -2))
    dragons = tf.reduce_sum(melds[..., 31:34, 1:5], axis=(-1, -2))
    winds = tf.reduce_sum(melds[..., 27:31, 1:5], axis=(-1, -2))
    green = tf.reduce_all(~sequence | (tiles == 19), axis=-1) & tf.reduce_all(
        ~triplet
        | tf.reduce_any(tiles[:, None] == [19, 20, 21, 23, 25, 32], -1),
        axis=-1,
    )
    yakuman = (
        ~opened  # Suuankou; no-meld hands also retain kokushi/chuuren.
        | (groups - dragons <= 1)  # Daisangen.
        | (groups - winds <= 1)  # Shousuushii/daisuushii.
        | green
        | no_sequence  # Four kans remain possible by upgrading pons.
    )
    # A three-group yaku must fit its missing groups in the unfixed slots.
    sanshoku = tf.reduce_any(
        tf.reduce_sum(
            tf.cast(
                tf.gather(
                    sequence, [[i, i + 9, i + 18] for i in range(7)], axis=-1
                ),
                tf.float32,
            ),
            -1,
        )
        + (4 - groups[..., None])
        >= 3,
        -1,
    )
    ittsuu = tf.reduce_any(
        tf.reduce_sum(
            tf.cast(
                tf.gather(
                    sequence, [[i, i + 3, i + 6] for i in (0, 9, 18)], axis=-1
                ),
                tf.float32,
            ),
            -1,
        )
        + (4 - groups[..., None])
        >= 3,
        -1,
    )
    doukou = tf.reduce_any(
        tf.reduce_sum(
            tf.cast(
                tf.gather(
                    triplet, [[i, i + 9, i + 18] for i in range(9)], axis=-1
                ),
                tf.float32,
            ),
            -1,
        )
        + (4 - groups[..., None])
        >= 3,
        -1,
    )
    return tf.stack(
        [
            ~opened & ~tf.reduce_any(triplet, -1),  # Pinfu.
            simple,
            ~opened & (groups <= 2),
            triplet[..., 31] | (groups < 4),
            triplet[..., 32] | (groups < 4),
            triplet[..., 33] | (groups < 4),
            (winds > 0) | (groups < 4),
            (winds > 0) | (groups < 4),
            sanshoku,
            ittsuu,
            outside & (~no_sequence | (groups < 4)),
            outside & no_sequence,
            no_sequence,
            tf.reduce_sum(tf.gather(melds, [0, 1, 2, 4], axis=-1), (-1, -2))
            <= 1,
            tf.reduce_sum(melds[..., 0], -1) <= 1,
            doukou,
            ~any_meld,
            (dragons <= 2) & (groups - dragons <= 2),
            flush,
            outside & ~honor & (~no_sequence | (groups < 4)),
            ~any_meld,
            flush & ~honor,
            yakuman,
        ],
        axis=-1,
    )


def public_melds(inputs):
    """Public groups in the same start-tile encoding as candidate melds."""
    starts = tf.reduce_min(
        tf.where(inputs["meld_tile_valid"] > 0, inputs["meld_tile_ids"], 34),
        axis=-1,
    )
    return tf.reduce_sum(
        tf.one_hot(starts, 34)[..., None]
        * tf.one_hot(tf.cast(inputs["meld_type"], tf.int32) - 1, 5)[
            ..., None, :
        ]
        * tf.cast(inputs["meld_valid"], tf.float32)[..., None, None],
        axis=2,
    )


class CompletionMask(tf.keras.layers.Layer):
    def call(self, values):
        logits, inputs = values
        allowed = support(public_melds(inputs))[:, :, None]
        logits = tf.where(allowed, logits, -1e9)
        # Each connected conflict is a scored-yaku exclusion, not a route.
        # For yaku i, competing alternatives contribute to its complement.
        # Thus every mutually exclusive clique has total probability <= 1.
        competing = tf.where(tf.constant(EXCLUSIVE), logits[..., None, :], -1e9)
        complement = tf.reduce_logsumexp(
            tf.concat([tf.zeros_like(logits[..., None]), competing], -1), -1
        )
        return logits - complement


SCORES = outcomes.SCORE_STATES


def score_class(han, fu, yakuman):
    if yakuman:
        return SCORES.index((0, han // 13))
    if han >= 5:
        return SCORES.index(
            (max(tier for tier in (5, 6, 8, 11, 13) if tier <= han), 0)
        )
    return SCORES.index(
        (5, 0) if outcomes.basic_points(han, fu) == 2000 else (han, fu)
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
    & (RON | (SCORE_HAN != 1) | (SCORE_FU != 110))
    & (PAYMENTS_BY_FACTOR[..., None] == outcomes.PAYMENT_VALUES).any(-1)[
        :, None, None
    ]
).reshape(-1, 3, len(SCORES))


def payment_amounts():
    return PAYMENTS_BY_FACTOR[PAYMENT_ENVIRONMENTS]


def minimum_han_mask(minimum, bonus=0):
    # Tier states represent intervals: e.g. haneman includes 6 and 7 han.
    maximum = numpy.array(
        [
            {6: 7, 8: 10, 11: 12, 13: numpy.inf}.get(han, han)
            for han in SCORE_HAN
        ]
    )
    return (tf.constant(SCORE_HAN) == 0) | (
        tf.constant(maximum, tf.float32)
        >= tf.cast(bonus, tf.float32)[..., None]
        + tf.maximum(
            1.0,
            tf.cast(minimum[..., None], tf.float32)
            + tf.constant(
                numpy.where(SCORE_FU == 25, 2, SCORE_FU == 20), tf.float32
            ),
        )
    )


def masked_score_logits(
    logits, dealer, meld_fu, bonus_han, opened, riichi_required, bonus, yakuman
):
    indices = tf.gather(
        tf.constant(PAYMENT_ENVIRONMENTS, tf.int32),
        tf.cast(dealer, tf.int32),
    ) * 129 + tf.gather(meld_fu, CELLS[:, 2], axis=1)
    valid = tf.gather(tf.constant(LEGAL_SCORE_STATES), indices)
    public_bonus = tf.gather(bonus, CELLS[:, 2], axis=1)[:, :, None]
    # Before riichi a closed hand can still open. Retain the union of legal
    # final scores without predicting opening or bonus-source categories.
    open_valid = (
        valid[:, :, 0, None]
        & ~tf.gather(riichi_required, CELLS[:, 2], axis=1)[..., None, None]
        & (tf.constant(bonus_han) <= 1)[None, None, :, None]
        & minimum_han_mask(tf.constant(bonus_han), bonus=public_bonus)
    )
    closed_valid = (
        valid[:, :, 1, None]
        & ~tf.gather(opened, CELLS[:, 2], axis=1)[..., None, None]
        & minimum_han_mask(
            tf.constant(bonus_han)[None, None]
            + tf.constant(CELLS[:, 0] == 1, tf.int32)[None, :, None],
            bonus=public_bonus,
        )
    )
    return tf.where(
        (open_valid | closed_valid)
        & (
            (tf.constant(SCORE_HAN) != 0)
            | tf.gather(yakuman, CELLS[:, 2], axis=1)[..., None]
        )[:, :, None],
        logits,
        -1e9,
    )


def score_payments(probability, dealer):
    """Push legal score probabilities for leading canonical payment cells."""
    matches = (
        payment_amounts()[:, : probability.shape[1], :, None]
        == outcomes.PAYMENT_VALUES
    )
    mapped = tf.gather(
        tf.constant(matches.argmax(-1), tf.int64), tf.cast(dealer, tf.int64)
    )
    # Sparse pushforward avoids a dense score-state x payment-class matrix
    # multiplication for every scoring scenario.
    # Some int32 arithmetic kernels require host memory; keep these expanded
    # indices on device through both the pushforward and its gradient.
    count = (
        tf.shape(probability, out_type=tf.int64)[0]
        * tf.shape(probability, out_type=tf.int64)[1]
    )
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
        fu = tf.cast(inputs["public_fu"], tf.int32)
        opened = tf.reduce_any(
            (inputs["meld_valid"] > 0) & (inputs["meld_type"] != 4), -1
        )
        required = inputs["riichi_state"] > 0
        table = tf.constant(LEGAL_SCORE_STATES.reshape(4, 129, 3, -1))
        modes = []
        for environment in (2, 0):
            allowed = tf.gather(table[environment], fu)
            modes.append(
                (
                    allowed[..., 0, :]
                    & ~required[..., None]
                    & minimum_han_mask(
                        tf.zeros_like(fu),
                        bonus=tf.cast(inputs["public_bonus"], tf.int32),
                    )
                )
                | (
                    allowed[..., 1, :]
                    & ~opened[..., None]
                    & minimum_han_mask(
                        tf.cast(required, tf.int32) + int(environment == 0),
                        bonus=tf.cast(inputs["public_bonus"], tf.int32),
                    )
                )
            )
        allowed = tf.stack(modes, axis=2)
        allowed &= (
            (tf.constant(SCORE_HAN) != 0)
            | support(public_melds(inputs))[..., -1, None]
        )[:, :, None]
        return tf.where(allowed, logits, -1e9)


class CompletionScoreCoupling(tf.keras.layers.Layer):
    """Lift score tails to the lower bounds implied by the same completion.

    Compatible yaku marginals are never added as if independent. Exclusive
    alternatives can be summed; the strongest exclusive-set bound wins.
    """

    def call(self, values):
        logits, yaku_logits, inputs = values
        yaku = tf.sigmoid(yaku_logits)
        # Current closed hands may open later. Only declared riichi and
        # closed-only yaku guarantee closed han and menzen-tsumo bonuses.
        required = inputs["riichi_state"] > 0
        closed_only = tf.constant(
            [
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
            ]
        )
        closed = required[..., None] | closed_only
        # Shousangen also guarantees two dragon triplets (four han total).
        minimum = (
            tf.cast(inputs["public_bonus"], tf.int32)[..., None]
            + tf.cast(required[..., None], tf.int32)
            + tf.where(
                closed,
                tf.constant(
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        4,
                        3,
                        3,
                        3,
                        6,
                        0,
                    ]
                ),
                tf.constant(
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        4,
                        2,
                        2,
                        3,
                        5,
                        0,
                    ]
                ),
            )
        )[:, :, None] + (
            tf.cast(closed, tf.int32)[:, :, None]
            * tf.constant([0, 1])[None, None, :, None]
        )
        levels = tf.constant(
            [(1, 2, 3, 4, 5, 6, 8, 11, 13, 0).index(han) for han in SCORE_HAN]
        )
        # Collapsed haneman/baiman/sanbaiman states include 7/10/12 han.
        upper_han = tf.constant(
            [
                {6: 7, 8: 10, 11: 12, 13: numpy.inf}.get(han, han)
                for han in SCORE_HAN
            ]
        )
        normal = tf.constant(SCORE_HAN > 0)
        possible = (
            (logits[..., None, :] > -1e8)
            & normal
            & (upper_han >= tf.cast(minimum[..., None], tf.float32))
        )
        # The yakuman label denotes a true yakuman, not counted yakuman.
        possible = tf.where(
            tf.range(23)[:, None] == 22,
            (logits[..., None, :] > -1e8) & ~normal,
            possible,
        )
        floor = tf.reduce_min(tf.where(possible, levels, 9), -1)
        required_tail = tf.minimum(
            1.0,
            tf.reduce_max(
                tf.einsum(
                    "...yt,cy->...ct",
                    yaku[..., None]
                    * tf.cast(floor[..., None] > tf.range(9), tf.float32),
                    tf.constant(EXCLUSIVE_CLIQUES),
                ),
                axis=-2,
            ),
        )
        # Keep true-yakuman mass identical across heads. Project the
        # remaining conditional distribution from low thresholds to high;
        # raising a later tail cannot undo any earlier lower bound.
        conditional_tail = tf.clip_by_value(
            tf.math.divide_no_nan(
                required_tail - yaku[..., -1:], 1 - yaku[..., -1:]
            ),
            0.0,
            1.0,
        )
        ordinary = tf.where(normal, logits, -1e9)
        for boundary in range(8):
            below = normal & (levels <= boundary)
            above = normal & (levels > boundary)
            lower = tf.reduce_logsumexp(tf.where(below, ordinary, -1e9), -1)
            upper = tf.reduce_logsumexp(tf.where(above, ordinary, -1e9), -1)
            bound = conditional_tail[..., boundary]
            # Log-space rescaling retains every feasible score when the
            # bound is below one, even if its starting mass underflows.
            adjustment = tf.where(
                below,
                (tf.math.log(tf.maximum(1 - bound, 1e-30)) - lower)[..., None],
                (tf.math.log(tf.maximum(bound, 1e-30)) - upper)[..., None],
            )
            ordinary = tf.where(
                normal & (logits > -1e8),
                tf.where(
                    (bound > tf.sigmoid(upper - lower))[..., None],
                    ordinary + adjustment,
                    ordinary,
                ),
                -1e9,
            )
        mass = tf.where(normal, 1 - yaku[..., -1:], yaku[..., -1:])
        conditional = tf.where(
            normal,
            tf.nn.log_softmax(ordinary),
            tf.nn.log_softmax(tf.where(~normal, logits, -1e9)),
        )
        return tf.where(
            (mass > 0) & (logits > -1e8),
            conditional + tf.math.log(tf.maximum(mass, 1e-30)),
            -1e9,
        )
