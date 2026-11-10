"""Action-conditioned settlements, immediate outcomes and policy values."""

import numpy
import tensorflow as tf

from log_dataset import scoring
from model import actions, joint_settlement, losses, outcomes, payment_scoring

STATISTICS_CHUNK = 512


def public_inputs(features, selected=None):
    """Guaranteed scoring constraints given an attempted action.

    Scores and the existing pot remain pre-action: settlement deposits account
    for a declaration only after its discard survives ron.
    """
    opened = tf.reduce_any(
        (features["meld_valid"] > 0) & (features["meld_type"] != 4), axis=-1
    )
    public = {
        "opened": opened,
        "has_meld": tf.reduce_any(features["meld_valid"] > 0, axis=-1),
        "deposit_allowed": ~opened
        & (features["riichi_state"] != 2)
        & (features["seat_context"][:, :, 0] >= 0.01),
        "deposit_required": features["riichi_state"] == 1,
        "riichi_required": features["riichi_state"] > 0,
        "ippatsu_allowed": (features["riichi_state"] == 0)
        | (features["ippatsu_alive"] > 0),
        "dealer": features["dealer_relative_seat"],
        "public_fu": tf.cast(features["public_fu"], tf.int32),
        "public_bonus": tf.cast(features["public_bonus"], tf.int32),
        "yakuman_allowed": payment_scoring.support(
            payment_scoring.public_melds(features)
        )[..., -1],
    }
    if selected is None:
        return public
    public = {
        name: tf.gather(value, selected[:, 0]) for name, value in public.items()
    }
    response = tf.gather(features["decision_phase"], selected[:, 0]) == 1
    hands = tf.stack(
        [
            selected[:, 0],
            tf.cast(
                tf.gather_nd(
                    actions.candidate_hand_indices(features), selected
                ),
                tf.int32,
            ),
        ],
        axis=1,
    )
    # Float32 keeps both gathers and slices on device; these small integer
    # counts are exact. TensorFlow's int32 StridedSlice uses host memory.
    melds = tf.gather_nd(
        tf.cast(features["candidate_melds"], tf.float32), hands
    )
    own_opened = tf.reduce_any(
        tf.gather(melds, [0, 1, 2, 4], axis=2) > 0, axis=(1, 2)
    )
    declaration = (selected[:, 1] >= 74) & (selected[:, 1] < 111)
    own = tf.stack(
        [
            tf.range(tf.shape(selected)[0]),
            tf.zeros(tf.shape(selected)[0], tf.int32),
        ],
        axis=1,
    )
    public["opened"] = tf.tensor_scatter_nd_update(
        public["opened"],
        own,
        tf.where(response, public["opened"][:, 0], own_opened),
    )
    for name in ("public_fu", "public_bonus"):
        public[name] = tf.tensor_scatter_nd_update(
            public[name],
            own,
            tf.where(
                response,
                public[name][:, 0],
                tf.cast(
                    tf.gather_nd(
                        tf.cast(features["candidate_" + name], tf.float32),
                        hands,
                    ),
                    tf.int32,
                ),
            ),
        )
    public["yakuman_allowed"] = tf.tensor_scatter_nd_update(
        public["yakuman_allowed"],
        own,
        tf.where(
            response,
            public["yakuman_allowed"][:, 0],
            payment_scoring.support(melds)[..., -1],
        ),
    )
    public["deposit_allowed"] &= ~public["opened"]
    for name in ("deposit_required", "riichi_required"):
        public[name] = tf.tensor_scatter_nd_update(
            public[name], own, public[name][:, 0] | declaration
        )
        # A competing ron can end the hand before a pending declaration
        # is accepted. Hypothetical call features remain in learned context.
        public[name] &= ~(
            response[:, None]
            & (tf.gather(features["riichi_state"], selected[:, 0]) == 1)
        )
    public["immediate_ankan"] = (selected[:, 1] >= 261) & (
        tf.reduce_sum(melds[:, :, 3], axis=-1)
        > tf.reduce_sum(
            tf.cast(
                tf.gather(
                    (features["meld_type"][:, 0] == 4)
                    & (features["meld_valid"][:, 0] > 0),
                    selected[:, 0],
                ),
                tf.float32,
            ),
            axis=-1,
        )
    )
    # Immediate endings precede any new deposit. Already accepted deposits
    # are included in the input scores/pot, never charged a second time.
    public["immediate_deposit_required"] = tf.zeros_like(
        public["deposit_required"]
    )
    public["immediate_ippatsu"] = public["ippatsu_allowed"]
    # Only an unconditional own discard can force a draw at this boundary.
    # Response passes still await opponents' claims, including last-tile ron.
    discard = actions.discard_decisions(features, selected) & ~response
    tile = tf.where(selected[:, 1] < 111, selected[:, 1] % 37, 0)
    abort = (
        tf.gather_nd(
            features["discard_abort"],
            tf.stack(
                [
                    selected[:, 0],
                    tf.cast(declaration, tf.int32),
                    tile,
                ],
                axis=1,
            ),
        )
        > 0
    )
    public["immediate_draw_kind"] = tf.where(
        discard & abort,
        2,
        tf.where(
            discard
            & (tf.gather(features["live_wall_count"], selected[:, 0]) == 0),
            1,
            0,
        ),
    )
    public["immediate_tenpai"] = tf.concat(
        [
            tf.cast(
                tf.gather_nd(
                    features["discard_action_all_shanten"],
                    tf.stack([selected[:, 0], tile], axis=1),
                )
                == 1,
                tf.int32,
            )[:, None],
            tf.where(public["riichi_required"][:, 1:], 1, -1),
        ],
        axis=1,
    )
    return public


class PublicContext(tf.keras.layers.Layer):
    def call(self, values):
        inputs, selected, structural = values
        tiles = tf.cast(structural[:, -1], tf.int32)
        # Code 37 denotes an action without a tile. Red fives retain their
        # identity in the one-hot feature, but share public safety with fives.
        base = tf.where(
            tiles == 37, 0, tf.where(tiles < 34, tiles, 4 + 9 * (tiles - 34))
        )
        tile_indices = tf.stack([selected[:, 0], base], axis=-1)
        has_tile = tf.cast(tiles != 37, tf.float32)[:, None]
        # Eighteen two-sided wait lines: 14, 25, 36, 47, 58, 69 in each suit.
        # Genbutsu at either endpoint rules out that entire ryanmen line.
        lower = tf.constant(
            [9 * suit + rank for suit in range(3) for rank in range(6)]
        )
        live_lines = ~(
            tf.gather(inputs["genbutsu_to_seat"], lower, axis=2) > 0
        ) & ~(tf.gather(inputs["genbutsu_to_seat"], lower + 3, axis=2) > 0)
        remaining = tf.gather(
            tf.reduce_sum(tf.cast(live_lines, tf.float32), axis=2),
            selected[:, 0],
        )
        tile_lines = (
            tf.reduce_sum(
                tf.cast(tf.gather(live_lines, selected[:, 0]), tf.float32)
                * tf.cast(
                    (base[:, None] == lower) | (base[:, None] == lower + 3),
                    tf.float32,
                )[:, None],
                axis=2,
            )
            * has_tile
        )
        return tf.concat(
            [
                tf.one_hot(tiles, 38),
                tf.one_hot(
                    tf.searchsorted(
                        [37, 74, 111, 112, 223, 260, 261],
                        selected[:, 1],
                        side="right",
                    ),
                    8,
                ),
                tf.cast(
                    tf.gather(inputs["riichi_state"], selected[:, 0]),
                    tf.float32,
                )
                / 2,
                tf.cast(
                    tf.gather(inputs["ippatsu_alive"], selected[:, 0]),
                    tf.float32,
                ),
                tf.cast(
                    tf.gather(
                        inputs["opponent_called_tile_count"], selected[:, 0]
                    ),
                    tf.float32,
                )
                / 16,
                tf.cast(
                    tf.gather(inputs["opponent_discard_count"], selected[:, 0]),
                    tf.float32,
                )
                / 24,
                tf.one_hot(
                    tf.cast(
                        tf.gather(
                            inputs["dealer_relative_seat"], selected[:, 0]
                        ),
                        tf.int32,
                    ),
                    4,
                ),
                *[
                    tf.cast(
                        tf.gather_nd(
                            tf.transpose(inputs[name], [0, 2, 1]), tile_indices
                        ),
                        tf.float32,
                    )
                    * has_tile
                    / scale
                    for name, scale in (
                        ("genbutsu_to_seat", 1),
                        ("sotogawa_earliest_turn_to_seat", 6),
                    )
                ],
                tf.cast(
                    tf.gather_nd(inputs["blocker_counts"], tile_indices),
                    tf.float32,
                )
                * has_tile
                / 4,
                *[
                    tf.cast(
                        tf.gather_nd(inputs[name], tile_indices), tf.float32
                    )[:, None]
                    * has_tile
                    / 4
                    for name in (
                        "known_unavailable_counts",
                        "dora_multiplicity",
                    )
                ],
                remaining / 18,
                tile_lines / 2,
                tf.math.divide_no_nan(tile_lines, remaining),
                tf.cast(
                    tf.gather_nd(
                        tf.transpose(
                            inputs["passed_unchanged_to_seat"], [0, 2, 1]
                        ),
                        tile_indices,
                    ),
                    tf.float32,
                )
                * has_tile,
                tf.reshape(
                    tf.cast(
                        tf.gather(inputs["riichi_push_count"], selected[:, 0]),
                        tf.float32,
                    ),
                    (-1, 16),
                )
                / 24,
            ],
            axis=-1,
        )


class HandContext(tf.keras.layers.Layer):

    def call(self, values):
        inputs, selected, structural = values
        hands = tf.stack(
            [
                selected[:, 0],
                tf.cast(
                    tf.gather_nd(
                        actions.candidate_hand_indices(inputs),
                        selected,
                    ),
                    tf.int32,
                ),
            ],
            axis=1,
        )
        melds = tf.gather_nd(
            tf.cast(inputs["candidate_melds"], tf.float32), hands
        )
        red = tf.reduce_sum(
            tf.gather_nd(
                tf.cast(inputs["candidate_hand_red"], tf.float32), hands
            ),
            axis=-1,
        ) + tf.reduce_sum(melds[:, :, 5], axis=-1)
        counts = (
            tf.gather_nd(
                tf.cast(inputs["candidate_hand_counts"], tf.float32), hands
            )
            + 3 * melds[:, :, 1]
            + 4 * tf.reduce_sum(melds[:, :, 2:5], axis=-1)
            + tf.add_n(
                [tf.roll(melds[:, :, 0], shift, axis=1) for shift in range(3)]
            )
        )
        dora = (
            tf.reduce_sum(
                counts
                * tf.cast(
                    tf.gather(inputs["dora_multiplicity"], selected[:, 0]),
                    tf.float32,
                ),
                axis=-1,
            )
            + red
        )
        return tf.concat(
            [
                tf.stack(
                    [
                        dora,
                        red,
                        tf.cast(
                            tf.gather(
                                inputs["live_wall_count"], selected[:, 0]
                            ),
                            tf.float32,
                        )
                        / 70,
                        tf.reduce_sum(
                            4
                            - tf.cast(
                                tf.gather(
                                    inputs["known_unavailable_counts"],
                                    selected[:, 0],
                                ),
                                tf.float32,
                            ),
                            axis=-1,
                        )
                        / 136,
                    ],
                    axis=-1,
                ),
                tf.math.asinh(tf.cast(structural[:, :-1], tf.float32)),
            ],
            axis=-1,
        )


class ImmediateOutcome(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(dtype="float32")
        self.state = tf.keras.layers.Dense(128, activation="gelu")
        self.eligibility = tf.keras.layers.Dense(3)
        self.score = tf.keras.layers.Dense(3 * len(payment_scoring.SCORES))
        self.draw = tf.keras.layers.Dense(16)

    def call(self, inputs):
        hidden = self.state(inputs["state"])
        probability = tf.minimum(
            inputs["immediate_wait"] * tf.sigmoid(self.eligibility(hidden)),
            1 - 1e-7,
        )
        raw_logits = tf.math.log(
            tf.maximum(probability, 1e-30)
        ) - tf.math.log1p(-probability)
        logits = tf.where(
            inputs["immediate_safe"]
            | (inputs["immediate_wait"] <= 0)
            | (inputs["immediate_ankan"][:, None] & inputs["has_meld"][:, 1:]),
            -1e9,
            raw_logits,
        )
        ron = tf.sigmoid(logits) * tf.cast(
            inputs["immediate_allowed"][:, None], tf.float32
        )
        # Separate opponent events coexist. Their conditional Bernoulli
        # product preserves all three marginals; no winner competes away
        # another claim. Three claims settle as a native triple-ron abort.
        masks = tf.constant(
            [
                [bool(mask & (1 << seat)) for seat in range(3)]
                for mask in range(8)
            ]
        )
        claims = tf.reduce_prod(
            tf.where(masks[None], ron[:, None], 1 - ron[:, None]), -1
        )
        branches = tf.gather(claims, [0, 1, 2, 4], axis=1)
        triple = claims[:, 7]
        score_logits = tf.reshape(
            self.score(hidden), (-1, 3, len(payment_scoring.SCORES))
        )
        environment = (
            tf.gather(
                tf.constant(
                    payment_scoring.PAYMENT_ENVIRONMENTS[:, :3], tf.int32
                ),
                tf.cast(inputs["dealer"], tf.int32),
            )
            * 129
            + inputs["public_fu"][:, 1:]
        )
        score_allowed = tf.gather(
            tf.gather(
                tf.constant(payment_scoring.LEGAL_SCORE_STATES), environment
            ),
            tf.cast(~inputs["opened"][:, 1:], tf.int32),
            axis=2,
            batch_dims=2,
        )
        minimum_han = tf.cast(inputs["riichi_required"][:, 1:], tf.int32) + (
            tf.cast(
                inputs["riichi_required"][:, 1:]
                & inputs["immediate_ippatsu"][:, 1:],
                tf.int32,
            )
        )
        score_allowed &= payment_scoring.minimum_han_mask(
            minimum_han, bonus=inputs["public_bonus"][:, 1:]
        )
        score_allowed &= (tf.constant(payment_scoring.SCORE_HAN) != 0) | inputs[
            "yakuman_allowed"
        ][:, 1:, None]
        # Only kokushi can rob a concealed kan under the native rules.
        score_allowed &= ~inputs["immediate_ankan"][:, None, None] | (
            (tf.constant(payment_scoring.SCORE_HAN, tf.int32) == 0)[None, None]
            & (tf.constant(payment_scoring.SCORE_FU, tf.int32)[None, None] == 1)
        )
        raw_score = score_logits
        score_logits = tf.where(score_allowed, score_logits, -1e9)
        amount_probability = payment_scoring.score_payments(
            tf.nn.softmax(score_logits)[:, :, None], inputs["dealer"]
        )[:, :, 0]
        draw_allowed = tf.reduce_all(
            (inputs["immediate_tenpai"][:, None] < 0)
            | (
                inputs["immediate_tenpai"][:, None]
                == tf.constant(joint_settlement.DEPOSITS, tf.int32)[None]
            ),
            axis=-1,
        )
        raw_draw = self.draw(hidden)
        draw_logits = tf.where(draw_allowed, raw_draw, -1e9)
        draw_probability = tf.nn.softmax(draw_logits)
        exhaustive = inputs["immediate_draw_kind"] == 1
        abort = inputs["immediate_draw_kind"] == 2
        return {
            "immediate_constraint_loss": (
                tf.cast(inputs["immediate_allowed"], tf.float32)
                * (
                    losses.constraint_divergence(
                        raw_logits, logits, binary=True
                    )
                    + tf.reduce_mean(
                        losses.constraint_divergence(raw_score, score_logits),
                        -1,
                    )
                )
                + tf.cast(exhaustive, tf.float32)
                * losses.constraint_divergence(raw_draw, draw_logits)
            ),
            "immediate_logits": logits,
            "conditional_ron_eligibility": tf.math.divide_no_nan(
                tf.sigmoid(logits), inputs["immediate_wait"]
            ),
            "immediate_amount_probability": amount_probability,
            "immediate_score_probability": tf.nn.softmax(score_logits),
            "immediate_ron_probability": ron,
            "immediate_claim_probability": claims,
            "immediate_draw_logits": draw_logits,
            # Continue, three actor-payer ron outcomes, sixteen exhaustive
            # tenpai masks, ordinary abort, triple-ron abort. Ordinary draws
            # occur only if the discard survives.
            "immediate_probability": tf.concat(
                [
                    branches[:, :1]
                    * tf.cast(~(exhaustive | abort), tf.float32)[:, None],
                    branches[:, 1:],
                    branches[:, :1]
                    * tf.cast(exhaustive, tf.float32)[:, None]
                    * draw_probability,
                    (branches[:, 0] * tf.cast(abort, tf.float32))[:, None],
                    triple[:, None],
                    tf.gather(claims, [3, 5, 6], axis=1),
                ],
                axis=-1,
            ),
        }


def immediate_loss_terms(labels, selected, structure, public, outputs):
    tiles = tf.minimum(tf.cast(structure[:, -1], tf.int32), 36)
    indices = tf.stack([selected[:, 0], tiles], axis=1)
    legal = tf.gather_nd(tf.transpose(labels["ron_legal"], [0, 2, 1]), indices)
    payments = tf.gather_nd(
        tf.transpose(labels["ron_payments"], [0, 2, 1]), indices
    )
    amount = tf.reduce_sum(
        outputs["immediate_amount_probability"]
        * tf.cast(
            tf.constant(outcomes.PAYMENT_VALUES, tf.int32)[None, None]
            == (payments - 300 * public["honba"][:, None])[..., None],
            tf.float32,
        ),
        axis=-1,
    )
    ready = tf.gather(labels["opponent_shanten"], selected[:, 0])
    draw = public["immediate_tenpai"][:, 0] + tf.reduce_sum(
        tf.cast(ready == 0, tf.int32) * tf.constant([2, 4, 8]), axis=-1
    )
    return {
        "risk_terms": (
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(tf.maximum(legal, 0), tf.float32),
                logits=outputs["immediate_logits"],
            ),
            public["immediate_allowed"][:, None] & (legal >= 0),
        ),
        "immediate_amount_terms": (
            -tf.math.log(tf.maximum(amount, 1e-30)),
            public["immediate_allowed"][:, None]
            & (legal == 1)
            & (payments > 0),
        ),
        "immediate_draw_terms": (
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=draw, logits=outputs["immediate_draw_logits"]
            )[:, None],
            (
                (public["immediate_draw_kind"] == 1)
                & tf.reduce_all(ready >= 0, axis=-1)
            )[:, None],
        ),
    }


class PaymentCritic(tf.keras.Model):
    """Sparse action states in; payment distributions and derived values out.

    The encoder must end before the actor's old payment head. There is no
    learned scalar value or state-payment predictor in this model.
    """

    def __init__(self):
        super().__init__(autocast=False)
        self.state = tf.keras.layers.Dense(128, activation="gelu")
        self.cell_state = tf.keras.layers.Dense(16 * 64, activation="gelu")
        self.deposit = tf.keras.layers.Dense(
            len(joint_settlement.EVENTS) * 16, dtype="float32"
        )
        self.event = tf.keras.layers.Dense(
            len(joint_settlement.EVENTS), dtype="float32"
        )
        self.bonus = tf.keras.layers.Dense(
            len(scoring.BONUS_HAN), dtype="float32"
        )
        self.bonus_embedding = self.add_weight(
            shape=(len(scoring.BONUS_HAN), 64),
            initializer="glorot_uniform",
            name="bonus_embedding",
        )
        self.hand_score = tf.keras.layers.Dense(
            len(payment_scoring.SCORES), dtype="float32"
        )
        self.immediate = ImmediateOutcome()

    def distributions(self, inputs):
        """Scored ron/tsumo events, exhaustive masks, abort and bank deposits.

        Twelve ron claimant slots plus four tsumo winners. All three tsumo
        payer shares use exactly the same score state and added-han count.
        """
        hidden = self.state(inputs["state"])
        cells = tf.reshape(self.cell_state(hidden), (-1, 16, 64))
        winning = tuple(range(12)) + tuple(
            next(
                i
                for i, cell in enumerate(scoring.CELLS[:24])
                if cell[0] == 1 and cell[2] == winner
            )
            for winner in range(4)
        )
        shares = [
            cell if cell < 12 else 12 + row[2]
            for cell, row in enumerate(scoring.CELLS[:24])
        ]
        bonus_logits = self.bonus(cells[:, : len(winning)])
        opened = tf.gather(
            inputs["opened"],
            [scoring.CELLS[cell][2] for cell in winning],
            axis=1,
        )
        required = tf.gather(
            inputs["riichi_required"],
            [scoring.CELLS[cell][2] for cell in winning],
            axis=1,
        )
        allowed = ~opened[..., None] | (tf.constant(scoring.BONUS_HAN) <= 1)
        allowed &= ~required[..., None] | (tf.constant(scoring.BONUS_HAN) >= 1)
        raw_bonus = bonus_logits
        bonus_logits = tf.where(allowed, bonus_logits, -1e9)
        known_fu = tf.gather(
            inputs["public_fu"],
            [scoring.CELLS[cell][2] for cell in winning],
            axis=1,
        )
        bonus_state = tf.nn.gelu(
            cells[:, : len(winning), None] + self.bonus_embedding[None, None]
        )
        score_logits = self.hand_score(
            tf.concat(
                [
                    bonus_state,
                    tf.broadcast_to(
                        tf.cast(known_fu, tf.float32)[:, :, None, None] / 128,
                        tf.concat([tf.shape(bonus_state)[:-1], [1]], axis=0),
                    ),
                ],
                axis=-1,
            )
        )
        # Decode a tsumo hand once; payer shares use the same logits.
        score_logits = tf.gather(score_logits, shares, axis=1)
        bonus_logits = tf.gather(bonus_logits, shares, axis=1)
        raw_score = score_logits
        score_logits = payment_scoring.masked_score_logits(
            logits=score_logits,
            dealer=inputs["dealer"],
            meld_fu=inputs["public_fu"],
            bonus_han=scoring.BONUS_HAN,
            opened=inputs["opened"],
            riichi_required=inputs["riichi_required"],
            bonus=inputs["public_bonus"],
            yakuman=inputs["yakuman_allowed"],
        )
        result = {
            "score_logits": score_logits,
            "score_probability": tf.reduce_sum(
                tf.nn.softmax(bonus_logits)[..., None]
                * tf.nn.softmax(score_logits),
                axis=2,
            ),
            "bonus_logits": bonus_logits,
            "event_logits": self.event(hidden),
            "deposit_logits": tf.reshape(
                self.deposit(hidden), (-1, len(joint_settlement.EVENTS), 16)
            ),
            **self.immediate(inputs),
        }

        _, _, legal = joint_settlement.event_probabilities({**inputs, **result})
        possible = tf.reduce_any(legal, -1)
        result["critic_constraint_loss"] = (
            result["immediate_constraint_loss"]
            + tf.reduce_mean(
                losses.constraint_divergence(
                    raw_bonus, tf.where(allowed, raw_bonus, -1e9)
                ),
                -1,
            )
            + tf.reduce_mean(
                losses.constraint_divergence(raw_score, score_logits), (1, 2)
            )
            + losses.constraint_divergence(
                result["event_logits"],
                tf.where(possible, result["event_logits"], -1e9),
            )
            + tf.math.divide_no_nan(
                tf.reduce_sum(
                    losses.constraint_divergence(
                        result["deposit_logits"],
                        tf.where(legal, result["deposit_logits"], -1e9),
                    )
                    * tf.cast(possible, tf.float32),
                    -1,
                ),
                tf.reduce_sum(tf.cast(possible, tf.float32), -1),
            )
        )
        return result

    def joint_distributions(self, inputs, weights=None):
        if weights is None:
            factors = self.distributions(inputs)
        else:
            with tf.keras.StatelessScope(
                state_mapping=zip(self.trainable_variables, weights)
            ):
                factors = PaymentCritic.distributions(self, inputs)
        result = {**factors, **joint_settlement.decode({**inputs, **factors})}
        result["payment_probability"] = joint_settlement.marginals(result)
        return result

    def projected_mean_std(self, inputs):
        """ST3 moments of the action's projected current-hand settlement."""
        forecast = joint_settlement.decode(
            {**inputs, **PaymentCritic.distributions(self, inputs)}
        )
        payoff = joint_settlement.continuation_payoffs(
            scores=forecast["joint_scores"],
            ranks=forecast["joint_ranks"],
            support=forecast["joint_support"],
            dealer=inputs["dealer"],
            hand_number=inputs["hand_number"],
            pot=inputs["pot"],
        )
        mean = joint_settlement.expectation(
            forecast["joint_probability"], payoff
        )
        return tf.stack(
            [
                mean,
                tf.sqrt(
                    joint_settlement.expectation(
                        forecast["joint_probability"],
                        tf.square(payoff - mean[:, None, None]),
                    )
                ),
            ],
            axis=1,
        )

    def chunk_statistics(self, inputs, labels, weights=None):
        # Inline the differentiable body into the chunk compilation. Nested
        # compiled gradients can specialize on forward tensor values.
        if weights is None:
            outputs = PaymentCritic.distributions(self, inputs)
        else:
            with tf.keras.StatelessScope(
                state_mapping=zip(self.trainable_variables, weights)
            ):
                outputs = PaymentCritic.distributions(self, inputs)

        def mean(values, mask):
            return losses.masked_totals(
                values,
                mask,
                tf.reshape(
                    labels["valid"], [-1] + [1] * (values.shape.rank - 1)
                ),
            )

        metrics = joint_settlement.observed_terms(
            labels, {**inputs, **outputs}, mean=mean, coverage_mean=mean
        )
        # Fresh self-play has realized transfers, without counterfactual
        # han/fu labels. Both paths train the same settlement likelihood.
        if "score" in labels:
            metrics["score_nll"] = score_cross_entropy(
                labels["score"], outputs["score_logits"], mean=mean
            )
            # Rank-3 cross-entropy inserts a dynamic shape assertion, making
            # the entire compiled forward/backward chunk stateful.
            metrics["bonus_han_nll"] = mean(
                tf.reshape(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.reshape(tf.maximum(labels["bonus"], 0), [-1]),
                        logits=tf.reshape(
                            outputs["bonus_logits"],
                            [-1, len(scoring.BONUS_HAN)],
                        ),
                    ),
                    tf.shape(labels["bonus"]),
                ),
                labels["bonus"] >= 0,
            )
        else:
            metrics["score_nll"] = tf.zeros(2)
            metrics["bonus_han_nll"] = tf.zeros(2)
        metrics["critic_constraint_loss"] = mean(
            outputs["critic_constraint_loss"],
            tf.ones_like(outputs["critic_constraint_loss"]),
        )
        return tf.stack([metrics[name] for name in CRITIC_METRICS])

    def chunk_gradients(self, inputs, labels, weights, upstream, gradients):
        """Keep recomputation, differentiation and accumulation together."""
        with tf.GradientTape() as tape:
            tape.watch([inputs["state"], inputs["immediate_wait"], *weights])
            statistics = PaymentCritic.chunk_statistics(
                self, inputs, labels, weights
            )
        values = tape.gradient(
            statistics,
            [inputs["state"], inputs["immediate_wait"], *weights],
            output_gradients=upstream,
        )
        return (
            values[0],
            values[1],
            [total + value for total, value in zip(gradients, values[2:])],
        )

    def supervised_statistics(self, inputs, labels, valid=None):
        labels = {
            **labels,
            "valid": (
                tf.ones(tf.shape(inputs["state"])[0])
                if valid is None
                else tf.cast(valid, tf.float32)
            ),
        }

        # Retain only loss sufficient statistics. Recompute one chunk at a
        # time in the backward pass instead of retaining every settlement.
        # Explicit weight tensors preserve the local replica's dependency;
        # implicit custom-gradient variable capture can select replica zero.
        @tf.custom_gradient
        def summarize(state, wait, *weights):
            statistics = tf.zeros((len(CRITIC_METRICS), 2))
            for begin in tf.range(0, tf.shape(state)[0], STATISTICS_CHUNK):
                statistics += self.chunk_statistics(
                    {
                        **{
                            name: value[begin : begin + STATISTICS_CHUNK]
                            for name, value in inputs.items()
                        },
                        "state": state[begin : begin + STATISTICS_CHUNK],
                        "immediate_wait": wait[
                            begin : begin + STATISTICS_CHUNK
                        ],
                    },
                    {
                        name: value[begin : begin + STATISTICS_CHUNK]
                        for name, value in labels.items()
                    },
                    weights,
                )

            @tf.function
            def backward(upstream):
                state_gradients = tf.TensorArray(
                    tf.float32, size=0, dynamic_size=True
                )
                wait_gradients = tf.TensorArray(
                    tf.float32, size=0, dynamic_size=True
                )
                gradients = [tf.zeros_like(value) for value in weights]
                for begin in tf.range(0, tf.shape(state)[0], STATISTICS_CHUNK):
                    values = self.chunk_gradients(
                        inputs={
                            **{
                                name: value[begin : begin + STATISTICS_CHUNK]
                                for name, value in inputs.items()
                            },
                            "state": state[begin : begin + STATISTICS_CHUNK],
                            "immediate_wait": wait[
                                begin : begin + STATISTICS_CHUNK
                            ],
                        },
                        labels={
                            name: value[begin : begin + STATISTICS_CHUNK]
                            for name, value in labels.items()
                        },
                        weights=weights,
                        upstream=upstream,
                        gradients=gradients,
                    )
                    state_gradients = state_gradients.write(
                        begin // STATISTICS_CHUNK, values[0]
                    )
                    wait_gradients = wait_gradients.write(
                        begin // STATISTICS_CHUNK, values[1]
                    )
                    gradients = values[2]
                return (
                    state_gradients.concat(),
                    wait_gradients.concat(),
                    *gradients,
                )

            return statistics, backward

        return summarize(
            inputs["state"],
            inputs["immediate_wait"],
            *[tf.convert_to_tensor(v) for v in self.trainable_variables],
        )

    def call(self, inputs):
        outputs = self.joint_distributions(inputs)
        return {
            **outputs,
            "terminal_payment_probability": tf.einsum(
                "ncv,ctpr->ntprv",
                outputs["payment_probability"],
                tf.constant(
                    numpy.eye(64, dtype=numpy.float32)[
                        numpy.ravel_multi_index(
                            numpy.array(scoring.CELLS).T, (4, 4, 4)
                        )
                    ].reshape(-1, 4, 4, 4)
                ),
            )
            + tf.cast(~tf.constant(outcomes.PAYMENT_CELLS), tf.float32)[
                None, ..., None
            ]
            * tf.one_hot(1, len(outcomes.PAYMENT_VALUES) + 4),
        }


def score_cross_entropy(labels, logits, mean=losses.mean_valid):
    """Average valid added-han targets within each observed payment cell."""
    labels = tf.cast(labels, tf.int64)
    indices = tf.range(tf.size(labels, out_type=tf.int64)) * tf.shape(
        logits, out_type=tf.int64
    )[-1] + tf.reshape(tf.maximum(labels, 0), (-1,))
    nll = -tf.reshape(
        tf.gather(tf.reshape(tf.nn.log_softmax(logits), (-1,)), indices),
        tf.shape(labels),
    )
    weights = tf.cast(labels >= 0, tf.float32)
    per_cell = tf.math.divide_no_nan(
        tf.reduce_sum(nll * weights, -1), tf.reduce_sum(weights, -1)
    )
    return mean(per_cell, tf.reduce_any(weights > 0, -1))


CRITIC_METRICS = (
    "payment_nll",
    "score_nll",
    "bonus_han_nll",
    "settlement_multi_ron_fraction",
    "settlement_supported_fraction",
    "critic_constraint_loss",
)


def settlement_metrics(statistics, mean=losses.mean_valid):
    return {
        name: mean(
            tf.math.divide_no_nan(
                statistics[index, 0],
                statistics[index, 1],
            ),
            statistics[index, 1] > 0,
            statistics[index, 1],
        )
        for index, name in enumerate(CRITIC_METRICS)
    }


def objective_metrics(outputs, mean=losses.mean_valid):
    metrics = settlement_metrics(outputs["critic_statistics"], mean)
    for name, terms in (
        ("risk_loss", "risk_terms"),
        ("immediate_amount_nll", "immediate_amount_terms"),
        ("immediate_draw_nll", "immediate_draw_terms"),
    ):
        metrics[name] = losses.candidate_mean(
            *outputs[terms], outputs["immediate_rows"], mean=mean
        )
    loss = (
        metrics["payment_nll"]
        + metrics["critic_constraint_loss"]
        + metrics["risk_loss"]
        + metrics["immediate_amount_nll"]
        + metrics["immediate_draw_nll"]
    )
    return loss, metrics


def chosen_actions(features, labels):
    return tf.where(
        features["decision_phase"] == 4,
        260,
        tf.where(
            labels["discard_policy_logits"] >= 0,
            tf.cast(labels["discard_policy_logits"], tf.int32),
            tf.where(
                labels["riichi_policy_logits"] >= 0,
                37 + tf.cast(labels["riichi_policy_logits"], tf.int32),
                tf.where(
                    labels["response_policy_logits"] >= 0,
                    tf.where(
                        labels["response_policy_logits"] == 149,
                        261 + tf.cast(features["trigger_tile_id"], tf.int32),
                        111
                        + tf.cast(labels["response_policy_logits"], tf.int32),
                    ),
                    260 + tf.cast(labels["kan_action_logits"], tf.int32),
                ),
            ),
        ),
    )


def active_candidates(features, selected):
    phase = tf.gather(features["decision_phase"], selected[:, 0])
    action = selected[:, 1]
    return (
        ((phase == 0) & (action < 37))
        | ((phase == 3) & (action >= 37) & (action < 111))
        | (
            (phase == 1)
            & (
                ((action >= 111) & (action < 260))
                | (
                    action
                    == 261
                    + tf.cast(
                        tf.gather(features["trigger_tile_id"], selected[:, 0]),
                        tf.int32,
                    )
                )
            )
        )
        | ((phase == 2) & (action >= 260))
        | ((phase == 4) & (action == 260))
    )


def candidate_probabilities(features, outputs):
    probability = {
        name: tf.nn.softmax(outputs[name])
        * tf.cast(features["decision_phase"][:, None] == phase, tf.float32)
        for name, phase in (
            ("discard_policy_logits", 0),
            ("riichi_policy_logits", 3),
            ("response_policy_logits", 1),
            ("kan_action_logits", 2),
        )
    }
    # Response daiminkan shares a candidate with the trigger tile's kan.
    probability = tf.tensor_scatter_nd_add(
        tf.concat(
            [
                probability["discard_policy_logits"],
                probability["riichi_policy_logits"],
                probability["response_policy_logits"][:, :149],
                probability["kan_action_logits"],
            ],
            axis=1,
        ),
        tf.stack(
            [
                tf.range(tf.shape(features["decision_phase"])[0]),
                261 + tf.cast(features["trigger_tile_id"], tf.int32),
            ],
            axis=1,
        ),
        probability["response_policy_logits"][:, 149],
    )
    # Win/pass exposes a current-hand forecast, not a fabricated
    # action-conditioned forecast.
    return tf.where(
        (features["decision_phase"] == 4)[:, None],
        tf.one_hot(260, 295),
        probability,
    )
