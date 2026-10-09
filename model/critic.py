"""Action-conditioned payment critic, context, and supervision."""

import numpy
import tensorflow as tf
from log_dataset import payment_bonus
from model import (
    beliefs,
    decision_layers,
    immediate,
    joint_settlement,
    model,
    objectives,
    outcomes,
    payment_scoring,
)

STATISTICS_CHUNK = 512


def public_inputs(features, selected=None):
    """Exact visible scoring constraints after each proposed action.

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
        "public_fu": payment_scoring.public_fu(features),
    }
    if selected is None:
        return public
    public = {
        name: tf.gather(value, selected[:, 0]) for name, value in public.items()
    }
    hands = tf.stack(
        [
            selected[:, 0],
            tf.cast(
                tf.gather_nd(
                    decision_layers.candidate_hand_indices(features), selected
                ),
                tf.int32,
            ),
        ],
        axis=1,
    )
    melds = tf.cast(tf.gather_nd(features["candidate_melds"], hands), tf.int32)
    own_opened = tf.reduce_any(
        tf.gather(melds, [0, 1, 2, 4], axis=2) > 0, axis=(1, 2)
    )
    tiles = tf.range(34)
    own_fu = tf.reduce_sum(
        melds[:, :, :5]
        * tf.constant([0, 2, 8, 16, 8])[None, None]
        * (
            1
            + tf.cast(
                (tiles >= 27) | (tiles % 9 == 0) | (tiles % 9 == 8), tf.int32
            )
        )[None, :, None],
        axis=(1, 2),
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
        public["opened"], own, own_opened
    )
    public["public_fu"] = tf.tensor_scatter_nd_update(
        public["public_fu"], own, own_fu
    )
    public["deposit_allowed"] &= ~public["opened"]
    for name in ("deposit_required", "riichi_required"):
        public[name] = tf.tensor_scatter_nd_update(
            public[name], own, public[name][:, 0] | declaration
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
                tf.int32,
            ),
            axis=-1,
        )
    )
    # Accepted riichi cannot regain ippatsu after a successful call/kan.
    called = ((selected[:, 1] >= 112) & (selected[:, 1] < 260)) | (
        selected[:, 1] >= 261
    )
    public["immediate_deposit_required"] = (
        public["deposit_required"]
        & (tf.range(4)[None] != 0)
        & ((selected[:, 1] >= 112) & (selected[:, 1] < 260))[:, None]
    )
    public["immediate_ippatsu"] = (
        public["ippatsu_allowed"]
        & ~((selected[:, 1] >= 112) & (selected[:, 1] < 260))[:, None]
    )
    public["ippatsu_allowed"] &= ~called[:, None] | ~public["riichi_required"]
    # Only a completed discard (or pass on the last discard) can exhaust
    # the wall. Kan/rinshan and a kan-pass are not terminal actions.
    discard = discard_decisions(features, selected)
    action = selected[:, 1]
    tile = tf.where(
        action < 111,
        action % 37,
        tf.where(action < 260, (action - 112) % 37, 0),
    )
    abort = (
        tf.gather_nd(
            features["discard_abort"],
            tf.stack(
                [selected[:, 0], tf.cast(declaration, tf.int32), tile], axis=1
            ),
        )
        > 0
    )
    public["immediate_draw_kind"] = tf.where(
        discard & abort,
        2,
        tf.where(
            (discard | (action == 111))
            & (tf.gather(features["live_wall_count"], selected[:, 0]) == 0),
            1,
            0,
        ),
    )
    shanten = tf.concat(
        [
            features["discard_action_all_shanten"],
            features["discard_action_all_shanten"],
            features["discard_action_all_shanten"],
            tf.cast(tf.round(features["self_efficiency"][:, :1] * 8), tf.int8)
            + 1,
            tf.reshape(features["chi_discard_shanten"], (-1, 111)),
            tf.reshape(features["pon_discard_shanten"], (-1, 37)),
            tf.zeros((tf.shape(opened)[0], 35), tf.int8),
        ],
        axis=1,
    )
    public["immediate_tenpai"] = tf.concat(
        [
            tf.cast(tf.gather_nd(shanten, selected) == 1, tf.int32)[:, None],
            tf.where(public["riichi_required"][:, 1:], 1, -1),
        ],
        axis=1,
    )
    return public


def discard_decisions(inputs, selected):
    """Actions ending in a discard, including chi/pon then discard."""
    phase = tf.gather(inputs["decision_phase"], selected[:, 0])
    return (
        ((phase == 0) & (selected[:, 1] < 37))
        | ((phase == 3) & (selected[:, 1] >= 37) & (selected[:, 1] < 111))
        | ((phase == 1) & (selected[:, 1] >= 112) & (selected[:, 1] < 260))
    )


def immediate_decisions(inputs, selected):
    """Discard ron and robbery of an actor's concealed/added kan."""
    return discard_decisions(inputs, selected) | (
        (tf.gather(inputs["decision_phase"], selected[:, 0]) == 2)
        & (selected[:, 1] >= 261)
    )


def ron_risk_labels(labels, selected, structure):
    """Legal discard ron, including unknown amounts but excluding triple ron."""
    legal = tf.gather_nd(
        tf.transpose(labels["ron_legal"], [0, 2, 1]),
        tf.stack(
            [
                selected[:, 0],
                tf.minimum(tf.cast(structure[:, -1], tf.int32), 36),
            ],
            axis=1,
        ),
    )
    risk = legal == 1
    return risk & ~tf.reduce_all(risk, axis=1)[:, None]


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
                        decision_layers.candidate_hand_indices(inputs),
                        selected,
                    ),
                    tf.int32,
                ),
            ],
            axis=1,
        )
        melds = tf.cast(
            tf.gather_nd(inputs["candidate_melds"], hands), tf.float32
        )
        red = tf.reduce_sum(
            tf.cast(
                tf.gather_nd(inputs["candidate_hand_red"], hands), tf.float32
            ),
            axis=-1,
        ) + tf.reduce_sum(melds[:, :, 5], axis=-1)
        counts = (
            tf.cast(
                tf.gather_nd(inputs["candidate_hand_counts"], hands), tf.float32
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
                tf.math.asinh(tf.cast(structural[:, :34], tf.float32)),
            ],
            axis=-1,
        )


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
            len(payment_bonus.SCENARIOS), dtype="float32"
        )
        self.scenario_groups = tf.constant(payment_bonus.SCORE_GROUPS)
        self.scoring_scenarios = payment_bonus.SCORE_SCENARIOS
        self.scenario_embedding = self.add_weight(
            shape=(len(self.scoring_scenarios), 64),
            initializer="glorot_uniform",
            name="scenario_embedding",
        )
        self.hand_score = tf.keras.layers.Dense(
            len(payment_scoring.SCORES), dtype="float32"
        )
        self.immediate = immediate.ImmediateOutcome()

    def distributions(self, inputs):
        """Scored ron/tsumo events, exhaustive masks, abort and bank deposits.

        Twelve ron claimant slots plus four tsumo winners. All three tsumo
        payer shares use exactly the same score state and bonus scenario.
        """
        hidden = self.state(inputs["state"])
        cells = tf.reshape(self.cell_state(hidden), (-1, 16, 64))
        winning = tuple(range(12)) + tuple(
            next(
                i
                for i, cell in enumerate(payment_bonus.CELLS[:24])
                if cell[0] == 1 and cell[2] == winner
            )
            for winner in range(4)
        )
        shares = [
            cell if cell < 12 else 12 + row[2]
            for cell, row in enumerate(payment_bonus.CELLS[:24])
        ]
        bonus_logits = self.bonus(cells[:, : len(winning)])
        opened = tf.gather(
            inputs["opened"],
            [payment_bonus.CELLS[cell][2] for cell in winning],
            axis=1,
        )
        allowed = numpy.array(
            [
                [
                    not (
                        kind == 0
                        and special in (2, 4)
                        or (kind == 1 and special == 3)
                        or (special == 4 and mode != 1)
                    )
                    for mode, ura, ippatsu, special in payment_bonus.SCENARIOS
                ]
                for kind, payer, recipient in [
                    payment_bonus.CELLS[cell] for cell in winning
                ]
            ]
        )
        mode = tf.constant([s[0] for s in payment_bonus.SCENARIOS])
        allowed = tf.constant(allowed)[None] & (
            ~opened[..., None] | (mode == 0)[None, None]
        )
        required = tf.gather(
            inputs["riichi_required"],
            [payment_bonus.CELLS[cell][2] for cell in winning],
            axis=1,
        )
        ippatsu = tf.gather(
            inputs["ippatsu_allowed"],
            [payment_bonus.CELLS[cell][2] for cell in winning],
            axis=1,
        )
        allowed &= ~required[..., None] | (mode >= 2)[None, None]
        allowed &= (
            ippatsu[..., None]
            | (tf.constant([s[2] for s in payment_bonus.SCENARIOS]) == 0)[
                None, None
            ]
        )
        bonus_logits = tf.where(allowed, bonus_logits, -1e9)
        known_fu = tf.gather(
            inputs["public_fu"],
            [payment_bonus.CELLS[cell][2] for cell in winning],
            axis=1,
        )
        scenario_state = tf.nn.gelu(
            cells[:, : len(winning), None] + self.scenario_embedding[None, None]
        )
        score_logits = self.hand_score(
            tf.concat(
                [
                    scenario_state,
                    tf.broadcast_to(
                        tf.cast(known_fu, tf.float32)[:, :, None, None] / 128,
                        tf.concat([tf.shape(scenario_state)[:-1], [1]], axis=0),
                    ),
                ],
                axis=-1,
            )
        )
        # Decode a tsumo hand once; payer shares use the same logits.
        score_logits = tf.gather(score_logits, shares, axis=1)
        bonus_logits = tf.gather(bonus_logits, shares, axis=1)
        score_logits, scenario_values = payment_scoring.payment_probabilities(
            score_logits,
            inputs["dealer"],
            inputs["public_fu"],
            self.scoring_scenarios,
        )
        # Probability grouping must not use reduced-precision GPU matmuls.
        scenario_probability = tf.reduce_sum(
            tf.where(
                self.scenario_groups[:, None]
                == tf.range(len(self.scoring_scenarios))[None],
                tf.nn.softmax(bonus_logits)[..., None],
                0.0,
            ),
            axis=-2,
        )
        return {
            "group_score_logits": score_logits,
            "group_scenario_logits": tf.math.log(
                tf.maximum(scenario_values, 1e-30)
            ),
            "scenario_groups": self.scenario_groups,
            "score_probability": tf.reduce_sum(
                scenario_probability[..., None] * tf.nn.softmax(score_logits),
                axis=2,
            ),
            "bonus_logits": bonus_logits,
            "event_logits": self.event(hidden),
            "deposit_logits": tf.reshape(
                self.deposit(hidden), (-1, len(joint_settlement.EVENTS), 16)
            ),
            **self.immediate(inputs),
        }

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
        metrics = joint_settlement.observed_terms(
            labels, {**inputs, **outputs}, mean=objectives.masked_totals
        )
        metrics["score_nll"] = score_cross_entropy(
            labels["score"],
            labels["score_weight"],
            outputs["group_score_logits"],
            mean=objectives.masked_totals,
        )
        count = tf.cast(tf.shape(inputs["state"])[0], tf.float32)
        return tf.stack(
            [
                (
                    metrics[name]
                    if metrics[name].shape.rank == 1
                    else tf.stack([metrics[name] * count, count])
                )
                for name in CRITIC_METRICS
            ]
        )

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

    def supervised_statistics(self, inputs, labels):
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
            "score_logits": tf.gather(
                outputs["group_score_logits"], self.scenario_groups, axis=2
            ),
            "terminal_payment_probability": tf.einsum(
                "ncv,ctpr->ntprv",
                outputs["payment_probability"],
                tf.constant(
                    numpy.eye(64, dtype=numpy.float32)[
                        numpy.ravel_multi_index(
                            numpy.array(payment_bonus.CELLS).T, (4, 4, 4)
                        )
                    ].reshape(-1, 4, 4, 4)
                ),
            )
            + tf.cast(~tf.constant(outcomes.PAYMENT_CELLS), tf.float32)[
                None, ..., None
            ]
            * tf.one_hot(1, len(outcomes.PAYMENT_VALUES) + 4),
            "scenario_logits": tf.gather(
                outputs["group_scenario_logits"], self.scenario_groups, axis=2
            ),
        }


def score_cross_entropy(labels, weights, logits, mean=objectives.mean_valid):
    """Group counts preserve each observed cell's original scenario mean."""
    labels = tf.cast(labels, tf.int64)
    indices = tf.range(tf.size(labels, out_type=tf.int64)) * tf.shape(
        logits, out_type=tf.int64
    )[-1] + tf.reshape(tf.maximum(labels, 0), (-1,))
    nll = -tf.reshape(
        tf.gather(tf.reshape(tf.nn.log_softmax(logits), (-1,)), indices),
        tf.shape(labels),
    )
    weights = tf.cast(weights, tf.float32)
    per_cell = tf.math.divide_no_nan(
        tf.reduce_sum(nll * weights, -1), tf.reduce_sum(weights, -1)
    )
    return mean(per_cell, tf.reduce_any(weights > 0, -1))


CRITIC_METRICS = (
    "payment_nll",
    "score_nll",
    "settlement_multi_ron_fraction",
    "settlement_supported_fraction",
)


def objective_metrics(outputs, auxiliary, mean=objectives.mean_valid):
    metrics = {
        name: mean(
            tf.math.divide_no_nan(
                outputs["critic_statistics"][index, 0],
                outputs["critic_statistics"][index, 1],
            ),
            outputs["critic_statistics"][index, 1] > 0,
            outputs["critic_statistics"][index, 1],
        )
        for index, name in enumerate(CRITIC_METRICS)
    }
    for name, terms in (
        ("risk_loss", "risk_terms"),
        ("immediate_amount_nll", "immediate_amount_terms"),
        ("immediate_draw_nll", "immediate_draw_terms"),
    ):
        metrics[name] = objectives.candidate_mean(
            *outputs[terms], outputs["immediate_rows"], mean=mean
        )
    loss = (
        metrics["payment_nll"]
        + metrics["risk_loss"]
        + metrics["immediate_amount_nll"]
        + metrics["immediate_draw_nll"]
    )
    return loss, metrics


def chosen_actions(features, labels):
    return tf.where(
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
                    111 + tf.cast(labels["response_policy_logits"], tf.int32),
                ),
                260 + tf.cast(labels["kan_action_logits"], tf.int32),
            ),
        ),
    )


class JointPayment(tf.keras.Model):

    def __init__(self, project_counts=True):
        super().__init__(autocast=False)
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        inputs, board, candidates, selected, structure = model.encode_inputs(
            True, 2
        )
        self.beliefs = beliefs.OpponentBeliefs(
            project_counts=project_counts,
            encoder=tf.keras.Model(
                inputs, board, name="belief_encoder", autocast=False
            ),
        )
        belief_logits = self.beliefs(inputs, hidden=board)
        belief_probability = beliefs.Probabilities(dtype="float32")(
            belief_logits
        )
        self.encoder = tf.keras.Model(
            inputs,
            dict(
                candidate=candidates,
                board=board,
                selected=selected,
                structure=structure,
                **belief_logits,
                belief_context=tf.keras.layers.Concatenate(dtype="float32")(
                    [
                        tf.keras.layers.Flatten(dtype="float32")(
                            belief_probability[name]
                        )
                        for name in beliefs.HEADS
                    ]
                ),
                belief_wait_probability=tf.keras.layers.Multiply(
                    dtype="float32"
                )(
                    [
                        belief_probability[beliefs.HEADS[0]][..., 0, None],
                        belief_probability[beliefs.HEADS[1]][..., 0, :],
                    ]
                ),
            ),
        )
        tf.keras.mixed_precision.set_global_policy("float32")
        self.context = HandContext(dtype="float32")
        self.public = PublicContext(dtype="float32")
        self.critic = PaymentCritic()

    def compile_training(self):
        """Call after initialization so tracing binds live Keras variables."""
        self.critic.distributions = tf.function(
            self.critic.distributions, jit_compile=True
        )
        self.critic.joint_distributions = tf.function(
            self.critic.joint_distributions, jit_compile=True
        )
        self.critic.chunk_statistics = tf.function(
            self.critic.chunk_statistics, jit_compile=True
        )
        self.critic.chunk_gradients = tf.function(
            self.critic.chunk_gradients, jit_compile=True
        )
        self.critic.supervised_statistics = tf.function(
            self.critic.supervised_statistics
        )
        decision_layers.numeric_features = tf.function(
            decision_layers.numeric_features, jit_compile=True
        )
        for block in self.encoder.get_layer("hand_bank").suits:
            block.call = tf.function(block.call, jit_compile=True)
        river = self.encoder.get_layer("river_encoder")
        river.encode = tf.function(river.encode, jit_compile=True)

    def candidate_context(self, features, encoded):
        state = tf.concat(
            [
                encoded["candidate"],
                tf.gather(encoded["belief_context"], encoded["selected"][:, 0]),
                tf.gather(encoded["board"], encoded["selected"][:, 0]),
                self.context(
                    [features, encoded["selected"], encoded["structure"]]
                ),
                self.public(
                    [features, encoded["selected"], encoded["structure"]]
                ),
            ],
            axis=-1,
        )
        tiles = tf.cast(encoded["structure"][:, -1], tf.int32)
        base = tf.where(
            tiles == 37, 0, tf.where(tiles < 34, tiles, 4 + 9 * (tiles - 34))
        )
        encoded["immediate_safe"] = (
            tf.gather_nd(
                tf.transpose(features["genbutsu_to_seat"][:, 1:], [0, 2, 1]),
                tf.stack([encoded["selected"][:, 0], base], axis=1),
            )
            > 0
        )
        encoded["immediate_wait"] = tf.gather_nd(
            tf.transpose(encoded["belief_wait_probability"], [0, 2, 1]),
            tf.stack([encoded["selected"][:, 0], base], axis=1),
        )
        return encoded, state

    def call(self, inputs):
        encoded, state = self.candidate_context(
            inputs[0], self.encoder(inputs[0])
        )
        return self.supervised_outputs(inputs, encoded, state)

    def supervised_outputs(self, inputs, encoded, state):
        features, labels, _ = inputs
        chosen = chosen_actions(features, labels)
        recorded = (
            tf.gather(chosen, encoded["selected"][:, 0])
            == encoded["selected"][:, 1]
        )
        tf.debugging.assert_equal(
            tf.boolean_mask(encoded["selected"][:, 0], recorded),
            tf.range(tf.shape(chosen)[0]),
        )
        allowed = immediate_decisions(features, encoded["selected"])
        selected = tf.boolean_mask(encoded["selected"], recorded)
        recorded_inputs = {
            **joint_settlement.public_inputs(features),
            "state": tf.boolean_mask(state, recorded),
            "immediate_safe": tf.boolean_mask(
                encoded["immediate_safe"], recorded
            ),
            **public_inputs(features, selected),
            "immediate_allowed": tf.boolean_mask(allowed, recorded),
            "immediate_wait": tf.boolean_mask(
                encoded["immediate_wait"], recorded
            ),
        }
        if not self.critic.built:
            self.critic(
                {name: value[:1] for name, value in recorded_inputs.items()}
            )
        outputs = {
            "critic_statistics": self.critic.supervised_statistics(
                recorded_inputs, inputs[2]
            )
        }
        candidate_public = {
            **public_inputs(features, encoded["selected"]),
            "immediate_allowed": allowed,
            "honba": tf.cast(
                tf.gather(features["honba"], encoded["selected"][:, 0]),
                tf.int32,
            ),
        }
        outputs["immediate_rows"] = encoded["selected"][:, 0]
        immediate_outputs = self.critic.immediate(
            {
                **candidate_public,
                "state": state,
                "immediate_safe": encoded["immediate_safe"],
                "immediate_wait": encoded["immediate_wait"],
            }
        )
        outputs.update(
            immediate.loss_terms(
                labels=labels,
                selected=encoded["selected"],
                structure=encoded["structure"],
                public=candidate_public,
                outputs=immediate_outputs,
            )
        )
        return outputs


class JointPolicyPayment(JointPayment):
    """Specialist/final-pass policy shares the trainable critic encoder."""

    def __init__(self, project_counts=True):
        super().__init__(project_counts=project_counts)
        inputs = self.encoder.input
        encoded = self.encoder.output
        probability = beliefs.Probabilities(dtype="float32")(encoded)
        outputs, context = model.build_board_heads(
            inputs=inputs,
            hidden=encoded["board"],
            heads=(
                "opponent_shanten_logits",
                "opponent_ukeire_by_shanten_logits",
                "opponent_hand_count_logits",
                "round_completion_yaku_logits",
                "round_completion_score_logits",
            ),
            predictions={
                name: (encoded[name], probability[name])
                for name in beliefs.HEADS
            },
        )
        outputs.update(
            model.build_policy_heads(
                inputs=inputs,
                hidden=encoded["board"],
                candidates=encoded["candidate"],
                selected=encoded["selected"],
                context=context,
            )
        )
        outputs.update(
            model.PolicySummaries(dtype="float32")([outputs, inputs])
        )
        # Specialists have visible-route and hindsight auxiliary targets.
        # Only predictions enter the actor; labels stay in the trainer.
        self.actor = tf.keras.Model(inputs, {**outputs, **encoded})
        # Rollout encoding and serving use the same yaku-conditioned critic.
        self.encoder = tf.keras.Model(
            inputs,
            {
                **encoded,
                "attack_yaku_logits": outputs["attack_yaku_logits"],
                "round_completion_yaku_logits": outputs[
                    "round_completion_yaku_logits"
                ],
                "round_completion_score_logits": outputs[
                    "round_completion_score_logits"
                ],
            },
        )

    def candidate_context(self, features, encoded):
        encoded, state = super().candidate_context(features, encoded)
        return encoded, tf.concat(
            [
                state,
                tf.sigmoid(encoded["attack_yaku_logits"]),
                tf.reshape(
                    tf.gather(
                        tf.sigmoid(encoded["round_completion_yaku_logits"]),
                        encoded["selected"][:, 0],
                    ),
                    (-1, 4 * 2 * 23),
                ),
                tf.reshape(
                    tf.gather(
                        tf.nn.softmax(encoded["round_completion_score_logits"]),
                        encoded["selected"][:, 0],
                    ),
                    (-1, 4 * 2 * len(payment_scoring.SCORES)),
                ),
            ],
            axis=-1,
        )

    def call(self, inputs):
        outputs = self.actor(inputs[0])
        # Reuse the actor encoder output for critic supervision.
        encoded, state = self.candidate_context(inputs[0], outputs)
        return {
            **outputs,
            **self.supervised_outputs(inputs, encoded, state),
            "critic_candidate_state": state,
            "immediate_safe": encoded["immediate_safe"],
            "immediate_wait": encoded["immediate_wait"],
        }
