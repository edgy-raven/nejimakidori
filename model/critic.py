"""Action-conditioned payment critic, context, and supervision."""

import numpy
import tensorflow as tf

from . import (
    decision_layers,
    model,
    objectives,
    outcomes,
    payment_bonus,
    payment_scoring,
)


def public_inputs(features):
    """Scoring constraints available from visible features at inference."""
    opened = tf.reduce_any(
        (features["meld_valid"] > 0) & (features["meld_type"] != 4), axis=-1
    )
    return {
        "opened": opened,
        "deposit_allowed": ~opened & (features["riichi_state"] != 2),
        "dealer": features["dealer_relative_seat"],
        "public_fu": payment_scoring.public_fu(features),
    }


def discard_decisions(inputs, selected):
    phase = tf.gather(inputs["decision_phase"], selected[:, 0])
    return ((phase == 0) & (selected[:, 1] < 37)) | (
        (phase == 3) & (selected[:, 1] >= 37) & (selected[:, 1] < 111)
    )


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


DEAL_IN_CELLS = [
    index
    for index, (kind, payer, recipient) in enumerate(payment_bonus.CELLS)
    if kind == 0 and payer == 0
]

ACTOR_CELLS = tuple(
    index
    for index, (kind, payer, recipient) in enumerate(payment_bonus.CELLS)
    if int(recipient == 0) - int(payer == 0) + int(kind == 3 and payer == 0)
)


class PaymentCritic(tf.keras.Model):
    """Sparse action states in; payment distributions and derived values out.

    The encoder must end before the actor's old payment head. There is no
    learned scalar value or state-payment predictor in this model.
    """

    def __init__(self):
        super().__init__(autocast=False)
        self.state = tf.keras.layers.Dense(128, activation="gelu")
        self.cell_state = tf.keras.layers.Dense(
            len(payment_bonus.CELLS) * 64, activation="gelu"
        )
        self.occurrence = tf.keras.layers.Dense(1, dtype="float32")
        self.bonus = tf.keras.layers.Dense(
            len(payment_bonus.SCENARIOS), dtype="float32"
        )
        # Equal added-han scenarios share a decoder, not event probabilities.
        factors = [
            (
                mode == 0,
                (2 if mode == 3 else int(mode == 2))
                + ura
                + ippatsu
                + int(special in (1, 2, 3)),
                special == 4,
                ura == 5,
            )
            for mode, ura, ippatsu, special in payment_bonus.SCENARIOS
        ]
        groups = sorted(set(factors))
        self.scenario_groups = tf.constant([groups.index(f) for f in factors])
        self.scoring_scenarios = tuple(
            payment_bonus.SCENARIOS[factors.index(group)] for group in groups
        )
        self.scenario_embedding = self.add_weight(
            shape=(len(groups) + 1, 64),
            initializer="glorot_uniform",
            name="scenario_embedding",
        )
        self.amount = tf.keras.layers.Dense(
            len(outcomes.PAYMENT_VALUES), dtype="float32"
        )
        self.hand_score = tf.keras.layers.Dense(
            len(payment_scoring.SCORES), dtype="float32"
        )
        self.immediate_risk = tf.keras.layers.Dense(
            1,
            dtype="float32",
            bias_initializer=tf.keras.initializers.Constant(-4),
        )

    def encode(self, state, cell_ids=None):
        if cell_ids is None:
            return tf.reshape(
                self.cell_state(self.state(state)),
                (-1, len(payment_bonus.CELLS), 64),
            )
        # Alternative-action risk only needs the three outgoing ron cells.
        # Slice the shared projection instead of evaluating all 40 cells.
        kernel = tf.reshape(
            self.cell_state.kernel, (128, len(payment_bonus.CELLS), 64)
        )
        bias = tf.reshape(self.cell_state.bias, (len(payment_bonus.CELLS), 64))
        return self.cell_state.activation(
            tf.reshape(
                tf.linalg.matmul(
                    self.state(state),
                    tf.reshape(tf.gather(kernel, cell_ids, axis=1), (128, -1)),
                ),
                (-1, len(cell_ids), 64),
            )
            + tf.gather(bias, cell_ids)
        )

    def immediate_logits(self, cells, safe):
        logits = self.immediate_risk(cells)[..., 0]
        return tf.where(safe, -1e9, logits)

    def distributions(self, inputs, actor_moments=False):
        """Forecast all payment classes or actor-cell first/second moments."""
        cell_ids = (
            ACTOR_CELLS
            if actor_moments
            else tuple(range(len(payment_bonus.CELLS)))
        )
        winning = tuple(cell for cell in cell_ids if cell < 24)
        cells = self.encode(
            inputs["state"], cell_ids if actor_moments else None
        )
        occurrence = self.occurrence(cells)[..., 0]
        immediate_logits = self.immediate_logits(
            tf.gather(
                cells, [cell_ids.index(cell) for cell in DEAL_IN_CELLS], axis=1
            ),
            inputs["immediate_safe"],
        )
        immediate = tf.einsum(
            "nr,rc->nc",
            tf.sigmoid(immediate_logits)
            * tf.cast(inputs["immediate_allowed"], tf.float32)[:, None],
            tf.one_hot(
                [cell_ids.index(cell) for cell in DEAL_IN_CELLS], len(cell_ids)
            ),
        )
        # Approximate no immediate ron across opponents. Bank deposits may
        # already be paid and do not share future-transfer survival.
        continuation = 1 - immediate
        survival = tf.reduce_prod(1 - immediate, axis=1, keepdims=True)
        continuation = tf.where(
            tf.constant(
                [payment_bonus.CELLS[cell][0] != 3 for cell in cell_ids]
            ),
            survival,
            continuation,
        )
        event_probability = immediate + continuation * tf.sigmoid(occurrence)
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
        bonus_logits = tf.where(allowed, bonus_logits, -1e9)
        known_fu = tf.gather(
            inputs["public_fu"],
            [payment_bonus.CELLS[cell][2] for cell in winning],
            axis=1,
        )
        scenario_state = tf.nn.gelu(
            cells[:, : len(winning), None]
            + self.scenario_embedding[None, None, :-1]
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
        score_logits, scenario_values = (
            payment_scoring.payment_moments(
                logits=score_logits,
                dealer=inputs["dealer"],
                meld_fu=inputs["public_fu"],
                scenarios=self.scoring_scenarios,
                cell_ids=winning,
            )
            if actor_moments
            else payment_scoring.payment_probabilities(
                score_logits,
                inputs["dealer"],
                inputs["public_fu"],
                self.scoring_scenarios,
            )
        )
        win_amount = tf.reduce_sum(
            tf.linalg.matmul(
                tf.nn.softmax(bonus_logits),
                tf.one_hot(self.scenario_groups, len(self.scoring_scenarios)),
            )[..., None]
            * scenario_values,
            axis=2,
        )
        other_logits = self.amount(
            tf.nn.gelu(
                cells[:, len(winning) :] + self.scenario_embedding[None, -1]
            )
        )
        other_allowed = numpy.array(
            [
                (outcomes.PAYMENT_VALUES != 0)
                & (
                    outcomes.PAYMENT_VALUES % 1000 == 0
                    if kind == 3
                    else outcomes.PAYMENT_VALUES > 0
                )
                for kind, payer, recipient in [
                    payment_bonus.CELLS[cell]
                    for cell in cell_ids[len(winning) :]
                ]
            ]
        )
        other_allowed = tf.constant(other_allowed)[None] & ~(
            tf.constant(
                [
                    payment_bonus.CELLS[cell][0] == 3
                    for cell in cell_ids[len(winning) :]
                ]
            )[None, :, None]
            & ~tf.gather(
                inputs["deposit_allowed"],
                [
                    payment_bonus.CELLS[cell][2]
                    for cell in cell_ids[len(winning) :]
                ],
                axis=1,
            )[..., None]
            & (tf.constant(outcomes.PAYMENT_VALUES) < 0)
        )
        other_amount = tf.nn.softmax(
            tf.where(other_allowed, other_logits, -1e9)
        )
        if actor_moments:
            values = tf.constant(outcomes.PAYMENT_VALUES, tf.float32)
            return {
                "actor_payment_moments": event_probability[..., None]
                * tf.concat(
                    [
                        win_amount,
                        tf.stack(
                            [
                                tf.reduce_sum(other_amount * values, axis=-1),
                                tf.reduce_sum(
                                    other_amount * tf.square(values), axis=-1
                                ),
                            ],
                            axis=-1,
                        ),
                    ],
                    axis=1,
                )
            }
        probability = event_probability[..., None] * tf.concat(
            [win_amount, other_amount], axis=1
        ) + (1 - event_probability)[..., None] * tf.one_hot(
            1, len(outcomes.PAYMENT_VALUES)
        )
        expected = tf.reduce_sum(
            probability * tf.constant(outcomes.PAYMENT_VALUES, tf.float32),
            axis=-1,
        )
        coefficients = numpy.array(
            [
                [
                    int(recipient == seat)
                    - int(payer == seat)
                    + int(kind == 3 and payer == seat)
                    for seat in range(4)
                ]
                for kind, payer, recipient in payment_bonus.CELLS
            ],
            numpy.float32,
        )
        return {
            "group_score_logits": score_logits,
            "group_scenario_logits": tf.math.log(
                tf.maximum(scenario_values, 1e-09)
            ),
            "scenario_groups": self.scenario_groups,
            "payment_probability": probability,
            "expected_points": tf.linalg.matmul(expected, coefficients) / 10000,
            "bonus_logits": bonus_logits,
            "immediate_logits": immediate_logits,
        }

    def call(self, inputs):
        outputs = self.distributions(inputs)
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
            * tf.one_hot(1, len(outcomes.PAYMENT_VALUES)),
            "scenario_logits": tf.gather(
                outputs["group_scenario_logits"], self.scenario_groups, axis=2
            ),
        }


def scenario_cross_entropy(labels, logits, groups, mean=objectives.mean_valid):
    # Unobserved cells have no loss or gradient. Avoid normalizing their
    # scoring distributions a second time solely to discard the result.
    selected = tf.where(tf.reduce_any(labels >= 0, axis=-1))
    labels = tf.gather_nd(labels, selected)[:, None]
    logits = tf.gather_nd(logits, selected)[:, None]
    valid = tf.cast(labels >= 0, tf.float32)
    shape = tf.shape(logits)
    indices = (
        tf.range(shape[0] * shape[1])[:, None] * shape[2] * shape[3]
        + groups[None] * shape[3]
        + tf.reshape(tf.maximum(labels, 0), (-1, tf.shape(labels)[-1]))
    )
    nll = -tf.reshape(
        tf.gather(tf.reshape(tf.nn.log_softmax(logits), (-1,)), indices),
        tf.shape(labels),
    )
    per_cell = tf.math.divide_no_nan(
        tf.reduce_sum(nll * valid, axis=-1), tf.reduce_sum(valid, axis=-1)
    )
    return mean(per_cell, tf.reduce_any(labels >= 0, -1))


def loss_and_metrics(labels, outputs, mean=objectives.mean_valid):
    selected = tf.gather(
        outputs["payment_probability"], labels["payment"], axis=2, batch_dims=2
    )
    payment_loss = -tf.reduce_mean(tf.math.log(tf.maximum(selected, 1e-09)))
    bonus_valid = labels["bonus"] >= 0
    bonus_loss = mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.maximum(labels["bonus"], 0),
            logits=outputs["bonus_logits"],
        ),
        bonus_valid,
    )
    scenario_valid = labels["scenario"] >= 0
    # Each observed winning cell has equal weight across legal scenarios.
    scenario_loss = scenario_cross_entropy(
        labels["scenario"],
        outputs["group_scenario_logits"],
        outputs["scenario_groups"],
        mean=mean,
    )
    scenario_value = tf.reduce_sum(
        tf.nn.softmax(outputs["group_scenario_logits"])
        * tf.constant(outcomes.PAYMENT_VALUES / 10000, tf.float32),
        axis=-1,
    )
    scenario_value = tf.gather(
        scenario_value, outputs["scenario_groups"], axis=2
    )
    scenario_error = tf.math.divide_no_nan(
        tf.reduce_sum(
            tf.square(
                scenario_value
                - tf.gather(
                    tf.constant(outcomes.PAYMENT_VALUES / 10000, tf.float32),
                    tf.maximum(labels["scenario"], 0),
                )
            )
            * tf.cast(scenario_valid, tf.float32),
            axis=-1,
        ),
        tf.reduce_sum(tf.cast(scenario_valid, tf.float32), axis=-1),
    )
    scenario_error = tf.math.divide_no_nan(
        tf.reduce_sum(scenario_error),
        tf.reduce_sum(tf.cast(tf.reduce_any(scenario_valid, -1), tf.float32)),
    )
    point_error = tf.reduce_mean(
        tf.square(outputs["expected_points"][:, 0] - labels["points"])
    )
    metrics = {
        "payment_nll": payment_loss,
        "bonus_nll": bonus_loss,
        "scenario_nll": scenario_loss,
        "scenario_mse": scenario_error,
        "point_mse": point_error,
        "point_log_mse": tf.reduce_mean(
            tf.square(
                objectives.signed_log(outputs["expected_points"][:, 0])
                - objectives.signed_log(labels["points"])
            )
        ),
    }
    metrics["score_nll"] = scenario_cross_entropy(
        labels["score"],
        outputs["group_score_logits"],
        outputs["scenario_groups"],
        mean=mean,
    )
    return (payment_loss + bonus_loss + scenario_loss, metrics)


def distribution_scores(probability, classes):
    """Proper amount-distance and severe-outgoing-payment forecast scores."""
    order = numpy.argsort(outcomes.PAYMENT_VALUES)
    values = tf.constant(outcomes.PAYMENT_VALUES / 10000, tf.float32)
    target = tf.gather(values, classes)
    cdf = tf.cumsum(tf.gather(probability, order, axis=-1), axis=-1)[..., :-1]
    payment_crps = tf.reduce_mean(
        tf.reduce_sum(
            tf.square(
                cdf
                - tf.cast(
                    target[..., None] <= tf.gather(values, order[:-1]),
                    tf.float32,
                )
            )
            * tf.constant(
                numpy.diff(outcomes.PAYMENT_VALUES[order]) / 10000, tf.float32
            ),
            axis=-1,
        )
    )
    outgoing = [
        cell
        for cell, (kind, payer, _) in enumerate(payment_bonus.CELLS)
        if kind < 2 and payer == 0
    ]
    tails = tf.reduce_sum(
        tf.gather(probability, outgoing, axis=1)[..., None]
        * tf.cast(values[:, None] >= tf.constant([0.8, 1.2, 2.4]), tf.float32),
        axis=-2,
    )
    occurred = tf.gather(target, outgoing, axis=1)[..., None] >= tf.constant(
        [0.8, 1.2, 2.4]
    )
    return {
        "payment_crps": payment_crps,
        "loss_tail_nll": -tf.reduce_mean(
            tf.math.log(tf.maximum(tf.where(occurred, tails, 1 - tails), 1e-09))
        ),
    }


def objective_metrics(outputs, auxiliary, mean=objectives.mean_valid):
    metrics = loss_and_metrics(auxiliary, outputs, mean=mean)[1]
    metrics.update(
        distribution_scores(
            outputs["payment_probability"], auxiliary["payment"]
        )
    )
    metrics["risk_loss"] = mean(*outputs["risk_terms"])
    metrics["ranking_loss"] = mean(*outputs["ranking_terms"])
    loss = (
        metrics["payment_nll"]
        + metrics["point_log_mse"]
        + metrics["risk_loss"]
        + 0.1
        * (
            metrics["bonus_nll"]
            + metrics["scenario_nll"]
            + metrics["score_nll"]
            + metrics["ranking_loss"]
        )
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


def discard_ranking_terms(logits, risk, selected, tiles, allowed, count):
    """Conditional tile choice loss on mixed safe/dangerous groups only."""
    keys = selected[:, 0] * 37 + tf.minimum(tiles, 36)
    first = tf.math.unsorted_segment_min(
        tf.range(tf.shape(keys)[0]), keys, count * 37
    )
    keep = allowed & (tf.range(tf.shape(keys)[0]) == tf.gather(first, keys))
    indices = tf.boolean_mask(selected[:, 0], keep)
    logits = tf.boolean_mask(logits, keep)
    risk = tf.cast(tf.boolean_mask(risk, keep), tf.float32)
    maximum = tf.math.unsorted_segment_max(logits, indices, count)
    normalizer = tf.math.unsorted_segment_sum(
        tf.exp(logits - tf.gather(maximum, indices)), indices, count
    )
    log_probability = logits - tf.gather(
        maximum + tf.math.log(tf.maximum(normalizer, 1e-30)), indices
    )
    positives = tf.math.unsorted_segment_sum(risk, indices, count)
    total = tf.math.unsorted_segment_sum(tf.ones_like(risk), indices, count)
    mixed = tf.cast((positives > 0) & (positives < total), tf.float32)
    cross_entropy = tf.math.divide_no_nan(
        tf.math.unsorted_segment_sum(-risk * log_probability, indices, count),
        positives,
    )
    return cross_entropy, mixed


class JointPayment(tf.keras.Model):

    def __init__(self):
        super().__init__(autocast=False)
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        inputs, board, candidates, selected, structure = model.encode_inputs(
            True, 2
        )
        self.encoder = tf.keras.Model(
            inputs,
            dict(
                candidate=candidates,
                board=board,
                selected=selected,
                structure=structure,
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
        return encoded, state

    def call(self, inputs):
        encoded, state = self.candidate_context(
            inputs[0], self.encoder(inputs[0])
        )
        return self.supervised_outputs(inputs, encoded, state)

    def supervised_outputs(self, inputs, encoded, state):
        features, labels, auxiliary = inputs
        chosen = chosen_actions(features, labels)
        recorded = (
            tf.gather(chosen, encoded["selected"][:, 0])
            == encoded["selected"][:, 1]
        )
        tf.debugging.assert_equal(
            tf.boolean_mask(encoded["selected"][:, 0], recorded),
            tf.range(tf.shape(chosen)[0]),
        )
        allowed = discard_decisions(features, encoded["selected"])
        rows = tf.boolean_mask(encoded["selected"][:, 0], recorded)
        outputs = self.critic(
            {
                "state": tf.boolean_mask(state, recorded),
                "immediate_safe": tf.boolean_mask(
                    encoded["immediate_safe"], recorded
                ),
                **{
                    name: tf.gather(auxiliary[name], rows)
                    for name in (
                        "opened",
                        "deposit_allowed",
                        "dealer",
                        "public_fu",
                    )
                },
                "immediate_allowed": tf.boolean_mask(allowed, recorded),
            }
        )
        payments = tf.gather_nd(
            tf.transpose(labels["ron_payments"], [0, 2, 1]),
            tf.stack(
                [
                    encoded["selected"][:, 0],
                    tf.minimum(
                        tf.cast(encoded["structure"][:, -1], tf.int32), 36
                    ),
                ],
                axis=1,
            ),
        )
        risk = payments != 0
        risk &= ~tf.reduce_all(risk, axis=1)[:, None]
        risk_logits = self.critic.immediate_logits(
            self.critic.encode(state, DEAL_IN_CELLS),
            encoded["immediate_safe"],
        )
        outputs["risk_terms"] = (
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(risk, tf.float32), logits=risk_logits
            ),
            tf.broadcast_to(allowed[:, None], tf.shape(risk_logits)),
        )
        outputs["ranking_terms"] = discard_ranking_terms(
            logits=risk_logits,
            risk=risk,
            selected=encoded["selected"],
            tiles=tf.cast(encoded["structure"][:, -1], tf.int32),
            allowed=allowed,
            count=tf.shape(chosen)[0],
        )
        return outputs


class JointPolicyPayment(JointPayment):
    """The previous specialist/final-pass policy shares the critic encoder."""

    def __init__(self):
        super().__init__()
        inputs = self.encoder.input
        encoded = self.encoder.output
        outputs, context = model.build_board_heads(
            inputs,
            encoded["board"],
            (
                "opponent_shanten_logits",
                "opponent_ukeire_by_shanten_logits",
                "opponent_hand_count_logits",
                "round_completion_yaku_logits",
                "round_placement_logits",
                "final_placement_logits",
            ),
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
        actor = tf.keras.Model(inputs, outputs)
        self.actor = tf.keras.Model(
            inputs,
            {
                **outputs,
                **encoded,
                **{
                    name: decision_layers.UnpackCandidates()(
                        [
                            actor.get_layer(name).output,
                            encoded["selected"],
                            encoded["board"],
                        ]
                    )
                    for name in (
                        "attack_predictions",
                        "defense_predictions",
                        "attack_decisions",
                        "defense_decisions",
                    )
                },
            },
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
        }
