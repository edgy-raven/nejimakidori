"""Trainable actor consuming the complete declared visible feature set."""

import math

import tensorflow as tf

import feature_vector

from . import decision_layers, outcomes


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class NumericFeatures(tf.keras.layers.Layer):
    def call(self, values):
        # Monotone compression retains information from numeric inputs while
        # keeping point magnitudes safe in mixed precision.
        return tf.math.asinh(tf.cast(values, tf.float32))


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class PolicyMask(tf.keras.layers.Layer):
    def __init__(self, head, **kwargs):
        super().__init__(**kwargs)
        self.head = head

    def get_config(self):
        return {**super().get_config(), "head": self.head}

    def call(self, values):
        logits, inputs = values
        mask = decision_layers.legal_choices(inputs)[self.head]
        return tf.where(
            tf.reduce_any(mask, -1, keepdims=True),
            tf.where(mask, logits, -1e4),
            tf.zeros_like(logits),
        )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class HeldCountMask(tf.keras.layers.Layer):
    def call(self, values):
        logits, inputs = values
        tile = tf.cast(inputs["meld_tile_ids"][:, 1:], tf.int32)
        red = tf.cast(inputs["meld_tile_red"][:, 1:], tf.bool)
        code = tf.where(red, 34 + tile // 9, tile)
        valid = tf.cast(inputs["meld_tile_valid"][:, 1:], tf.float32)
        lower = tf.reduce_sum(
            tf.one_hot(code, 37) * valid[..., None], axis=(2, 3)
        )
        unseen = 4 - tf.cast(inputs["known_unavailable_counts"], tf.float32)
        unseen_red = (
            1
            - tf.cast(inputs["candidate_hand_red"][:, 0], tf.float32)
            - tf.cast(inputs["public_visible_red_counts"], tf.float32)
        )
        correction = tf.linalg.matmul(unseen_red, tf.one_hot([4, 13, 22], 34))
        available = tf.concat([unseen - correction, unseen_red], -1)
        upper = tf.minimum(4.0, lower + tf.maximum(0.0, available[:, None, :]))
        classes = tf.cast(tf.range(5), tf.float32)
        valid = (classes >= lower[..., None]) & (classes <= upper[..., None])
        return tf.where(valid, logits, -1e4)


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class PaymentMask(tf.keras.layers.Layer):
    def legal_payments(self, inputs):
        points = tf.constant(outcomes.PAYMENT_VALUES)
        bank = (tf.range(len(outcomes.PAYMENT_TYPES)) == 3)[:, None, None, None]
        legal = (points == 0) | (
            tf.constant(outcomes.PAYMENT_CELLS)[..., None]
            & tf.where(bank, points % 1000 == 0, points > 0)
        )
        # Concealed kan preserves a closed hand. Open hands can receive
        # sticks, but cannot pay a deposit; already-paid deposits are past.
        opened = tf.reduce_any(
            (inputs["meld_valid"] > 0) & (inputs["meld_type"] != 4), axis=-1
        )
        return legal[None] & ~(
            bank[None]
            & (opened | (inputs["riichi_state"] == 2))[:, None, :, None, None]
            & (points < 0)
        )

    def call(self, values):
        logits, inputs = values
        return tf.where(self.legal_payments(inputs), logits, -1e9)


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class PaymentDistribution(PaymentMask):
    def call(self, values):
        # Every type/cell has its own occurrence gate and conditional amount.
        legal_nonzero = self.legal_payments(values[1]) & (
            tf.constant(outcomes.PAYMENT_VALUES) != 0
        )
        conditional = tf.nn.log_softmax(
            tf.where(legal_nonzero, values[0], -1e9), axis=-1
        )
        occurrence = values[0][..., 1, None]
        return tf.where(
            tf.constant(outcomes.PAYMENT_VALUES) == 0,
            tf.where(
                tf.reduce_any(legal_nonzero, axis=-1, keepdims=True),
                -tf.nn.softplus(occurrence),
                0.0,
            ),
            tf.where(
                legal_nonzero,
                conditional - tf.nn.softplus(-occurrence),
                -1e9,
            ),
        )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class PolicySummaries(tf.keras.layers.Layer):
    def call(self, values):
        outputs, inputs = values
        riichi = tf.reshape(outputs["riichi_policy_logits"], (-1, 2, 38))
        response = outputs["response_policy_logits"]
        chi = tf.reshape(response[:, 1:112], (-1, 3, 37))
        pon = tf.reshape(response[:, 112:149], (-1, 1, 37))
        draw = tf.where(
            tf.cast(inputs["current_draw_is_red"], tf.bool),
            34 + tf.cast(inputs["current_draw_tile_id"], tf.int32) // 9,
            tf.cast(inputs["current_draw_tile_id"], tf.int32),
        )
        # Discard and riichi decisions have independently trained policies.
        # Use the active policy for the no-declaration branch.
        without = tf.where(
            (inputs["decision_phase"] == 3)[:, None],
            riichi[:, 0],
            outputs["discard_policy_logits"],
        )
        branches = tf.stack([without, riichi[:, 1]], axis=1)
        origins = tf.stack(
            [
                tf.gather(branches, draw, axis=2, batch_dims=1),
                branches[:, :, 37],
            ],
            axis=-1,
        )
        origins = tf.where(
            tf.cast(inputs["current_draw_valid"], tf.bool)[:, None, None],
            origins,
            tf.zeros_like(origins),
        )
        ready = (
            tf.concat(
                [
                    inputs["discard_action_all_shanten"],
                    tf.gather(
                        inputs["discard_action_all_shanten"], draw, batch_dims=1
                    )[:, None],
                ],
                axis=-1,
            )
            == 1
        )
        decision = tf.stack(
            [
                tf.reduce_logsumexp(tf.where(ready, without, -1e9), -1),
                tf.reduce_logsumexp(tf.where(~ready, without, -1e9), -1),
                tf.where(
                    inputs["decision_phase"] == 3,
                    tf.reduce_logsumexp(riichi[:, 1], -1),
                    -1e9,
                ),
            ],
            axis=-1,
        )
        shanten_log = tf.nn.log_softmax(outputs["opponent_shanten_logits"], -1)
        improving_log = -tf.nn.softplus(
            -outputs["opponent_ukeire_by_shanten_logits"]
        )
        other_log = -tf.nn.softplus(
            outputs["opponent_ukeire_by_shanten_logits"]
        )
        return {
            "opponent_ukeire_logits": (
                tf.reduce_logsumexp(shanten_log[..., None] + improving_log, -2)
                - tf.reduce_logsumexp(shanten_log[..., None] + other_log, -2)
            ),
            "opponent_wait_logits": (
                shanten_log[..., 0, None]
                + improving_log[..., 0, :]
                - tf.reduce_logsumexp(
                    tf.stack(
                        [
                            shanten_log[..., 0, None] + other_log[..., 0, :],
                            tf.broadcast_to(
                                tf.reduce_logsumexp(shanten_log[..., 1:], -1)[
                                    ..., None
                                ],
                                tf.shape(other_log[..., 0, :]),
                            ),
                        ],
                        -1,
                    ),
                    -1,
                )
            ),
            "opponent_tenpai_logits": (
                outputs["opponent_shanten_logits"][..., 0]
                - tf.reduce_logsumexp(
                    outputs["opponent_shanten_logits"][..., 1:], -1
                )
            ),
            "discard_origin_logits": origins,
            "riichi_action_logits": tf.reduce_logsumexp(riichi, -1),
            "tenpai_decision_logits": decision,
            "response_action_logits": tf.stack(
                [
                    response[:, 0],
                    tf.reduce_logsumexp(chi, (1, 2)),
                    tf.reduce_logsumexp(pon, (1, 2)),
                    response[:, 149],
                ],
                axis=-1,
            ),
            "chi_pattern_logits": tf.reduce_logsumexp(chi, -1),
        }


def build_model():
    inputs = {
        name: tf.keras.Input(
            shape=definition["shape"], dtype=definition["dtype"], name=name
        )
        for name, definition in feature_vector.INPUTS.items()
    }
    bank = decision_layers.HandBank(name="hand_bank")(inputs)
    encoded = [
        bank[:, 0],
        tf.keras.layers.GlobalAveragePooling1D()(bank),
        decision_layers.RiverEncoding(name="river_encoder")(inputs),
        *[
            tf.keras.layers.Flatten()(NumericFeatures(dtype="float32")(value))
            for name, value in inputs.items()
            if name
            not in {
                "candidate_hand_counts",
                "candidate_hand_red",
                "candidate_melds",
                "candidate_hand_valid",
                "river_tile_id",
            }
        ],
    ]
    hidden = tf.keras.layers.Dense(
        512, activation="gelu", name="board_projection"
    )(tf.keras.layers.Concatenate()(encoded))
    for block in range(4):
        residual = tf.keras.layers.LayerNormalization(
            name=f"board_block_{block}_norm"
        )(hidden)
        residual = tf.keras.layers.Dense(
            2400, activation="gelu", name=f"board_block_{block}_expand"
        )(residual)
        residual = tf.keras.layers.Dense(
            512, name=f"board_block_{block}_contract"
        )(residual)
        hidden = tf.keras.layers.Add(name=f"board_block_{block}_add")(
            [hidden, residual]
        )
    hidden = tf.keras.layers.Dense(512, activation="gelu", name="board_state")(
        hidden
    )
    hidden = tf.keras.layers.LayerNormalization(
        center=False,
        scale=False,
        epsilon=1e-5,
        dtype="float32",
        name="board_state_normalized",
    )(hidden)
    candidates, selected, _ = decision_layers.DecisionCandidates(
        name="decision_candidates"
    )([inputs, bank, hidden])
    candidates = tf.keras.layers.LayerNormalization(
        center=False,
        scale=False,
        epsilon=1e-5,
        dtype="float32",
        name="candidate_state_normalized",
    )(candidates)
    attack = tf.keras.layers.Dense(
        256, activation="gelu", name="attack_residual"
    )(candidates)
    defense = tf.keras.layers.Dense(
        256, activation="gelu", name="defense_residual"
    )(candidates)
    attack_predictions = tf.keras.layers.Dense(
        3, dtype="float32", name="attack_predictions"
    )(attack)
    defense_predictions = tf.keras.layers.Dense(
        6, dtype="float32", name="defense_predictions"
    )(defense)
    attack_decisions = tf.keras.layers.Dense(
        1, dtype="float32", name="attack_decisions"
    )(attack)
    defense_decisions = tf.keras.layers.Dense(
        1, dtype="float32", name="defense_decisions"
    )(defense)
    outputs = {
        "win_decision_logits": PolicyMask(
            "win_decision_logits", name="win_legality", dtype="float32"
        )(
            [
                tf.keras.layers.Rescaling(
                    scale=0.0,
                    offset=[0.0, 1e4],
                    dtype="float32",
                    name="win_decision_logits",
                )(hidden[:, :2]),
                inputs,
            ]
        )
    }
    auxiliary_logits = []
    main_heads = (
        "discard_policy_logits",
        "riichi_policy_logits",
        "response_policy_logits",
    )
    for name in (
        "opponent_shanten_logits",
        "opponent_ukeire_by_shanten_logits",
        "opponent_hand_count_logits",
        "terminal_payment_logits",
        "round_completion_yaku_logits",
        "round_placement_logits",
        "final_placement_logits",
    ):
        values = tf.keras.layers.Dense(
            math.prod(feature_vector.ACTOR_OUTPUTS[name]), dtype="float32"
        )(hidden)
        values = tf.keras.layers.Reshape(
            feature_vector.ACTOR_OUTPUTS[name], dtype="float32"
        )(values)
        auxiliary_logits.append(values)
        if name == "opponent_hand_count_logits":
            values = HeldCountMask(name=name, dtype="float32")([values, inputs])
        if name == "terminal_payment_logits":
            values = PaymentDistribution(name=name, dtype="float32")(
                [values, inputs]
            )
        outputs[name] = values
    # Raw, unmasked auxiliary logits avoid feeding -1e9 legality sentinels
    # into the final pass. Their own supervised losses retain the masks.
    context = tf.keras.layers.Dense(128, activation="gelu")(
        tf.keras.layers.Concatenate()(
            [
                hidden,
                *[
                    tf.keras.layers.Flatten()(NumericFeatures()(value))
                    for value in auxiliary_logits
                ],
            ]
        )
    )
    context = tf.keras.layers.Concatenate()([hidden, context])
    final = decision_layers.FinalCandidates()(
        [
            tf.keras.layers.Concatenate()(
                [
                    candidates,
                    attack_decisions,
                    defense_decisions,
                    attack_predictions,
                    defense_predictions,
                ]
            ),
            context,
            selected,
        ]
    )
    final = tf.keras.layers.Dense(256, activation="gelu", name="final_pass")(
        final
    )
    final = tf.keras.layers.LayerNormalization(
        center=False,
        scale=False,
        epsilon=1e-5,
        dtype="float32",
        name="final_pass_normalized",
    )(final)
    final = decision_layers.UnpackCandidates()(
        [
            tf.keras.layers.Dense(1, dtype="float32", name="final_scores")(
                final
            ),
            selected,
            hidden,
        ]
    )[:, :, 0]
    for name, values in (
        ("discard_policy_logits", final[:, :38]),
        ("riichi_policy_logits", final[:, 38:114]),
        (
            "response_policy_logits",
            decision_layers.ResponseKan(dtype="float32")([final, inputs]),
        ),
        ("kan_action_logits", final[:, 263:298]),
    ):
        outputs[name] = PolicyMask(name, name=name, dtype="float32")(
            [values, inputs]
        )
    outputs.update(PolicySummaries(dtype="float32")([outputs, inputs]))
    return tf.keras.Model(inputs, outputs, name="visible_actor")
