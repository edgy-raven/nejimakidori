"""Immediate hand endings, separate from the surviving-hand critic.

Own native-legal wins and nine-terminals aborts are resolved by the action
bridge. This model handles hidden ron exposure and terminal draw masks for
candidate actions. Future payments never supply its amount distribution.
"""

import tensorflow as tf

from model import joint_settlement, outcomes, payment_scoring


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
        logits = tf.where(
            inputs["immediate_safe"]
            | (inputs["immediate_wait"] <= 0)
            | (inputs["immediate_ankan"][:, None] & inputs["has_meld"][:, 1:]),
            -1e9,
            tf.math.log(tf.maximum(probability, 1e-30))
            - tf.math.log1p(-probability),
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
        score_allowed &= payment_scoring.minimum_han_mask(minimum_han)
        # Only kokushi can rob a concealed kan under the native rules.
        score_allowed &= ~inputs["immediate_ankan"][:, None, None] | (
            (tf.constant(payment_scoring.SCORE_HAN, tf.int32) == 0)[None, None]
            & (tf.constant(payment_scoring.SCORE_FU, tf.int32)[None, None] == 1)
        )
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
        draw_logits = tf.where(draw_allowed, self.draw(hidden), -1e9)
        draw_probability = tf.nn.softmax(draw_logits)
        exhaustive = inputs["immediate_draw_kind"] == 1
        abort = inputs["immediate_draw_kind"] == 2
        return {
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


def loss_terms(labels, selected, structure, public, outputs):
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
