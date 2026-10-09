"""Explicit opponent beliefs on a reusable board encoder."""

import math

import tensorflow as tf

from model import feature_vector, model, objectives

HEADS = (
    "opponent_shanten_logits",
    "opponent_ukeire_by_shanten_logits",
    "opponent_hand_count_logits",
)


class Probabilities(tf.keras.layers.Layer):
    def call(self, logits):
        return {
            name: (
                tf.sigmoid(logits[name])
                if name == HEADS[1]
                else tf.nn.softmax(logits[name], axis=-1)
            )
            for name in HEADS
        }


class OpponentBeliefs(tf.keras.Model):
    """Supervised heads on a supplied shared or standalone board encoder."""

    def __init__(self, encoder=None, project_counts=True):
        super().__init__(name="opponent_beliefs", autocast=False)
        if encoder is None:
            policy = tf.keras.mixed_precision.global_policy()
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
            inputs, hidden, _, _, _ = model.encode_inputs(True, 2)
            tf.keras.mixed_precision.set_global_policy(policy)
            encoder = tf.keras.Model(
                inputs, hidden, name="belief_encoder", autocast=False
            )
        self.encoder = encoder
        self.heads = {
            name: tf.keras.layers.Dense(
                math.prod(feature_vector.ACTOR_OUTPUTS[name]),
                dtype="float32",
                name=name.removesuffix("_logits") + "_scores",
            )
            for name in HEADS
        }
        self.held_count_mask = model.HeldCountMask(dtype="float32")
        self.count_projection = (
            CountProjection(dtype="float32") if project_counts else None
        )

    def call(self, inputs, hidden=None):
        if hidden is None:
            hidden = self.encoder(inputs)
        outputs = {
            name: tf.reshape(
                self.heads[name](hidden),
                [-1, *feature_vector.ACTOR_OUTPUTS[name]],
            )
            for name in HEADS
        }
        outputs[HEADS[2]] = self.held_count_mask([outputs[HEADS[2]], inputs])
        if self.count_projection is not None:
            outputs[HEADS[2]] = self.count_projection(
                [outputs[HEADS[2]], inputs]
            )
        outputs.update(model.opponent_summaries(outputs))
        return outputs


def count_weights(features):
    """Shared strategic weights for count supervision and projection."""
    base = tf.constant(list(range(34)) + [4, 13, 22])
    winds = tf.cast(features["seat_wind"][:, 1:], tf.int32) + 27
    prevailing = tf.cast(features["prevailing_wind"], tf.int32) + 27
    yakuhai = (
        (base[None, None] >= 31)
        | (base[None, None] == winds[..., None])
        | (base[None, None] == prevailing[:, None, None])
    )
    return (
        1
        + 9
        * tf.gather(
            tf.cast(features["dora_multiplicity"], tf.float32), base, axis=-1
        )[:, None]
        + 9 * tf.cast(tf.range(37) >= 34, tf.float32)
        + 4 * tf.cast(yakuhai, tf.float32)
    )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class CountProjection(tf.keras.layers.Layer):
    """Weighted minimum-KL marginal projection using public constraints.

    Input logits must already have HeldCountMask applied. This enforces
    expected concealed sizes and shared physical copy limits, not a joint
    distribution over legal hands. No hidden labels enter the projection.
    """

    def call(self, values):
        logits, features = values
        lower, available, size = model.held_count_context(features)
        classes = tf.range(5, dtype=tf.float32) - lower[..., None]
        weights = count_weights(features)

        def moments(alpha, beta):
            probability = tf.nn.softmax(
                logits
                + classes
                * (alpha[..., None, None] + beta[:, None, :, None])
                / weights[..., None],
                -1,
            )
            mean = tf.reduce_sum(probability * classes, -1)
            curvature = (
                tf.reduce_sum(
                    probability * tf.square(classes - mean[..., None]), -1
                )
                / weights
            )
            return mean, curvature

        def step(iteration, alpha, beta, error):
            mean, curvature = moments(alpha, beta)
            alpha += tf.clip_by_value(
                (size - tf.reduce_sum(mean, -1))
                / tf.maximum(tf.reduce_sum(curvature, -1), 1e-7),
                -2.0,
                2.0,
            )
            mean, curvature = moments(alpha, beta)
            beta = tf.minimum(
                0.0,
                beta
                + tf.clip_by_value(
                    (available - tf.reduce_sum(mean, 1))
                    / tf.maximum(tf.reduce_sum(curvature, 1), 1e-7),
                    -2.0,
                    2.0,
                ),
            )
            mean, _ = moments(alpha, beta)
            error = tf.maximum(
                tf.reduce_max(tf.abs(tf.reduce_sum(mean, -1) - size)),
                tf.reduce_max(
                    tf.abs(
                        tf.where(
                            beta < 0,
                            tf.reduce_sum(mean, 1) - available,
                            tf.nn.relu(tf.reduce_sum(mean, 1) - available),
                        )
                    )
                ),
            )
            return iteration + 1, alpha, beta, error

        _, alpha, beta, error = tf.while_loop(
            lambda iteration, alpha, beta, error: (iteration < 2048)
            & (error > 1e-5),
            step,
            (0, tf.zeros_like(size), tf.zeros_like(available), float("inf")),
        )
        tf.debugging.assert_less_equal(
            error, 1e-4, message="Count projection failed to converge"
        )
        return (
            logits
            + classes
            * (alpha[..., None, None] + beta[:, None, :, None])
            / weights[..., None]
        )


def loss_terms(features, labels, outputs, mean=objectives.mean_valid):
    """Belief-only supervision; shared by standalone and joint trainers."""
    terms = {
        "opponent_shanten_loss": objectives.classification(
            labels["opponent_shanten"], outputs[HEADS[0]], mean=mean
        ),
    }
    terms["opponent_ukeire_loss"], metrics = objectives.ukeire_loss(
        labels, outputs, mean=mean
    )
    terms["opponent_hand_count_loss"] = objectives.classification(
        labels["opponent_hand_counts"],
        outputs[HEADS[2]],
        count_weights(features),
        mean=mean,
    )
    lower, available, size = model.held_count_context(features)
    concealed = tf.reduce_sum(
        tf.nn.softmax(outputs[HEADS[2]])
        * (tf.range(5, dtype=tf.float32) - lower[..., None]),
        -1,
    )
    metrics.update(
        {
            name: mean(value, tf.ones_like(value))
            for name, value in {
                "count_size_error": tf.reduce_max(
                    tf.abs(tf.reduce_sum(concealed, -1) - size), -1
                ),
                "count_copy_overflow": tf.reduce_max(
                    tf.nn.relu(tf.reduce_sum(concealed, 1) - available), -1
                ),
            }.items()
        }
    )
    return terms, metrics
