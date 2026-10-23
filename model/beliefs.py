"""Opponent belief heads, constraints and supervision."""

import math

import tensorflow as tf

from model import actions, losses

TENPAI_UKEIRE_WEIGHT = 9.0


def held_count_context(inputs):
    """Known meld copies, unseen physical copies and concealed hand sizes."""
    tile = tf.cast(inputs["meld_tile_ids"][:, 1:], tf.int32)
    red = tf.cast(inputs["meld_tile_red"][:, 1:], tf.bool)
    code = tf.where(red, 34 + tile // 9, tile)
    valid = tf.cast(inputs["meld_tile_valid"][:, 1:], tf.float32)
    lower = tf.reduce_sum(tf.one_hot(code, 37) * valid[..., None], axis=(2, 3))
    unseen = 4 - tf.cast(inputs["known_unavailable_counts"], tf.float32)
    unseen_red = (
        1
        - tf.cast(inputs["candidate_hand_red"][:, 0], tf.float32)
        - tf.cast(inputs["public_visible_red_counts"], tf.float32)
    )
    correction = tf.linalg.matmul(unseen_red, tf.one_hot([4, 13, 22], 34))
    available = tf.concat([unseen - correction, unseen_red], -1)
    size = 13 - 3 * tf.reduce_sum(
        tf.cast(inputs["meld_valid"][:, 1:], tf.float32), axis=-1
    )
    return lower, available, size


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class HeldCountMask(tf.keras.layers.Layer):
    def call(self, values):
        logits, inputs = values
        lower, available, _ = held_count_context(inputs)
        upper = tf.minimum(4.0, lower + tf.maximum(0.0, available[:, None, :]))
        classes = tf.cast(tf.range(5), tf.float32)
        valid = (classes >= lower[..., None]) & (classes <= upper[..., None])
        return tf.where(valid, logits, -1e4)


def opponent_summaries(outputs):
    shanten_log = tf.nn.log_softmax(outputs["opponent_shanten_logits"], -1)
    improving_log = -tf.nn.softplus(
        -outputs["opponent_ukeire_by_shanten_logits"]
    )
    other_log = -tf.nn.softplus(outputs["opponent_ukeire_by_shanten_logits"])
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
    }


BELIEF_HEADS = (
    "opponent_shanten_logits",
    "opponent_ukeire_by_shanten_logits",
    "opponent_hand_count_logits",
)


class Probabilities(tf.keras.layers.Layer):
    def call(self, logits):
        return {
            name: (
                tf.sigmoid(logits[name])
                if name == BELIEF_HEADS[1]
                else tf.nn.softmax(logits[name], axis=-1)
            )
            for name in BELIEF_HEADS
        }


class OpponentBeliefs(tf.keras.Model):
    """Supervised opponent heads on the joint model's board encoder."""

    def __init__(self, encoder):
        super().__init__(name="opponent_beliefs", autocast=False)
        self.encoder = encoder
        self.heads = {
            name: tf.keras.layers.Dense(
                math.prod(actions.ACTOR_OUTPUTS[name]),
                dtype="float32",
                name=name.removesuffix("_logits") + "_scores",
            )
            for name in BELIEF_HEADS
        }
        self.held_count_mask = HeldCountMask(dtype="float32")
        self.count_projection = CountProjection(dtype="float32")

    def call(self, inputs, hidden):
        outputs = {
            name: tf.reshape(
                self.heads[name](hidden),
                [-1, *actions.ACTOR_OUTPUTS[name]],
            )
            for name in BELIEF_HEADS
        }
        outputs[BELIEF_HEADS[2]] = self.held_count_mask(
            [outputs[BELIEF_HEADS[2]], inputs]
        )
        outputs[BELIEF_HEADS[2]] = self.count_projection(
            [outputs[BELIEF_HEADS[2]], inputs]
        )
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
        lower, available, size = held_count_context(features)
        classes = tf.range(5, dtype=tf.float32) - lower[..., None]
        weights = count_weights(features)

        def solve(logits):
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

            @tf.function(jit_compile=True)
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
                (
                    0,
                    tf.zeros_like(size),
                    tf.zeros_like(available),
                    float("inf"),
                ),
            )
            return (
                (
                    logits
                    + classes
                    * (alpha[..., None, None] + beta[:, None, :, None])
                    / weights[..., None]
                ),
                error,
                beta,
            )

        @tf.custom_gradient
        def projected(logits):
            result, error, beta = tf.function(solve, jit_compile=True)(
                tf.stop_gradient(logits)
            )

            @tf.function(jit_compile=True)
            def gradient(upstream, error_gradient):
                probability = tf.cast(tf.nn.softmax(result), tf.float64)
                counts = tf.cast(classes, tf.float64)
                weight = tf.cast(weights, tf.float64)
                mean = tf.reduce_sum(probability * counts, -1)
                curvature = (
                    tf.reduce_sum(
                        probability * tf.square(counts - mean[..., None]), -1
                    )
                    / weight
                )
                active = tf.cast(beta < 0, tf.float64)
                cross = curvature * active[:, None]
                column_curvature = tf.reduce_sum(cross, 1)
                right = (
                    tf.reduce_sum(tf.cast(upstream, tf.float64) * counts, -1)
                    / weight
                )
                right_column = tf.reduce_sum(right, 1) * active
                scaled = tf.math.divide_no_nan(cross, column_curvature[:, None])
                # Eliminate tile multipliers; only a 3x3 hand system remains.
                # Float64 protects cancellation in the Schur complement.
                schur = tf.linalg.diag(tf.reduce_sum(curvature, -1)) - (
                    scaled @ tf.transpose(cross, [0, 2, 1])
                )
                right_row = tf.reduce_sum(right, -1) - tf.reduce_sum(
                    scaled * right_column[:, None], -1
                )
                eigenvalues, eigenvectors = tf.linalg.eigh(schur)
                # Redundant physical constraints leave zero eigenvalues.
                # Their pseudoinverse leaves projected probabilities unique.
                inverse = tf.where(
                    eigenvalues
                    > tf.reduce_max(eigenvalues, -1, keepdims=True) * 1e-10,
                    tf.math.divide_no_nan(
                        tf.ones_like(eigenvalues), eigenvalues
                    ),
                    0.0,
                )
                row = tf.linalg.matvec(
                    eigenvectors,
                    inverse
                    * tf.linalg.matvec(
                        eigenvectors, right_row, transpose_a=True
                    ),
                )
                column = tf.math.divide_no_nan(
                    right_column - tf.reduce_sum(cross * row[..., None], 1),
                    column_curvature,
                )
                return upstream - tf.cast(
                    probability
                    * (counts - mean[..., None])
                    * (row[..., None, None] + column[:, None, :, None]),
                    upstream.dtype,
                )

            return (result, error), gradient

        result, error = projected(logits)
        tf.debugging.assert_less_equal(
            error, 1e-4, message="Count projection failed to converge"
        )
        return result


def belief_loss_terms(features, labels, outputs, mean=losses.mean_valid):
    """Opponent supervision for joint training."""
    terms = {
        "opponent_shanten_loss": losses.classification(
            labels["opponent_shanten"], outputs[BELIEF_HEADS[0]], mean=mean
        ),
    }
    terms["opponent_ukeire_loss"], metrics = ukeire_loss(
        labels, outputs, mean=mean
    )
    terms["opponent_hand_count_loss"] = losses.classification(
        labels["opponent_hand_counts"],
        outputs[BELIEF_HEADS[2]],
        count_weights(features),
        mean=mean,
    )
    lower, available, size = held_count_context(features)
    concealed = tf.reduce_sum(
        tf.nn.softmax(outputs[BELIEF_HEADS[2]])
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


def ukeire_loss(labels, outputs, mean=losses.mean_valid):
    target = tf.cast(labels["opponent_ukeire"], tf.float32)
    distance = tf.cast(labels["opponent_shanten"], tf.int32)
    logits = tf.gather(
        outputs["opponent_ukeire_by_shanten_logits"],
        tf.maximum(distance, 0),
        axis=2,
        batch_dims=2,
    )
    valid = (target >= 0) & (distance[..., None] >= 0)
    tenpai = valid & (distance[..., None] == 0)
    other = valid & (distance[..., None] > 0)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.maximum(target, 0), logits=logits
    )
    tenpai_loss = mean(losses, tenpai)
    other_loss = mean(losses, other)
    loss = tf.math.divide_no_nan(
        TENPAI_UKEIRE_WEIGHT * tenpai_loss + other_loss,
        TENPAI_UKEIRE_WEIGHT * mean(tf.ones_like(target), tenpai)
        + mean(tf.ones_like(target), other),
    )
    probability = tf.math.sigmoid(logits)
    metrics = {
        "tenpai_ukeire_loss": tenpai_loss,
        "other_ukeire_loss": other_loss,
        "tenpai_wait_recall_at_0_5": mean(
            tf.cast(probability >= 0.5, tf.float32), tenpai & (target == 1)
        ),
        "tenpai_wait_precision_at_0_5": mean(
            target, tenpai & (probability >= 0.5)
        ),
        "tenpai_wait_brier": mean(tf.square(probability - target), tenpai),
        "tenpai_wait_positive_brier": mean(
            tf.square(probability - target), tenpai & (target == 1)
        ),
        "tenpai_wait_recall_at_5": mean(
            tf.reduce_sum(
                tf.one_hot(tf.math.top_k(probability, 5).indices, 34), -2
            ),
            tenpai & (target == 1),
        ),
        "joint_wait_brier": mean(
            tf.square(
                tf.math.sigmoid(outputs["opponent_wait_logits"])
                - tf.cast(tenpai & (target == 1), tf.float32)
            ),
            valid,
        ),
        "opponent_ukeire_brier": mean(
            tf.square(
                tf.math.sigmoid(outputs["opponent_ukeire_logits"]) - target
            ),
            valid,
        ),
    }
    return loss, metrics
