"""Candidate route support and opponent ron liability predictions."""

import tensorflow as tf

from model import actions, input_features, losses, payment_scoring


def predict(*, inputs, candidates, selected, wait_logits, eligibility_logits):
    """Predict candidate targets once for each selected local action."""
    raw_route = tf.keras.layers.Dense(
        3, dtype="float32", name="route_affinity"
    )(candidates)
    route = RouteMask(dtype="float32")([raw_route, inputs, selected])
    return {
        "route_affinity_logits": route,
        "route_constraint_loss": losses.ConstraintLoss(
            binary=True, dtype="float32"
        )([raw_route, route, selected[:, 0]]),
        "candidate_ron_logits": RonRisk(dtype="float32")(
            [eligibility_logits, wait_logits, inputs, selected]
        ),
    }


class RonRisk(tf.keras.layers.Layer):
    """Ron requires a structural wait on the discarded tile and eligibility."""

    def call(self, values):
        eligibility, wait, inputs, selected = values
        # Local discard, dama and declaration each index the same 37 cuts.
        # Accepted-call children reach this head as local discard decisions.
        tile = tf.gather(list(range(34)) + [4, 13, 22], selected[:, 1] % 37)
        index = tf.stack([selected[:, 0], tile], axis=-1)
        eligibility = tf.gather_nd(tf.transpose(eligibility, [0, 2, 1]), index)
        wait = tf.gather_nd(
            tf.transpose(tf.stop_gradient(wait), [0, 2, 1]), index
        )
        safe = tf.gather_nd(
            tf.transpose(actions.guaranteed_safe_tiles(inputs), [0, 2, 1]),
            index,
        )
        # logit(sigmoid(w) * sigmoid(e)) without a probability-space log.
        logits = -tf.reduce_logsumexp(
            tf.stack([-wait, -eligibility, -wait - eligibility]), axis=0
        )
        return tf.where(
            actions.discard_decisions(inputs, selected)[:, None] & ~safe,
            logits,
            -1e4,
        )


class RouteMask(tf.keras.layers.Layer):
    def call(self, values):
        logits, inputs, selected = values
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
        allowed = payment_scoring.support(
            tf.gather_nd(
                tf.cast(input_features.candidate_melds(inputs), tf.float32),
                hands,
            )
        )
        # Outside routes include sequence-free honroutou/chinroutou.
        return tf.where(
            tf.stack(
                [
                    allowed[..., 1],
                    allowed[..., 18],
                    allowed[..., 10] | allowed[..., 11],
                ],
                axis=-1,
            ),
            logits,
            -1e9,
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
    return risk & ~tf.reduce_all(risk, axis=1)[:, None], legal >= 0


def loss_terms(features, labels, outputs, mean=losses.mean_valid):
    selected = outputs["selected"]
    hand_indices = actions.candidate_hand_indices(features)
    affinity = tf.cast(
        tf.gather_nd(
            labels["candidate_yaku_affinity"],
            tf.stack(
                [
                    selected[:, 0],
                    tf.cast(tf.gather_nd(hand_indices, selected), tf.int32),
                ],
                axis=1,
            ),
        ),
        tf.float32,
    )
    # Parent records have local-hand route labels only. Accepted-call child
    # hands are not represented in that bank and must not inherit hand zero.
    affinity = tf.where(
        ((selected[:, 1] >= 112) & (selected[:, 1] < 260))[:, None],
        -1.0,
        affinity,
    )
    terms = {
        "route_affinity_loss": losses.candidate_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.maximum(affinity, 0.0),
                logits=outputs["route_affinity_logits"],
            ),
            affinity >= 0,
            selected[:, 0],
            mean=mean,
        ),
    }
    risk, known = ron_risk_labels(labels, selected, outputs["structure"])
    terms["candidate_ron_loss"] = losses.candidate_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(risk, tf.float32),
            logits=outputs["candidate_ron_logits"],
        ),
        known & actions.discard_decisions(features, selected)[:, None],
        selected[:, 0],
        mean=mean,
    )
    return terms
