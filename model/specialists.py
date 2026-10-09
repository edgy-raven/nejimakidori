"""Direct attack/defense supervision alongside the final AWR objective."""

import tensorflow as tf

from model import critic, decision_layers, objectives

CONTRACT = {
    "affinity": "candidate_shortest_route_tanyao_flush_outside_equal_bce",
    "defense": "binary_legal_ron_per_opponent_no_payment_magnitude",
    "sakigiri": "preferred_early_discard_soft_pair_endpoint_normalized",
    "sakigiri_scores": "defense_only",
    "scaling": "shared_auxiliary_loss_weight",
    "candidate_normalization": "mean_valid_entries_then_mean_decisions",
}


def order_loss(labels, scores, mean=objectives.mean_valid):
    pair = tf.cast(labels["sakigiri_pair"], tf.int32)
    values = tf.gather(scores, tf.maximum(pair, 0), batch_dims=1)
    return mean(
        tf.nn.softplus(values[:, 1] - values[:, 0]),
        tf.reduce_all(pair >= 0, axis=1),
        tf.maximum(labels["sakigiri_weight"], 0.0),
    )


def loss_terms(inputs, outputs, mean=objectives.mean_valid):
    features, labels, _ = inputs
    selected = outputs["selected"]
    hand_indices = decision_layers.candidate_hand_indices(features)
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
    terms = {
        "attack_affinity_loss": objectives.candidate_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.maximum(affinity, 0.0),
                logits=outputs["attack_yaku_logits"],
            ),
            affinity >= 0,
            selected[:, 0],
            mean=mean,
        ),
    }
    risk = critic.ron_risk_labels(labels, selected, outputs["structure"])
    terms["defense_ron_loss"] = objectives.candidate_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(risk, tf.float32),
            logits=outputs["defense_ron_logits"],
        ),
        tf.broadcast_to(
            critic.discard_decisions(features, selected)[:, None],
            tf.shape(risk),
        ),
        selected[:, 0],
        mean=mean,
    )
    terms["sakigiri_defense_loss"] = order_loss(
        labels, outputs["defense_scores"][:, :37], mean
    )
    return terms
