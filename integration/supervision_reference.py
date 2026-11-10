"""Combined head supervision for model gradient and export checks."""

import tensorflow as tf

from model import actor as actor_layers
from model import beliefs, critic, losses


def objective_metrics(inputs, outputs, statistics=None):
    means = losses.MaskedMeans()
    if statistics is None:
        objective_terms(inputs, outputs, mean=means.collect)
        means.reduce()
    else:
        means.values = iter(
            tf.unstack(
                tf.math.divide_no_nan(statistics[:, 0], statistics[:, 1])
            )
        )
    return objective_terms(inputs, outputs, mean=means.read)


def objective_terms(inputs, outputs, mean=losses.mean_valid):
    loss, metrics = critic.objective_metrics(outputs, mean=mean)
    metrics["critic_loss"] = loss
    loss *= 0.1
    metrics["value_loss"] = mean(
        tf.square(outputs["state_value"] - inputs[2]["returns"]),
        tf.ones_like(inputs[2]["returns"]),
    )
    metrics["value_bias"] = mean(
        outputs["state_value"] - inputs[2]["returns"],
        tf.ones_like(inputs[2]["returns"]),
    )
    loss += metrics["value_loss"]
    metrics.update(actor_layers.offline_loss_terms(inputs, outputs, mean=mean))
    belief_terms, belief_metrics = beliefs.belief_loss_terms(
        inputs[0], inputs[1], outputs, mean=mean
    )
    metrics.update(belief_terms)
    metrics["belief_loss"] = tf.add_n(list(belief_terms.values())) / 3
    metrics.update(belief_metrics)
    metrics["board_loss"] = 0.01 * (
        metrics["completion_yaku_loss"] + metrics["completion_score_loss"]
    )
    loss += (
        metrics["policy_loss"]
        + metrics["policy_prior_loss"]
        + metrics["board_loss"]
        + metrics["belief_loss"]
        - actor_layers.POLICY_ENTROPY_WEIGHT * metrics["policy_entropy"]
    )
    specialist_terms = actor_layers.auxiliary_loss_terms(
        inputs, outputs, mean=mean
    )
    metrics.update(specialist_terms)
    metrics["specialist_loss"] = 0.01 * tf.add_n(
        list(specialist_terms.values())
    )
    metrics["scoring_loss"] = 0.01 * (
        metrics["score_nll"] + metrics["bonus_han_nll"]
    )
    loss += metrics["specialist_loss"] + metrics["scoring_loss"]
    metrics["constraint_loss"] = mean(
        outputs["constraint_loss"], tf.ones_like(outputs["constraint_loss"])
    )
    loss += 0.01 * metrics["constraint_loss"]
    metrics["joint_loss"] = loss
    return loss, metrics


def validation_statistics(inputs, outputs):
    means = losses.MaskedMeans()
    metrics = objective_terms(inputs, outputs, mean=means.collect)[1]
    return {
        "statistics": tf.stack(means.statistics),
        "counts": {
            "example_count": tf.cast(
                tf.shape(inputs[2]["points"])[0], tf.float32
            ),
            **{
                name: value
                for name, value in metrics.items()
                if name.endswith(("_count", "_correct", "_nll_sum"))
            },
        },
    }
