"""Detached action advantages from the jointly trained payment distribution."""

import numpy
import tensorflow as tf

from . import critic, objectives, outcomes, payment_bonus

ACTION_CHUNK = 512


def placement_utility(prediction, variance, scores, remaining, scale):
    """Integrate ST3 rank value over payment and future-score uncertainty.

    Net actor transfers are shared equally by opponents. Independent Gumbel
    continuation shocks give coherent rank probabilities at each quadrature
    node. Current-score constants cancel from the action advantage.
    """
    nodes, weights = numpy.polynomial.hermite.hermgauss(17)
    delta = prediction[:, None] + tf.sqrt(2 * variance[:, None]) * tf.constant(
        nodes, tf.float32
    )
    probability = objectives.placement_prior(
        scores[:, None, :]
        + delta[..., None] * tf.constant([1.0, -1 / 3, -1 / 3, -1 / 3]),
        remaining[:, None],
        scale=scale,
    )
    return prediction / 1000 + tf.reduce_sum(
        tf.reduce_sum(
            probability * tf.constant([125.0, 60.0, -5.0, -255.0]), -1
        )
        * tf.constant(weights / numpy.sqrt(numpy.pi), tf.float32),
        axis=-1,
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
    return tf.tensor_scatter_nd_add(
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


def point_moments(probability, correlation):
    """Honba-free actor points; empirical correlation approximates covariance."""
    values = tf.constant(outcomes.PAYMENT_VALUES, tf.float32)
    return net_moments(
        tf.stack(
            [
                tf.reduce_sum(probability * values, axis=-1),
                tf.reduce_sum(probability * tf.square(values), axis=-1),
            ],
            axis=-1,
        ),
        correlation,
    )


def net_moments(moments, correlation, cell_ids=None):
    """Reduce per-cell moments with the same actor payment covariance model."""
    coefficients = numpy.array(
        [
            int(recipient == 0)
            - int(payer == 0)
            + int(kind == 3 and payer == 0)
            for kind, payer, recipient in payment_bonus.CELLS
        ],
        numpy.float32,
    )
    if cell_ids is not None:
        coefficients = coefficients[list(cell_ids)]
    means = moments[..., 0]
    variances = tf.maximum(moments[..., 1] - tf.square(means), 0)
    signed_std = tf.boolean_mask(
        tf.sqrt(variances) * coefficients, coefficients != 0, axis=1
    )
    return (
        tf.reduce_sum(means * coefficients, axis=1),
        tf.maximum(
            tf.reduce_sum(
                tf.linalg.matmul(signed_std, correlation) * signed_std, axis=1
            ),
            0,
        ),
    )


@tf.function(jit_compile=True)
def action_utilities(predictor, inputs, settings):
    """Fuse scoring and payment moments into one device-side utility vector."""
    prediction, variance = net_moments(
        predictor.distributions(inputs, actor_moments=True)[
            "actor_payment_moments"
        ],
        tf.constant(settings["correlation"], tf.float32),
        critic.ACTOR_CELLS,
    )
    return placement_utility(
        prediction=prediction,
        variance=settings["variance_scale"] * variance,
        scores=inputs["scores"],
        remaining=inputs["remaining"],
        scale=settings["placement_scale"],
    )


@tf.function
def advantages(network, inputs, outputs, settings):
    """Score active legal actions in bounded chunks; never backpropagate AWR."""
    features, _, auxiliary = inputs
    active = active_candidates(features, outputs["selected"])
    selected = tf.boolean_mask(outputs["selected"], active)
    state = tf.stop_gradient(
        tf.boolean_mask(outputs["critic_candidate_state"], active)
    )
    safe = tf.boolean_mask(outputs["immediate_safe"], active)
    allowed = critic.discard_decisions(features, selected)
    utilities = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for begin in tf.range(0, tf.shape(selected)[0], ACTION_CHUNK):
        # Repeat the final action to keep the compiled decoder batch fixed.
        # Per-action forecasts are independent; discard padded results below.
        indices = tf.minimum(
            begin + tf.range(ACTION_CHUNK), tf.shape(selected)[0] - 1
        )
        rows = tf.gather(selected[:, 0], indices)
        utility = action_utilities(
            network.critic,
            {
                "state": tf.gather(state, indices),
                "immediate_safe": tf.gather(safe, indices),
                "immediate_allowed": tf.gather(allowed, indices),
                "scores": tf.gather(features["seat_context"][:, :, 0], rows)
                * 100000,
                "remaining": tf.gather(
                    features["scheduled_hands_remaining"], rows
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
            },
            settings,
        )
        utilities = utilities.write(
            begin // ACTION_CHUNK,
            utility[: tf.minimum(ACTION_CHUNK, tf.shape(selected)[0] - begin)],
        )
    utility = tf.stop_gradient(utilities.concat())
    probability = tf.gather_nd(
        candidate_probabilities(features, outputs), selected
    )
    count = tf.shape(features["decision_phase"])[0]
    tf.debugging.assert_near(
        tf.math.unsorted_segment_sum(probability, selected[:, 0], count),
        tf.ones(count),
        atol=1e-5,
        message="utility support must match the active legal policy",
    )
    baseline = tf.math.unsorted_segment_sum(
        probability * utility, selected[:, 0], count
    )
    # Recorded action identifiers already have the response-kan mapping.
    chosen = critic.chosen_actions(features, inputs[1])
    recorded = tf.gather(chosen, selected[:, 0]) == selected[:, 1]
    tf.debugging.assert_equal(
        tf.boolean_mask(selected[:, 0], recorded), tf.range(count)
    )
    return tf.stop_gradient(
        (tf.boolean_mask(utility, recorded) - baseline)
        / settings["temperature"]
    )
