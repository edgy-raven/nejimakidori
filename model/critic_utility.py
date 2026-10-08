"""Detached action advantages from the jointly trained payment distribution."""

import tensorflow as tf

from model import critic, joint_settlement

ACTION_CHUNK = 1024
POSITION_CHUNK = 128


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


def payoff_contexts(inputs):
    """Share fixed payoff tables across identical public hand contexts.

    Public scores and bank amounts are integral points. Keep every rule
    input in the key, including tie priority and the current round.
    """
    keys, rows = tf.raw_ops.UniqueV2(
        x=tf.concat(
            [
                tf.reshape(tf.cast(value, tf.float32), [tf.shape(value)[0], -1])
                for value in inputs.values()
            ],
            axis=1,
        ),
        axis=[0],
    )
    first = tf.math.unsorted_segment_min(
        tf.range(tf.shape(rows)[0]), rows, tf.shape(keys)[0]
    )
    return {
        name: tf.gather(value, first) for name, value in inputs.items()
    }, rows


@tf.function(jit_compile=True)
def position_payoffs(inputs):
    states = joint_settlement.settlement_states(inputs)
    return joint_settlement.continuation_payoffs(
        scores=states["joint_scores"],
        ranks=states["joint_ranks"],
        support=states["joint_support"],
        dealer=inputs["dealer"],
        hand_number=inputs["hand_number"],
        pot=inputs["pot"],
    )


@tf.function(jit_compile=True)
def action_utilities(predictor, inputs, payoffs):
    """Integrate action probabilities against shared public-state payoffs."""
    return tf.stop_gradient(
        joint_settlement.expectation(
            joint_settlement.probabilities(
                {**inputs, **predictor.distributions(inputs)}
            )[0],
            payoffs,
        )
    )


@tf.function
def advantages(network, inputs, outputs, settings):
    """Score active legal actions in bounded chunks; never backpropagate AWR."""
    features, _, _ = inputs
    active = active_candidates(features, outputs["selected"])
    selected = tf.boolean_mask(outputs["selected"], active)
    state = tf.stop_gradient(
        tf.boolean_mask(outputs["critic_candidate_state"], active)
    )
    safe = tf.boolean_mask(outputs["immediate_safe"], active)
    wait = tf.boolean_mask(outputs["immediate_wait"], active)
    public = joint_settlement.public_inputs(features)
    count = tf.shape(features["decision_phase"])[0]
    contexts, context_rows = payoff_contexts(
        {**public, "dealer": features["dealer_relative_seat"]}
    )
    context_count = tf.shape(contexts["dealer"])[0]
    payoffs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for begin in tf.range(0, context_count, POSITION_CHUNK):
        indices = tf.minimum(
            begin + tf.range(POSITION_CHUNK), context_count - 1
        )
        values = position_payoffs(
            {
                name: tf.gather(value, indices)
                for name, value in contexts.items()
            }
        )
        payoffs = payoffs.write(
            begin // POSITION_CHUNK,
            values[: tf.minimum(POSITION_CHUNK, context_count - begin)],
        )
    payoffs = payoffs.concat()
    candidate_public = critic.public_inputs(features, selected)
    allowed = critic.immediate_decisions(features, selected)
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
                "immediate_wait": tf.gather(wait, indices),
                "immediate_allowed": tf.gather(allowed, indices),
                **{
                    name: tf.gather(value, rows)
                    for name, value in public.items()
                },
                **{
                    name: tf.gather(value, indices)
                    for name, value in candidate_public.items()
                },
            },
            tf.gather(payoffs, tf.gather(context_rows, rows)),
        )
        utilities = utilities.write(
            begin // ACTION_CHUNK,
            utility[: tf.minimum(ACTION_CHUNK, tf.shape(selected)[0] - begin)],
        )
    utility = tf.stop_gradient(utilities.concat())
    probability = tf.gather_nd(
        candidate_probabilities(features, outputs), selected
    )
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
