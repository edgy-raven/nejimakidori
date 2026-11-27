"""Assembly of the shared actor, beliefs and payment critic."""

import tensorflow as tf

from model import actor as actor_layers
from model import (
    beliefs,
    candidate_predictions,
    critic,
    encoder,
    joint_settlement,
)


class JointPolicyPayment(tf.keras.Model):
    """Shared policy, beliefs, auxiliaries and payment critic."""

    def __init__(self):
        super().__init__(autocast=False)
        inputs, board, candidates, selected, structure, critic_rivers = (
            encoder.encode_inputs()
        )
        self.beliefs = beliefs.Beliefs()
        belief_logits = self.beliefs(inputs, hidden=board, rivers=critic_rivers)
        # Downstream heads read these probabilities as fixed evidence. The
        # original logits retain gradients for each head's supervised loss.
        belief_probability = {
            name: tf.keras.ops.stop_gradient(value)
            for name, value in beliefs.Probabilities(dtype="float32")(
                belief_logits
            ).items()
        }
        wait = beliefs.StructuralWait(dtype="float32")(belief_logits)
        eligibility = tf.keras.layers.Reshape((3, 34), dtype="float32")(
            tf.keras.layers.Dense(
                3 * 34, dtype="float32", name="candidate_ron"
            )(board)
        )
        # Critic directional constraints operate on conditional improving-tile
        # probabilities. Policy evidence weights each slice by its shanten mass.
        shanten, improving, counts = (
            belief_probability[name] for name in beliefs.BELIEF_HEADS
        )
        flatten = tf.keras.layers.Flatten(dtype="float32")
        concatenate = tf.keras.layers.Concatenate(dtype="float32")
        completion = concatenate(
            [
                flatten(belief_probability[name])
                for name in beliefs.COMPLETION_HEADS
            ]
        )
        predicted = tf.keras.layers.Dense(
            128, activation="gelu", dtype="float32", name="policy_evidence"
        )(
            concatenate(
                [
                    tf.keras.ops.stop_gradient(board),
                    flatten(shanten),
                    flatten(
                        tf.keras.layers.Multiply(
                            name="shanten_weighted_ukeire", dtype="float32"
                        )([shanten[..., None], improving])
                    ),
                    flatten(counts),
                    completion,
                ]
            )
        )
        context = concatenate([tf.keras.ops.stop_gradient(board), predicted])
        context = tf.keras.layers.Add(dtype="float32")(
            [
                context,
                tf.keras.Sequential(
                    [
                        tf.keras.layers.LayerNormalization(dtype="float32"),
                        tf.keras.layers.Dense(
                            256, activation="gelu", dtype="float32"
                        ),
                        tf.keras.layers.Dense(
                            640, kernel_initializer="zeros", dtype="float32"
                        ),
                    ],
                    name="actor_refinement",
                )(context),
            ]
        )
        # Critic and belief losses retain their direct encoder paths.
        # Only the policy consumes fixed shared representations.
        encoded = {
            "candidate": candidates,
            "board": board,
            "selected": selected,
            "structure": structure,
            "critic_rivers": critic_rivers,
            **belief_logits,
            "opponent_wait_logits": wait,
            "ron_eligibility_logits": eligibility,
            "belief_context": concatenate(
                [flatten(shanten), flatten(improving), flatten(counts)]
            ),
            "completion_context": completion,
            "policy_context": context,
        }
        self.context = critic.HandContext(dtype="float32")
        self.public = critic.PublicContext(dtype="float32")
        self.player_context = critic.PlayerContext(dtype="float32")
        self.critic = critic.PaymentCritic(
            belief_offset=candidates.shape[-1],
            shanten_offset=(
                candidates.shape[-1] + critic.BELIEF_WIDTH + board.shape[-1] + 4
            ),
            player_width=self.player_context(
                [inputs, critic_rivers, selected]
            ).shape[-1],
        )

        predictions = candidate_predictions.predict(
            inputs=inputs,
            candidates=candidates,
            selected=selected,
            wait_logits=wait,
            eligibility_logits=eligibility,
        )
        outputs = {
            **predictions,
            **actor_layers.build_board_heads(inputs, context),
            **actor_layers.build_policy_heads(
                inputs=inputs,
                context=context,
                candidates=tf.keras.ops.stop_gradient(candidates),
                selected=selected,
                predictions=predictions,
            ),
        }
        outputs["constraint_loss"] = tf.keras.layers.Add(dtype="float32")(
            [
                encoded["belief_constraint_loss"],
                outputs["route_constraint_loss"],
                encoded["round_completion_yaku_logits_constraint_loss"],
                encoded["round_completion_score_logits_constraint_loss"],
            ]
        )
        # Auxiliary heads have visible-route and hindsight targets.
        # Only predictions enter the actor; labels stay in the trainer.
        self.actor = actor_layers.CallPolicy(
            tf.keras.Model(inputs, {**outputs, **encoded}, autocast=False),
            self.context,
            self.beliefs,
        )
        # Both paths reuse the same local/accepted-call encodings. Unused
        # final policy readouts have no path to the critic objective.
        self.encoder = self.actor

    def compile_training(self):
        """Call after initialization so tracing binds live Keras variables."""
        self.critic.distributions = tf.function(
            self.critic.distributions, jit_compile=True
        )
        self.critic.joint_distributions = tf.function(
            self.critic.joint_distributions, jit_compile=True
        )
        self.critic.chunk_statistics = tf.function(
            self.critic.chunk_statistics, jit_compile=True
        )
        self.critic.chunk_gradients = tf.function(
            self.critic.chunk_gradients, jit_compile=True
        )
        self.critic.candidate_chunk = tf.function(
            self.critic.candidate_chunk, jit_compile=True
        )
        self.critic.supervised_statistics = tf.function(
            self.critic.supervised_statistics
        )
        for block in self.actor.policy.get_layer("hand_bank").suits:
            block.call = tf.function(block.call, jit_compile=True)
        river = self.actor.policy.get_layer("river_encoder")
        river.encode = tf.function(river.encode, jit_compile=True)
        self.beliefs.tile_attention.call = tf.function(
            self.beliefs.tile_attention.call, jit_compile=True
        )

    def candidate_context(self, features, encoded):
        return tf.concat(
            [
                encoded["candidate_context"],
                self.public(
                    [features, encoded["selected"], encoded["structure"]]
                ),
                tf.sigmoid(tf.stop_gradient(encoded["route_affinity_logits"])),
                encoded["candidate_completion"],
                tf.reshape(
                    self.player_context(
                        [
                            features,
                            encoded["critic_rivers"],
                            encoded["selected"],
                        ]
                    ),
                    (-1, 4 * self.critic.player_width),
                ),
            ],
            axis=-1,
        )

    def call(self, inputs):
        outputs = self.actor(inputs[0])
        # Reuse the actor encoder output for critic supervision.
        state = self.candidate_context(inputs[0], outputs)
        return {
            **outputs,
            **self.supervised_outputs(inputs, outputs, state),
        }

    def chosen_inputs(self, features, encoded, state, chosen):
        """One observed action per row, with the shared transition contract."""
        recorded = (
            tf.gather(chosen, encoded["selected"][:, 0])
            == encoded["selected"][:, 1]
        )
        selected = tf.boolean_mask(encoded["selected"], recorded)
        return {
            **joint_settlement.public_inputs(features),
            "state": tf.boolean_mask(state, recorded),
            **critic.public_inputs(features, selected),
        }

    def supervised_outputs(self, inputs, encoded, state):
        features, labels, _ = inputs
        recorded_inputs = self.chosen_inputs(
            features, encoded, state, critic.chosen_actions(features, labels)
        )
        if not self.critic.built:
            self.critic(
                {name: value[:1] for name, value in recorded_inputs.items()}
            )
        return {
            "critic_statistics": self.critic.supervised_statistics(
                recorded_inputs, inputs[2]
            )
        }
