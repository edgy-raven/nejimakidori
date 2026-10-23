"""Assembly of the shared actor, beliefs and payment critic."""

import tensorflow as tf

from model import actions
from model import actor as actor_layers
from model import beliefs, critic, encoder, joint_settlement, payment_scoring


class JointPolicyPayment(tf.keras.Model):
    """Shared policy, beliefs, auxiliaries and payment critic."""

    def __init__(self):
        super().__init__(autocast=False)
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        inputs, board, candidates, selected, structure = encoder.encode_inputs()
        self.beliefs = beliefs.OpponentBeliefs(
            encoder=tf.keras.Model(
                inputs, board, name="belief_encoder", autocast=False
            ),
        )
        belief_logits = self.beliefs(inputs, hidden=board)
        belief_probability = beliefs.Probabilities(dtype="float32")(
            belief_logits
        )
        self.encoder = tf.keras.Model(
            inputs,
            dict(
                candidate=candidates,
                board=board,
                selected=selected,
                structure=structure,
                **belief_logits,
                belief_context=tf.keras.layers.Concatenate(dtype="float32")(
                    [
                        tf.keras.layers.Flatten(dtype="float32")(
                            belief_probability[name]
                        )
                        for name in beliefs.BELIEF_HEADS
                    ]
                ),
                belief_wait_probability=tf.keras.layers.Multiply(
                    dtype="float32"
                )(
                    [
                        belief_probability[beliefs.BELIEF_HEADS[0]][
                            ..., 0, None
                        ],
                        belief_probability[beliefs.BELIEF_HEADS[1]][..., 0, :],
                    ]
                ),
            ),
        )
        tf.keras.mixed_precision.set_global_policy("float32")
        self.context = critic.HandContext(dtype="float32")
        self.public = critic.PublicContext(dtype="float32")
        self.critic = critic.PaymentCritic()

        inputs = self.encoder.input
        encoded = self.encoder.output
        probability = beliefs.Probabilities(dtype="float32")(encoded)
        outputs, context = actor_layers.build_board_heads(
            inputs=inputs,
            hidden=encoded["board"],
            predictions={
                name: (encoded[name], probability[name])
                for name in beliefs.BELIEF_HEADS
            },
        )
        outputs.update(
            actor_layers.build_policy_heads(
                inputs=inputs,
                hidden=encoded["board"],
                candidates=encoded["candidate"],
                selected=encoded["selected"],
                context=context,
            )
        )
        outputs.update(
            actor_layers.PolicySummaries(dtype="float32")([outputs, inputs])
        )
        # Auxiliary heads have visible-route and hindsight targets.
        # Only predictions enter the actor; labels stay in the trainer.
        self.actor = tf.keras.Model(inputs, {**outputs, **encoded})
        # Rollout encoding and serving use the same yaku-conditioned critic.
        self.encoder = tf.keras.Model(
            inputs,
            {
                **encoded,
                "route_affinity_logits": outputs["route_affinity_logits"],
                "round_completion_yaku_logits": outputs[
                    "round_completion_yaku_logits"
                ],
                "round_completion_score_logits": outputs[
                    "round_completion_score_logits"
                ],
            },
        )

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
        self.critic.supervised_statistics = tf.function(
            self.critic.supervised_statistics
        )
        for block in self.encoder.get_layer("hand_bank").suits:
            block.call = tf.function(block.call, jit_compile=True)
        river = self.encoder.get_layer("river_encoder")
        river.encode = tf.function(river.encode, jit_compile=True)

    def candidate_context(self, features, encoded):
        state = tf.concat(
            [
                encoded["candidate"],
                tf.gather(encoded["belief_context"], encoded["selected"][:, 0]),
                tf.gather(encoded["board"], encoded["selected"][:, 0]),
                self.context(
                    [features, encoded["selected"], encoded["structure"]]
                ),
                self.public(
                    [features, encoded["selected"], encoded["structure"]]
                ),
            ],
            axis=-1,
        )
        tiles = tf.cast(encoded["structure"][:, -1], tf.int32)
        base = tf.where(
            tiles == 37, 0, tf.where(tiles < 34, tiles, 4 + 9 * (tiles - 34))
        )
        encoded["immediate_safe"] = tf.gather_nd(
            tf.transpose(
                actions.guaranteed_safe_tiles(features),
                [0, 2, 1],
            ),
            tf.stack([encoded["selected"][:, 0], base], axis=1),
        )
        encoded["immediate_wait"] = tf.gather_nd(
            tf.transpose(encoded["belief_wait_probability"], [0, 2, 1]),
            tf.stack([encoded["selected"][:, 0], base], axis=1),
        )
        return encoded, tf.concat(
            [
                state,
                tf.sigmoid(encoded["route_affinity_logits"]),
                tf.reshape(
                    tf.gather(
                        tf.sigmoid(encoded["round_completion_yaku_logits"]),
                        encoded["selected"][:, 0],
                    ),
                    (-1, 4 * 2 * 23),
                ),
                tf.reshape(
                    tf.gather(
                        tf.nn.softmax(encoded["round_completion_score_logits"]),
                        encoded["selected"][:, 0],
                    ),
                    (-1, 4 * 2 * len(payment_scoring.SCORES)),
                ),
            ],
            axis=-1,
        )

    def call(self, inputs):
        outputs = self.actor(inputs[0])
        # Reuse the actor encoder output for critic supervision.
        encoded, state = self.candidate_context(inputs[0], outputs)
        return {
            **outputs,
            **self.supervised_outputs(inputs, encoded, state),
            "critic_candidate_state": state,
            "immediate_safe": encoded["immediate_safe"],
            "immediate_wait": encoded["immediate_wait"],
        }

    def chosen_inputs(self, features, encoded, state, chosen):
        """One observed action per row, with the shared transition contract."""
        recorded = (
            tf.gather(chosen, encoded["selected"][:, 0])
            == encoded["selected"][:, 1]
        )
        tf.debugging.assert_equal(
            tf.boolean_mask(encoded["selected"][:, 0], recorded),
            tf.range(tf.shape(chosen)[0]),
        )
        selected = tf.boolean_mask(encoded["selected"], recorded)
        return {
            **joint_settlement.public_inputs(features),
            "state": tf.boolean_mask(state, recorded),
            "immediate_safe": tf.boolean_mask(
                encoded["immediate_safe"], recorded
            ),
            **critic.public_inputs(features, selected),
            "immediate_allowed": actions.immediate_decisions(
                features, selected
            ),
            "immediate_wait": tf.boolean_mask(
                encoded["immediate_wait"], recorded
            ),
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
        outputs = {
            "critic_statistics": self.critic.supervised_statistics(
                recorded_inputs, inputs[2]
            )
        }
        candidate_public = {
            **critic.public_inputs(features, encoded["selected"]),
            "immediate_allowed": actions.immediate_decisions(
                features, encoded["selected"]
            ),
            "honba": tf.cast(
                tf.gather(features["honba"], encoded["selected"][:, 0]),
                tf.int32,
            ),
        }
        outputs["immediate_rows"] = encoded["selected"][:, 0]
        immediate_outputs = self.critic.immediate(
            {
                **candidate_public,
                "state": state,
                "immediate_safe": encoded["immediate_safe"],
                "immediate_wait": encoded["immediate_wait"],
            }
        )
        outputs.update(
            critic.immediate_loss_terms(
                labels=labels,
                selected=encoded["selected"],
                structure=encoded["structure"],
                public=candidate_public,
                outputs=immediate_outputs,
            )
        )
        return outputs
