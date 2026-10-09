"""Export the joint actor, board heads and action payment critic."""

import argparse
import hashlib
import json
import pathlib
import shutil

import numpy
import tensorflow as tf
from log_dataset import critic_data, payment_bonus
from model import (
    critic,
    critic_utility,
    decision_layers,
    feature_vector,
    joint_settlement,
    model,
    outcomes,
    source_snapshot,
)


class ServingModel(tf.Module):
    def __init__(self, network):
        super().__init__()
        # Track weights without exporting Keras' training-only call signatures.
        self.network = self._no_dependency(network)
        self.parameters = [variable.value for variable in network.variables]
        outputs = dict(network.actor.output)
        for head in ("attack", "defense"):
            scores = decision_layers.UnpackCandidates()(
                [
                    network.actor.get_layer(head + "_decisions").output,
                    network.actor.get_layer("decision_candidates").output[1],
                    network.actor.get_layer("board_state_normalized").output,
                ]
            )[:, :, 0]
            for phase, begin, end in (("discard", 0, 37), ("riichi", 37, 111)):
                outputs[f"{head}_{phase}_logits"] = model.PolicyMask(
                    phase + "_policy_logits", dtype="float32"
                )([scores[:, begin:end], network.actor.input])
        self.actor = self._no_dependency(
            tf.keras.Model(network.actor.input, outputs)
        )

    @tf.function
    def policy(self, **features):
        outputs = self.network.actor(features, training=False)
        return {
            name: outputs[name]
            for name in (
                *feature_vector.TRAIN_POLICY_HEADS,
                "win_decision_logits",
            )
        }

    @tf.function
    def serve(self, **features):
        outputs = self.actor(features, training=False)
        encoded, state = self.network.candidate_context(features, outputs)
        active = critic_utility.active_candidates(features, encoded["selected"])
        selected = tf.boolean_mask(encoded["selected"], active)
        state = tf.boolean_mask(state, active)
        safe = tf.boolean_mask(encoded["immediate_safe"], active)
        allowed = critic.immediate_decisions(features, selected)
        public = joint_settlement.public_inputs(features)
        candidate_public = critic.public_inputs(features, selected)
        probability = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            element_shape=[
                None,
                len(payment_bonus.CELLS),
                len(outcomes.PAYMENT_VALUES) + 4,
            ],
        )
        ranks = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, element_shape=[None, 4, 4]
        )
        utility = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, element_shape=[None]
        )
        final_ranks = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, element_shape=[None, 4]
        )
        points = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, element_shape=[None, 4]
        )
        settlement = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, element_shape=[None, 2, 5]
        )
        eligibility = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, element_shape=[None, 3]
        )
        legal_ron = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, element_shape=[None, 3]
        )
        immediate = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, element_shape=[None, 3]
        )
        for begin in tf.range(0, tf.shape(selected)[0], 4):
            rows = selected[begin : begin + 4, 0]
            inputs = {
                "state": state[begin : begin + 4],
                "immediate_safe": safe[begin : begin + 4],
                "immediate_wait": tf.boolean_mask(
                    encoded["immediate_wait"], active
                )[begin : begin + 4],
                "immediate_allowed": allowed[begin : begin + 4],
                **{
                    name: tf.gather(value, rows)
                    for name, value in public.items()
                },
            }
            inputs.update(
                {
                    name: value[begin : begin + 4]
                    for name, value in candidate_public.items()
                }
            )
            forecast = self.network.critic.joint_distributions(inputs)
            probability = probability.write(
                begin // 4, forecast["payment_probability"]
            )
            ranks = ranks.write(
                begin // 4, tf.exp(forecast["round_placement_logits"])
            )
            payoff, final_probability = joint_settlement.continuation_forecast(
                scores=forecast["joint_scores"],
                ranks=forecast["joint_ranks"],
                support=forecast["joint_support"],
                dealer=inputs["dealer"],
                hand_number=inputs["hand_number"],
                pot=inputs["pot"],
            )
            utility = utility.write(
                begin // 4,
                joint_settlement.expectation(
                    forecast["joint_probability"], payoff
                ),
            )
            final_ranks = final_ranks.write(
                begin // 4,
                joint_settlement.expectation(
                    forecast["joint_probability"], final_probability
                ),
            )
            points = points.write(
                begin // 4,
                joint_settlement.expectation(
                    forecast["joint_probability"],
                    forecast["joint_scores"] - inputs["scores"][:, None, None],
                ),
            )
            settlement = settlement.write(
                begin // 4,
                tf.stack(joint_settlement.settlement_moments(forecast), axis=1),
            )
            eligibility = eligibility.write(
                begin // 4, forecast["conditional_ron_eligibility"]
            )
            legal_ron = legal_ron.write(
                begin // 4,
                tf.sigmoid(forecast["immediate_logits"])
                * tf.cast(allowed[begin : begin + 4, None], tf.float32),
            )
            immediate = immediate.write(
                begin // 4,
                forecast["immediate_ron_probability"]
                - forecast["immediate_probability"][:, 21, None],
            )
        probability = probability.concat()
        weights = tf.gather_nd(
            critic_utility.candidate_probabilities(features, outputs), selected
        )
        count = tf.shape(features["decision_phase"])[0]
        tf.debugging.assert_near(
            tf.math.unsorted_segment_sum(weights, selected[:, 0], count),
            tf.ones(count),
            atol=1e-5,
        )
        mixture = tf.math.unsorted_segment_sum(
            probability * weights[:, None, None], selected[:, 0], count
        )
        moments = tf.math.unsorted_segment_sum(
            settlement.concat() * weights[:, None, None], selected[:, 0], count
        )
        return {
            **{
                name: outputs[name]
                for name in feature_vector.ACTOR_OUTPUTS
                if name
                not in ("terminal_payment_logits", "round_placement_logits")
            },
            **{
                f"{head}_{phase}_logits": outputs[f"{head}_{phase}_logits"]
                for head in ("attack", "defense")
                for phase in ("discard", "riichi")
            },
            "round_placement_logits": tf.math.log(
                tf.maximum(
                    tf.math.unsorted_segment_sum(
                        ranks.concat() * weights[:, None, None],
                        selected[:, 0],
                        count,
                    ),
                    1e-30,
                )
            ),
            "payment_values": joint_settlement.payment_values(public),
            "final_placement_probabilities": tf.math.unsorted_segment_sum(
                final_ranks.concat() * weights[:, None], selected[:, 0], count
            ),
            "settlement_mean": moments[:, 0],
            "settlement_variance": tf.maximum(
                moments[:, 1] - moments[:, 0] ** 2, 0
            ),
            "critic_projected_st3": utility.concat(),
            "critic_expected_points": points.concat(),
            "terminal_payment_probabilities": tf.einsum(
                "ncv,ctpr->ntprv",
                mixture,
                tf.constant(
                    numpy.eye(64, dtype=numpy.float32)[
                        numpy.ravel_multi_index(
                            numpy.array(payment_bonus.CELLS).T, (4, 4, 4)
                        )
                    ].reshape(-1, 4, 4, 4)
                ),
            )
            + tf.cast(~tf.constant(outcomes.PAYMENT_CELLS), tf.float32)[
                None, ..., None
            ]
            * tf.one_hot(1, len(outcomes.PAYMENT_VALUES) + 4),
            "critic_action_indices": selected,
            "critic_payment_probabilities": probability,
            "critic_immediate_probabilities": immediate.concat(),
            "critic_legal_ron_probabilities": legal_ron.concat(),
            "critic_structural_wait_probabilities": tf.boolean_mask(
                encoded["immediate_wait"], active
            ),
            "critic_conditional_ron_eligibility": eligibility.concat(),
        }


def run(args):
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    inputs = critic_data.parse_batch(
        next(
            iter(
                critic_data.dataset(
                    root=args.data,
                    sampling="natural",
                    partition="train",
                    batch_size=8,
                    repeat=False,
                )
            )
        )
    )[0]
    network = critic.JointPolicyPayment()
    outputs = network.actor(inputs, training=False)
    encoded, state = network.candidate_context(inputs, outputs)
    network.critic.joint_distributions(
        {
            **{
                name: tf.gather(value, encoded["selected"][:1, 0])
                for name, value in joint_settlement.public_inputs(
                    inputs
                ).items()
            },
            "state": state[:1],
            "immediate_safe": encoded["immediate_safe"][:1],
            "immediate_wait": encoded["immediate_wait"][:1],
            "immediate_allowed": critic.immediate_decisions(
                inputs, encoded["selected"][:1]
            ),
            **critic.public_inputs(inputs, encoded["selected"][:1]),
        }
    )
    network.built = True
    network.load_weights(args.weights)
    network.compile_training()
    module = ServingModel(network)
    signature = module.serve.get_concrete_function(
        **{
            name: tf.TensorSpec(
                [None, *definition["shape"]], definition["dtype"], name=name
            )
            for name, definition in feature_vector.INPUTS.items()
        }
    )
    root = pathlib.Path(args.output)
    pending = root.with_name(root.name + ".pending")
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    tf.saved_model.save(
        module,
        str(pending / "saved_model"),
        signatures={
            "serving_default": signature,
            "policy": module.policy.get_concrete_function(
                **signature.structured_input_signature[1]
            ),
        },
    )
    loaded = tf.saved_model.load(str(pending / "saved_model"))
    actual = loaded.signatures["serving_default"](**inputs)
    expected = module.serve(**inputs)
    policy = loaded.signatures["policy"](**inputs)
    for name, value in policy.items():
        numpy.testing.assert_allclose(value, actual[name], rtol=1e-5, atol=1e-5)
    errors = {}
    for name in expected:
        numpy.testing.assert_allclose(
            actual[name], expected[name], rtol=1e-5, atol=1e-5
        )
        errors[name] = float(
            tf.reduce_max(tf.abs(actual[name] - expected[name]))
        )
    for name in (
        "terminal_payment_probabilities",
        "critic_payment_probabilities",
        "final_placement_probabilities",
    ):
        numpy.testing.assert_allclose(
            tf.reduce_sum(actual[name], -1), 1, rtol=1e-4, atol=1e-5
        )
    for value in actual.values():
        assert numpy.isfinite(value.numpy()).all()
    with pathlib.Path(args.weights).open("rb") as stream:
        weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    (pending / "export-verification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "weights": str(args.weights),
                "weight_sha256": weight_hash,
                "data": str(args.data),
                "output_max_errors": errors,
                "critic_candidates": int(
                    tf.shape(actual["critic_action_indices"])[0]
                ),
                "payment_forecast": "active_policy_mixture_honba_free",
                "policy_signature_matches_full_export": True,
            },
            indent=2,
        )
        + "\n"
    )

    for name in ("model", "log_dataset"):
        source_snapshot.freeze(
            pathlib.Path(__file__).parents[1] / name,
            pending / "source" / name,
            include_native=True,
        )
    pending.rename(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
