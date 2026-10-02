"""Job: Export the joint actor, board heads and visible action payment critic."""

import argparse
import hashlib
import json
import pathlib

import numpy
import tensorflow as tf

import feature_vector
from model import critic, critic_utility, outcomes, payment_bonus


class ServingModel(tf.Module):
    def __init__(self, network):
        super().__init__()
        # Track weights without exporting Keras' training-only call signatures.
        self.network = self._no_dependency(network)
        self.parameters = [variable.value for variable in network.variables]

    @tf.function
    def serve(self, **features):
        outputs = self.network.actor(features, training=False)
        encoded, state = self.network.candidate_context(features, outputs)
        active = critic_utility.active_candidates(features, encoded["selected"])
        selected = tf.boolean_mask(encoded["selected"], active)
        state = tf.boolean_mask(state, active)
        safe = tf.boolean_mask(encoded["immediate_safe"], active)
        allowed = critic.discard_decisions(features, selected)
        public = critic.public_inputs(features)
        probability = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            element_shape=[
                None,
                len(payment_bonus.CELLS),
                len(outcomes.PAYMENT_VALUES),
            ],
        )
        immediate = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, element_shape=[None, 3]
        )
        for begin in tf.range(0, tf.shape(selected)[0], 16):
            rows = selected[begin : begin + 16, 0]
            forecast = self.network.critic.distributions(
                {
                    "state": state[begin : begin + 16],
                    "immediate_safe": safe[begin : begin + 16],
                    "immediate_allowed": allowed[begin : begin + 16],
                    **{
                        name: tf.gather(value, rows)
                        for name, value in public.items()
                    },
                }
            )
            probability = probability.write(
                begin // 16, forecast["payment_probability"]
            )
            immediate = immediate.write(
                begin // 16,
                tf.sigmoid(forecast["immediate_logits"])
                * tf.cast(allowed[begin : begin + 16, None], tf.float32),
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
        active_slots = tf.scatter_nd(
            selected,
            tf.ones(tf.shape(selected)[0], dtype=tf.bool),
            tf.shape(outputs["attack_decisions"])[:2],
        )
        return {
            **{
                f"{head}_policy_logits": tf.where(
                    active_slots, outputs[f"{head}_decisions"][:, :, 0], -1e9
                )
                for head in ("attack", "defense")
            },
            **{
                name: outputs[name]
                for name in feature_vector.ACTOR_OUTPUTS
                if name != "terminal_payment_logits"
            },
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
            * tf.one_hot(1, len(outcomes.PAYMENT_VALUES)),
            "critic_action_indices": selected,
            "critic_payment_probabilities": probability,
            "critic_immediate_probabilities": immediate.concat(),
        }


def run(args):
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    with numpy.load(args.example) as example:
        inputs = {
            name: tf.convert_to_tensor(example[name])
            for name in feature_vector.INPUTS
        }
    network = critic.JointPolicyPayment()
    outputs = network.actor(inputs, training=False)
    encoded, state = network.candidate_context(inputs, outputs)
    network.critic(
        {
            "state": state[:1],
            "immediate_safe": encoded["immediate_safe"][:1],
            "immediate_allowed": critic.discard_decisions(
                inputs, encoded["selected"][:1]
            ),
            **{
                name: tf.gather(value, encoded["selected"][:1, 0])
                for name, value in critic.public_inputs(inputs).items()
            },
        }
    )
    network.built = True
    network.load_weights(args.weights)
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
    root.mkdir(parents=True, exist_ok=False)
    tf.saved_model.save(
        module,
        str(root / "saved_model"),
        signatures={"serving_default": signature},
    )
    actual = tf.saved_model.load(str(root / "saved_model")).signatures[
        "serving_default"
    ](**inputs)
    expected = tf.saved_model.load(args.policy).signatures["serving_default"](
        **inputs
    )
    errors = {}
    for name in feature_vector.TRAIN_POLICY_HEADS:
        numpy.testing.assert_allclose(
            actual[name], expected[name], rtol=1e-5, atol=1e-5
        )
        errors[name] = float(
            tf.reduce_max(tf.abs(actual[name] - expected[name]))
        )
    for name in (
        "terminal_payment_probabilities",
        "critic_payment_probabilities",
    ):
        numpy.testing.assert_allclose(
            tf.reduce_sum(actual[name], -1), 1, rtol=1e-4, atol=1e-5
        )
    for value in actual.values():
        assert numpy.isfinite(value.numpy()).all()
    with pathlib.Path(args.weights).open("rb") as stream:
        weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    (root / "export-verification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "weights": str(args.weights),
                "weight_sha256": weight_hash,
                "evaluated_policy": str(args.policy),
                "policy_max_errors": errors,
                "critic_candidates": int(
                    tf.shape(actual["critic_action_indices"])[0]
                ),
                "payment_forecast": "active_policy_mixture_honba_free",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--example", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
