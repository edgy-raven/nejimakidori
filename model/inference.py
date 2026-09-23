"""Validated SavedModel inference shared by HTTP and local callers."""

import dataclasses
import hashlib
import pathlib
import time

import numpy
import tensorflow as tf

import feature_vector
from log_dataset import shanten

from . import completion_yaku, outcomes


@dataclasses.dataclass
class Inference:
    predictions: list
    latency_ms: float
    metadata: dict
    outputs: dict

    def as_dict(self):
        return {
            "predictions": self.predictions,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class Predictor:
    max_batch_size = 256

    def __init__(self, saved_model):
        self.model_path = pathlib.Path(saved_model)
        self.module = tf.saved_model.load(str(saved_model))
        self.function = self.module.signatures["serving_default"]
        self.input_specs = {
            name: (tuple(value["shape"]), value["dtype"])
            for name, value in feature_vector.INPUTS.items()
        }
        self.output_specs = self.function.structured_outputs
        if set(self.output_specs) != set(feature_vector.ACTOR_OUTPUTS) or any(
            tuple(self.output_specs[name].shape[1:]) != tuple(shape)
            for name, shape in feature_vector.ACTOR_OUTPUTS.items()
        ):
            raise ValueError("SavedModel output contract requires retraining")
        digest = hashlib.sha256()
        for path in sorted(self.model_path.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(self.model_path)).encode())
                with path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
        self.metadata = {
            "revision": digest.hexdigest(),
            "model_contract": "typed_payments",
            "payment_types": list(outcomes.PAYMENT_TYPES),
            "payment_values": outcomes.PAYMENT_VALUES.tolist(),
            "tile_names": list(shanten.TILE_NAMES),
            "completion_yaku_names": list(completion_yaku.NAMES),
            "opponent_tile_target": "structural_ukeire_by_shanten",
            "opponent_shanten_classes": [0, 1, 2, "3+"],
            "opponent_wait_target": "joint_tenpai_and_structural_wait",
        }

    def tensors(self, instances):
        if not isinstance(instances, list) or not 1 <= len(instances) <= 256:
            raise ValueError("instances must contain 1 to 256 feature objects")
        for instance in instances:
            if (
                not isinstance(instance, dict)
                or instance.keys() != self.input_specs.keys()
            ):
                raise ValueError("feature keys do not match model contract")
            for name, (shape, dtype) in self.input_specs.items():
                if numpy.asarray(instance[name]).shape != shape:
                    raise ValueError(
                        f"incorrect shape for {name}: expected {shape}"
                    )
        return {
            name: tf.convert_to_tensor(
                [row[name] for row in instances], dtype=dtype
            )
            for name, (_, dtype) in self.input_specs.items()
        }

    def predict(self, **features):
        return self.function(**features)

    def infer(self, instances):
        start = time.perf_counter()
        outputs = self.predict(**self.tensors(instances))
        probabilities = {}
        for name, logits in outputs.items():
            if name in {
                "opponent_wait_logits",
                "opponent_ukeire_logits",
                "opponent_ukeire_by_shanten_logits",
                "round_completion_yaku_logits",
                "opponent_tenpai_logits",
            }:
                values = tf.math.sigmoid(logits)
            else:
                values = tf.nn.softmax(logits, -1)
            probabilities[name.replace("logits", "probabilities")] = (
                values.numpy()
            )
        payment_probabilities = probabilities["terminal_payment_probabilities"]
        payment_means = outcomes.expected_payments(payment_probabilities)
        payment_variances = numpy.maximum(
            0,
            payment_probabilities @ (outcomes.PAYMENT_VALUES**2)
            - payment_means**2,
        )
        hand_counts = probabilities[
            "opponent_hand_count_probabilities"
        ] @ numpy.arange(5)
        predictions = []
        for index in range(len(instances)):
            row = {
                name: values[index].tolist()
                for name, values in probabilities.items()
            }
            mean = payment_means[index].sum(axis=0)
            variance = payment_variances[index].sum(axis=0)
            row["structured_outcome"] = {
                "format": "typed_payments",
                "types": list(outcomes.PAYMENT_TYPES),
                "probabilities": row["terminal_payment_probabilities"],
                "values": outcomes.PAYMENT_VALUES.tolist(),
                "payment_mean_by_type": payment_means[index].tolist(),
                "payment_stddev_by_type": numpy.sqrt(
                    payment_variances[index]
                ).tolist(),
                "payment_mean": mean.tolist(),
                "payment_stddev": numpy.sqrt(variance).tolist(),
                "settlement_mean": numpy.append(
                    mean.sum(0) - mean.sum(1) + mean.diagonal(), -mean.trace()
                ).tolist(),
                "settlement_stddev": numpy.sqrt(
                    numpy.append(
                        variance.sum(0) + variance.sum(1) - variance.diagonal(),
                        variance.trace(),
                    )
                ).tolist(),
                "uncertainty": "independent payment types and cells",
            }
            row["opponent_hand_expected_counts"] = hand_counts[index].tolist()
            predictions.append(row)
        return Inference(
            predictions,
            (time.perf_counter() - start) * 1000,
            self.metadata,
            outputs,
        )
