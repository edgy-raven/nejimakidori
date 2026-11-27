"""Strict SavedModel loading and batched inference."""

import dataclasses
import hashlib
import pathlib
import time

import numpy
import tensorflow as tf

from log_dataset import call_states, scoring, shanten
from model import actions, outcomes


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
            for name, value in call_states.INPUTS.items()
        }
        signature = self.function.structured_input_signature[1]
        if signature != call_states.input_specs():
            raise ValueError("SavedModel inputs do not match feature producer")
        self.output_specs = self.function.structured_outputs
        expected = {
            **actions.SERVING_ACTOR_OUTPUTS,
            "terminal_payment_probabilities": [
                4,
                4,
                4,
                len(outcomes.PAYMENT_VALUES) + 4,
            ],
            "payment_values": [len(outcomes.PAYMENT_VALUES) + 4],
            "final_placement_probabilities": [4],
            "review_outcome_probabilities": [7],
            "settlement_mean": [5],
            "settlement_variance": [5],
            "critic_expected_points": [4],
            "critic_action_indices": [2],
        }
        if self.output_specs.keys() != expected.keys():
            raise ValueError("SavedModel must include joint actor and critic")
        for name, shape in expected.items():
            dtype = tf.int32 if name == "critic_action_indices" else tf.float32
            if (
                self.output_specs[name].shape.as_list() != [None, *shape]
                or self.output_specs[name].dtype != dtype
            ):
                raise ValueError(f"Incorrect serving output contract: {name}")
        digest = hashlib.sha256()
        for path in sorted(self.model_path.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(self.model_path)).encode())
                with path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
        self.metadata = {
            "revision": digest.hexdigest(),
            "model_contract": "joint_policy_payment_critic",
            "payment_forecast": "active_policy_mixture_honba_free",
            "payment_events": "coexisting_ron_claims_tsumo_exhaustive_abort",
            "bank_payment": "accepted_deposits_and_priority_pot_award",
            "critic_action_ranges": {
                "discard": [0, 37],
                "riichi": [37, 111],
                "response_without_daiminkan": [111, 260],
                "kan": [260, 295],
            },
            "daiminkan_action": "261 + trigger_tile_id",
            "payment_types": list(outcomes.PAYMENT_TYPES),
            "tile_names": list(shanten.TILE_NAMES),
            "completion_yaku_names": list(scoring.NAMES),
            "completion_modes": ["ron", "tsumo"],
            "completion_score_states": list(outcomes.SCORE_STATES),
            "completion_score_seats": ["self"],
            "opponent_tile_target": "structural_ukeire_by_shanten",
            "opponent_shanten_classes": [0, 1, 2, "3+"],
            "opponent_wait_target": "joint_tenpai_and_structural_wait",
        }

    def tensors(self, instances):
        if (
            not isinstance(instances, list)
            or not 1 <= len(instances) <= self.max_batch_size
        ):
            raise ValueError(
                f"instances must contain 1 to {self.max_batch_size} "
                "feature objects"
            )
        rows = []
        for instance in instances:
            if (
                not isinstance(instance, dict)
                or instance.keys() != self.input_specs.keys()
            ):
                raise ValueError("feature keys do not match model contract")
            count = len(instance["post_call_option"])
            row = {}
            for name, (shape, dtype) in self.input_specs.items():
                expected = tuple(
                    count if size is None else size for size in shape
                )
                row[name] = numpy.asarray(instance[name], dtype=dtype)
                # Empty JSON arrays carry no trailing dimension information.
                if name.startswith("post_call_") and count == 0:
                    row[name] = row[name].reshape(expected)
                if row[name].shape != expected:
                    raise ValueError(
                        f"incorrect shape for {name}: expected {expected}"
                    )
            if (
                numpy.any(row["post_call_owner"] != 0)
                or numpy.any(
                    (row["post_call_option"] < 0)
                    | (row["post_call_option"] > 3)
                )
                or len(numpy.unique(row["post_call_option"])) != count
            ):
                raise ValueError("invalid per-instance call branches")
            if int(instance["decision_phase"]) not in (0, 1, 2, 3, 4):
                raise ValueError("Inference requires a discretionary decision")
            rows.append(row)
        return {
            name: tf.convert_to_tensor(value, dtype=self.input_specs[name][1])
            for name, value in call_states.batch(rows).items()
        }

    def predict(self, **features):
        return self.function(**features)

    def infer(self, instances):
        start = time.perf_counter()
        outputs = self.predict(**self.tensors(instances))
        probabilities = {}
        for name, logits in outputs.items():
            if not name.endswith("logits"):
                continue
            probabilities[name.replace("logits", "probabilities")] = (
                tf.nn.softmax(logits, -1).numpy()
            )
        probabilities["terminal_payment_probabilities"] = outputs[
            "terminal_payment_probabilities"
        ].numpy()
        probabilities["final_placement_probabilities"] = outputs[
            "final_placement_probabilities"
        ].numpy()
        probabilities["review_outcome_probabilities"] = outputs[
            "review_outcome_probabilities"
        ].numpy()
        payment_values = outputs["payment_values"].numpy()
        action_points = outputs["critic_expected_points"].numpy()
        selected = outputs["critic_action_indices"].numpy()
        settlement_mean = outputs["settlement_mean"].numpy()
        settlement_variance = outputs["settlement_variance"].numpy()
        predictions = []
        for index in range(len(instances)):
            row = {
                name: values[index].tolist()
                for name, values in probabilities.items()
            }
            row["action_values"] = [
                {
                    "action": int(selected[action, 1]),
                    "expected_points": action_points[action].tolist(),
                }
                for action in numpy.flatnonzero(selected[:, 0] == index)
            ]
            row["structured_outcome"] = {
                "format": "typed_payments",
                "conditioning": "active_policy_mixture",
                "honba_included": False,
                "types": list(outcomes.PAYMENT_TYPES),
                "probabilities": row["terminal_payment_probabilities"],
                "values": payment_values[index].tolist(),
                "settlement_mean": settlement_mean[index].tolist(),
                "settlement_stddev": numpy.sqrt(
                    settlement_variance[index]
                ).tolist(),
                "uncertainty": "joint hand settlement",
            }
            predictions.append(row)
        return Inference(
            predictions,
            (time.perf_counter() - start) * 1000,
            self.metadata,
            outputs,
        )
