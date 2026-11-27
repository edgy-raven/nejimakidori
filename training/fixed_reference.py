"""Score immutable policies with their own frozen replay feature producer."""

import hashlib
import importlib.util
import itertools
import json
import pathlib

import numpy
import tensorflow as tf

from model import actor


class Policy:
    """Batch exact frozen features at inclusive MJAI event positions."""

    def __init__(self, directory, batch_size=64):
        directory = pathlib.Path(directory).resolve(strict=True)
        source = directory.parent / "source/log_dataset/_native_collection.so"
        spec = importlib.util.spec_from_file_location(
            "fixed_reference._native_collection", source
        )
        self.native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.native)
        self.module = tf.saved_model.load(str(directory))
        self.policy = self.module.signatures["policy"]
        self.batch_size = batch_size
        self.fields = {
            name: self.native.INPUTS[name]
            for name in self.policy.structured_input_signature[1]
        }
        self.provenance = {
            "files_sha256": {
                str(path.relative_to(directory.parent)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in [
                    source,
                    *sorted(
                        path for path in directory.rglob("*") if path.is_file()
                    ),
                ]
            },
        }

    def predict(self, events, requests):
        """Return five policy heads in the original request order.

        Each request is (last applied event index, actor, decision phase).
        Replay stays inside the selected artifact's native implementation;
        compact learner features never substitute for frozen model inputs.
        """
        return self._predict_many([(events, requests)])

    def _predict_many(self, items):
        items = [
            (events, numpy.asarray(requests, dtype=numpy.int64).reshape(-1, 3))
            for events, requests in items
        ]
        output = {
            name: numpy.empty(
                (sum(len(requests) for _, requests in items), *spec.shape[1:]),
                spec.dtype.as_numpy_dtype,
            )
            for name, spec in self.policy.structured_outputs.items()
        }

        def rows():
            offset = 0
            for events, requests in items:
                replay = self.native.FeatureReplay()
                cursor = 0
                for position in numpy.argsort(requests[:, 0], kind="stable"):
                    event, seat, phase = requests[position]
                    replay.apply(json.dumps(events[cursor : event + 1]))
                    cursor = int(event) + 1
                    yield offset + position, replay.pack(int(seat), int(phase))
                offset += len(requests)

        for batch in itertools.batched(rows(), self.batch_size):
            selected = numpy.asarray([position for position, _ in batch])
            # Pad only the final cross-game batch to the frozen graph's shape.
            padded = numpy.minimum(
                numpy.arange(self.batch_size), len(batch) - 1
            )
            scores = self.policy(
                **{
                    name: tf.convert_to_tensor(
                        numpy.stack(
                            [
                                numpy.frombuffer(buffers[name], spec["dtype"])
                                .reshape(spec["shape"])
                                .copy()
                                for _, buffers in batch
                            ]
                        )[padded]
                    )
                    for name, spec in self.fields.items()
                }
            )
            for name, values in scores.items():
                output[name][selected] = values.numpy()[: len(batch)]
        return output

    def distribution(self, events, requests, heads, legal):
        """Return masked, selected-head log probabilities for learner rows."""
        return actor.distribution(
            self.predict(events, requests), heads, legal
        ).numpy()

    def distribution_many(self, items):
        """Score (events, requests, heads, legal) items in concatenated order."""
        items = list(items)
        return actor.distribution(
            self._predict_many(
                (events, requests) for events, requests, _, _ in items
            ),
            numpy.concatenate([heads for _, _, heads, _ in items]),
            numpy.concatenate([legal for _, _, _, legal in items]),
        ).numpy()
