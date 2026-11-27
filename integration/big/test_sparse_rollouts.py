"""Sparse call ownership survives shuffled packed replay and replica slices."""

import numpy
import tensorflow as tf

from log_dataset import call_states
from training.corrections import rollouts


def test_sparse_packed_rollouts(tmp_path):
    packed = {
        "base": {"float32": numpy.arange(4, dtype="float32")[:, None]},
        "post": {"float32": numpy.array([[10], [11], [30]], "float32")},
        "owner": numpy.array([1, 1, 3], "int32"),
        "option": numpy.array([0, 2, 1], "int32"),
    }
    selected = call_states.packed_select(packed, [3, 0, 1, 3])
    numpy.testing.assert_array_equal(selected["owner"], [0, 2, 2, 3])
    numpy.testing.assert_array_equal(
        selected["post"]["float32"][:, 0], [30, 10, 11, 30]
    )
    tensor = call_states.tensor_packed_select(
        tf.nest.map_structure(tf.convert_to_tensor, packed), [3, 0, 1, 3]
    )
    for actual, expected in zip(
        tf.nest.flatten(tensor), tf.nest.flatten(selected)
    ):
        numpy.testing.assert_array_equal(actual, expected)
    empty = call_states.packed_select(packed, [])
    joined = call_states.packed_concatenate([selected, empty, packed])
    numpy.testing.assert_array_equal(joined["owner"], [0, 2, 2, 3, 5, 5, 7])
    rows = {
        "features": joined,
        "chosen": numpy.arange(8),
        "reference_log_probability": numpy.arange(
            8 * 150, dtype=numpy.float32
        ).reshape(8, 150),
    }
    selected_rows = rollouts.select(rows, [7, 0, 2])
    numpy.testing.assert_array_equal(
        selected_rows["features"]["owner"], [0, 1, 2, 2]
    )
    combined_rows = rollouts.concatenate([selected_rows, selected_rows])
    numpy.testing.assert_array_equal(
        combined_rows["reference_log_probability"],
        rows["reference_log_probability"][[7, 0, 2, 7, 0, 2]],
    )
    path = tmp_path / "rollouts.npz"
    numpy.savez_compressed(path, **call_states.flatten(joined))
    with numpy.load(path) as archive:
        restored = call_states.unflatten(dict(archive))
    for actual, expected in zip(
        tf.nest.flatten(restored), tf.nest.flatten(joined)
    ):
        numpy.testing.assert_array_equal(actual, expected)
