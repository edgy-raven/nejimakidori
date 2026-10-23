"""CPU packing and TensorFlow packing share one lossless wire format."""

import numpy
import tensorflow as tf
from log_dataset import training_data


def test_numpy_and_tensor_packing_preserve_bits_and_nested_shapes():
    inputs = (
        {
            "mask": numpy.array([[True, False, True], [False, True, False]]),
            "signed": numpy.array([[-128, -1, 127], [42, 0, -7]], "int8")[
                :, ::-1
            ],
            "other_signed": numpy.array([[[-1]], [[-128]]], "int8"),
            "aligned": numpy.array([[-128, -1, 0, 127], [1, 2, 3, 4]], "int8"),
            "short": numpy.array([[-32768, -1, 32767], [0, 7, -9]], "int16"),
        },
        {
            "integer": numpy.array([-2147483648, 2147483647], "int32"),
            "float": numpy.array([[0.0, -0.0], [numpy.nan, -4.0]], "float32"),
            "half": numpy.array([[0.0, -0.0], [1.25, -4.0]], "float16"),
        },
    )
    packer = training_data.PackedBatch(inputs)
    packed = packer.pack_numpy(*inputs)
    tensor_packed = packer.pack(*inputs)
    assert {name: value.shape for name, value in packed.items()} == {
        "bool": (2, 1),
        "int8": (2, 3),
        "int16": (2, 2),
        "int32": (2, 1),
        "float32": (2, 2),
        "float16": (2, 2),
    }
    for name, value in packed.items():
        assert value.dtype == tensor_packed[name].numpy().dtype
        assert value.tobytes() == tensor_packed[name].numpy().tobytes()
    restored = tf.function(packer.unpack)(packed)
    tf.nest.assert_same_structure(inputs, restored)
    for expected, actual in zip(
        tf.nest.flatten(inputs), tf.nest.flatten(restored)
    ):
        assert actual.numpy().dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.numpy().tobytes() == expected.tobytes()
