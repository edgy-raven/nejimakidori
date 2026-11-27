"""Sparse accepted-call states alongside ordinary decision feature rows."""

import numpy

from log_dataset import _native_collection

ROW_DTYPE = numpy.dtype(
    [
        (name, spec["dtype"], tuple(spec["shape"]))
        for name, spec in _native_collection.INPUTS.items()
    ]
)
CALL_DTYPE = numpy.dtype([("option", "int8"), ("features", ROW_DTYPE)])
INPUTS = {
    **_native_collection.INPUTS,
    **{
        "post_call_" + name: {**spec, "shape": [None, *spec["shape"]]}
        for name, spec in _native_collection.INPUTS.items()
    },
    "post_call_owner": {"shape": [None], "dtype": "int32"},
    "post_call_option": {"shape": [None], "dtype": "int32"},
}


def input_specs():
    import tensorflow as tf

    return {
        name: tf.TensorSpec(
            (
                spec["shape"]
                if name.startswith("post_call_")
                else [None, *spec["shape"]]
            ),
            spec["dtype"],
            name=name,
        )
        for name, spec in INPUTS.items()
    }


def base(features):
    return {name: features[name] for name in _native_collection.INPUTS}


def post(features):
    return {
        name: features["post_call_" + name]
        for name in _native_collection.INPUTS
    }


def combine(features, calls, owner, option):
    return {
        **features,
        **{"post_call_" + name: value for name, value in calls.items()},
        "post_call_owner": owner,
        "post_call_option": option,
    }


def from_request(request):
    rows = numpy.frombuffer(request["features"], ROW_DTYPE)
    calls = numpy.frombuffer(request["post_call_features"], ROW_DTYPE)
    return combine(
        {name: rows[name].copy() for name in ROW_DTYPE.names},
        {name: calls[name].copy() for name in ROW_DTYPE.names},
        numpy.asarray(request["post_call_owner"], dtype="int32"),
        numpy.frombuffer(request["post_call_option"], dtype="uint8").astype(
            "int32"
        ),
    )


def batch(instances):
    return concatenate(
        [
            combine(
                {name: value[None] for name, value in base(row).items()},
                post(row),
                row["post_call_owner"],
                row["post_call_option"],
            )
            for row in instances
        ]
    )


def concatenate(batches):
    if not batches:
        rows = numpy.empty(0, ROW_DTYPE)
        return combine(
            {name: rows[name] for name in ROW_DTYPE.names},
            {name: rows[name] for name in ROW_DTYPE.names},
            numpy.empty(0, dtype="int32"),
            numpy.empty(0, dtype="int32"),
        )
    offsets = numpy.cumsum(
        [0, *[len(next(iter(base(row).values()))) for row in batches[:-1]]],
        dtype="int32",
    )
    return combine(
        {
            name: numpy.concatenate([row[name] for row in batches])
            for name in _native_collection.INPUTS
        },
        {
            name: numpy.concatenate(
                [row["post_call_" + name] for row in batches]
            )
            for name in _native_collection.INPUTS
        },
        numpy.concatenate(
            [
                row["post_call_owner"] + offset
                for row, offset in zip(batches, offsets)
            ]
        ),
        numpy.concatenate([row["post_call_option"] for row in batches]),
    )


def select(features, indices):
    indices = numpy.asarray(indices, dtype="int64")
    owners = numpy.asarray(features["post_call_owner"])
    starts = numpy.searchsorted(owners, indices, side="left")
    ends = numpy.searchsorted(owners, indices, side="right")
    selected = numpy.concatenate(
        [
            numpy.empty(0, dtype="int64"),
            *[numpy.arange(start, end) for start, end in zip(starts, ends)],
        ]
    )
    return combine(
        {name: value[indices] for name, value in base(features).items()},
        {name: value[selected] for name, value in post(features).items()},
        numpy.repeat(numpy.arange(len(indices), dtype="int32"), ends - starts),
        features["post_call_option"][selected],
    )


def pack_numpy(features, packing):
    return {
        "base": packing.pack_numpy(base(features)),
        "post": packing.pack_numpy(post(features)),
        "owner": features["post_call_owner"],
        "option": features["post_call_option"],
    }


def pack(features, packing):
    return {
        "base": packing.pack(base(features)),
        "post": packing.pack(post(features)),
        "owner": features["post_call_owner"],
        "option": features["post_call_option"],
    }


def unpack(packed, packing):
    return combine(
        packing.unpack(packed["base"]),
        packing.unpack(packed["post"]),
        packed["owner"],
        packed["option"],
    )


def serialize(features):
    calls = numpy.empty(len(features["post_call_owner"]), CALL_DTYPE)
    calls["option"] = features["post_call_option"]
    for name, value in post(features).items():
        calls["features"][name] = value
    return calls.tobytes()


def decode(serialized):
    import tensorflow as tf

    lengths = tf.strings.length(serialized, unit="BYTE")
    counts = lengths // CALL_DTYPE.itemsize
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                counts * CALL_DTYPE.itemsize,
                lengths,
                message="Post-call record width mismatch; rebuild from source logs",
            )
        ]
    ):
        raw = tf.reshape(
            tf.io.decode_raw(tf.strings.reduce_join(serialized), tf.int8),
            [-1, CALL_DTYPE.itemsize],
        )
    calls = {}
    for name, (dtype, offset) in ROW_DTYPE.fields.items():
        spec = _native_collection.INPUTS[name]
        bytes_ = raw[:, 1 + offset : 1 + offset + dtype.itemsize]
        scalar = tf.as_dtype(spec["dtype"])
        if scalar.size > 1:
            bytes_ = tf.bitcast(
                tf.reshape(
                    bytes_, [-1, dtype.itemsize // scalar.size, scalar.size]
                ),
                scalar,
            )
        calls["post_call_" + name] = tf.reshape(bytes_, [-1, *spec["shape"]])
    return {
        **calls,
        "post_call_owner": tf.repeat(tf.range(tf.size(counts)), counts),
        "post_call_option": tf.cast(raw[:, 0], tf.int32),
    }


def packed_select(packed, indices):
    indices = numpy.asarray(indices, dtype="int64")
    starts = numpy.searchsorted(packed["owner"], indices, side="left")
    ends = numpy.searchsorted(packed["owner"], indices, side="right")
    selected = numpy.concatenate(
        [
            numpy.empty(0, dtype="int64"),
            *[numpy.arange(start, end) for start, end in zip(starts, ends)],
        ]
    )
    return {
        "base": {
            name: value[indices] for name, value in packed["base"].items()
        },
        "post": {
            name: value[selected] for name, value in packed["post"].items()
        },
        "owner": numpy.repeat(
            numpy.arange(len(indices), dtype="int32"), ends - starts
        ),
        "option": packed["option"][selected],
    }


def packed_concatenate(batches):
    offsets = numpy.cumsum(
        [0, *[len(next(iter(row["base"].values()))) for row in batches[:-1]]],
        dtype="int32",
    )
    return {
        group: {
            name: numpy.concatenate([row[group][name] for row in batches])
            for name in batches[0][group]
        }
        for group in ("base", "post")
    } | {
        "owner": numpy.concatenate(
            [row["owner"] + offset for row, offset in zip(batches, offsets)]
        ),
        "option": numpy.concatenate([row["option"] for row in batches]),
    }


def tensor_packed_select(packed, indices):
    import tensorflow as tf

    groups = tf.RaggedTensor.from_value_rowids(
        tf.range(tf.size(packed["owner"])),
        tf.cast(packed["owner"], tf.int64),
        nrows=tf.cast(
            tf.shape(next(iter(packed["base"].values())))[0], tf.int64
        ),
    )
    selected = tf.gather(groups, indices)
    return {
        "base": {
            name: tf.gather(value, indices)
            for name, value in packed["base"].items()
        },
        "post": {
            name: tf.gather(value, selected.flat_values)
            for name, value in packed["post"].items()
        },
        "owner": tf.cast(selected.value_rowids(), tf.int32),
        "option": tf.gather(packed["option"], selected.flat_values),
    }


def flatten(packed):
    return {
        group + "/" + name: value
        for group in ("base", "post")
        for name, value in packed[group].items()
    } | {"owner": packed["owner"], "option": packed["option"]}


def unflatten(arrays):
    return {
        group: {
            name.removeprefix(group + "/"): value
            for name, value in arrays.items()
            if name.startswith(group + "/")
        }
        for group in ("base", "post")
    } | {"owner": arrays["owner"], "option": arrays["option"]}
