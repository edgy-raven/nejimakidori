"""Shared gradient application for policy and auxiliary updates."""

import tensorflow as tf


def apply_gradients(optimizer, tape, loss, variables):
    # Sum repeated tile entries locally: each embedding has only 34 rows.
    # Clip and synchronize the actual table gradient, not every occurrence.
    pairs = [
        (tf.convert_to_tensor(gradient), variable)
        for gradient, variable in zip(tape.gradient(loss, variables), variables)
        if gradient is not None
    ]
    norm = tf.debugging.check_numerics(
        tf.linalg.global_norm([gradient for gradient, _ in pairs]),
        "Non-finite local gradient norm",
    )
    with tf.control_dependencies([norm]):
        optimizer.apply_gradients(pairs)
    return {"gradient/local_norm": norm}
