"""Expert records and policy labels for structural restoration."""

import tensorflow as tf

from model import actions


def policy_choices(features, labels):
    """Recover the active policy head, observed choice and legal mask."""
    choices = tf.stack(
        [
            tf.cast(labels[name], tf.int32)
            for name in actions.TRAIN_POLICY_HEADS
        ],
        axis=1,
    )
    tf.debugging.assert_equal(
        tf.reduce_sum(tf.cast(choices >= 0, tf.int32), axis=1), 1
    )
    head = tf.argmax(choices >= 0, axis=1, output_type=tf.int32)
    expert = tf.gather(choices, head, batch_dims=1)
    masks = actions.legal_choices(features)
    legal = tf.gather(
        tf.stack(
            [
                tf.pad(masks[name], [[0, 0], [0, 150 - masks[name].shape[-1]]])
                for name in actions.TRAIN_POLICY_HEADS
            ],
            axis=1,
        ),
        head,
        batch_dims=1,
    )
    tf.debugging.assert_equal(tf.gather(legal, expert, batch_dims=1), True)
    return head, expert, legal
