"""Necessary completion conditions from irreversible public melds."""

import tensorflow as tf

from model import decision_layers


def support(melds):
    """Twenty-three named-yaku masks; concealed future tiles stay unknown."""
    tiles = tf.range(34)
    sequence = melds[..., 0] > 0
    triplet = tf.reduce_any(melds[..., 1:5] > 0, axis=-1)
    used = sequence | triplet
    any_meld = tf.reduce_any(used, axis=-1)
    opened = tf.reduce_any(
        tf.gather(melds, [0, 1, 2, 4], axis=-1) > 0, axis=(-1, -2)
    )
    no_sequence = ~tf.reduce_any(sequence, axis=-1)
    terminal = (tiles >= 27) | (tiles % 9 == 0) | (tiles % 9 == 8)
    simple = ~tf.reduce_any(
        (triplet & terminal)
        | (sequence & ((tiles % 9 == 0) | (tiles % 9 == 6))),
        axis=-1,
    )
    outside = ~tf.reduce_any(
        (triplet & ~terminal)
        | (sequence & (tiles % 9 != 0) & (tiles % 9 != 6)),
        axis=-1,
    )
    honor = tf.reduce_any(used[..., 27:], axis=-1)
    suits = tf.reduce_sum(
        tf.cast(
            tf.reduce_any(
                tf.reshape(
                    used[..., :27], tf.concat([tf.shape(used)[:-1], [3, 9]], 0)
                ),
                axis=-1,
            ),
            tf.int32,
        ),
        axis=-1,
    )
    flush = suits <= 1
    free = tf.ones_like(opened)
    return tf.stack(
        [
            ~opened & ~tf.reduce_any(triplet, -1),  # Pinfu.
            simple,
            ~opened,
            free,
            free,
            free,
            free,
            free,
            tf.reduce_sum(tf.cast(triplet, tf.int32), -1) <= 1,
            tf.reduce_sum(tf.cast(triplet, tf.int32), -1) <= 1,
            outside,
            outside & no_sequence,
            no_sequence,
            tf.reduce_sum(tf.gather(melds, [0, 1, 2, 4], axis=-1), (-1, -2))
            <= 1,
            tf.reduce_sum(tf.cast(sequence, tf.int32), -1) <= 1,
            tf.reduce_sum(tf.cast(sequence, tf.int32), -1) <= 1,
            ~any_meld,
            free,
            flush,
            outside & ~honor,
            ~any_meld,
            flush & ~honor,
            free,
        ],
        axis=-1,
    )


class CompletionMask(tf.keras.layers.Layer):
    def call(self, values):
        logits, inputs = values
        starts = tf.reduce_min(
            tf.where(
                inputs["meld_tile_valid"] > 0, inputs["meld_tile_ids"], 34
            ),
            axis=-1,
        )
        melds = tf.reduce_sum(
            tf.one_hot(starts, 34)[..., None]
            * tf.one_hot(tf.cast(inputs["meld_type"], tf.int32) - 1, 5)[
                ..., None, :
            ]
            * tf.cast(inputs["meld_valid"], tf.float32)[..., None, None],
            axis=2,
        )
        return tf.where(support(melds)[:, :, None], logits, -1e9)


class RouteMask(tf.keras.layers.Layer):
    def call(self, values):
        logits, inputs, selected = values
        hands = tf.stack(
            [
                selected[:, 0],
                tf.cast(
                    tf.gather_nd(
                        decision_layers.candidate_hand_indices(inputs), selected
                    ),
                    tf.int32,
                ),
            ],
            axis=1,
        )
        allowed = support(tf.gather_nd(inputs["candidate_melds"], hands))
        return tf.where(tf.gather(allowed, [1, 18, 10], axis=-1), logits, -1e9)
