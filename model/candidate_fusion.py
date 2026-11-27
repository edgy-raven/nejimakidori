"""Attention across the complete legal action set for each decision."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class CandidateFusion(tf.keras.layers.Layer):
    """Keep one output per legal action, independent of candidate ordering."""

    def __init__(self, width=128, **kwargs):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        self.width = width
        self.normalization = tf.keras.layers.LayerNormalization()
        self.query_key_value = tf.keras.layers.Dense(3 * width)

    def build(self, shape):
        self.correction = tf.keras.layers.Dense(
            shape[0][-1], kernel_initializer="zeros", bias_initializer="zeros"
        )
        self.normalization.build(shape[0])
        self.query_key_value.build(shape[0])
        self.correction.build((*shape[0][:-1], 3 * self.width))
        super().build(shape)

    def get_config(self):
        return {
            **super().get_config(),
            "width": self.width,
        }

    def call(self, values):
        candidates, owners, batch_size = values
        # CallPolicy supplies canonical order: each position's rows are
        # contiguous. Project only real candidates before padding attention.
        counts = tf.math.unsorted_segment_sum(
            tf.ones_like(owners), owners, batch_size
        )
        indices = tf.stack(
            [
                owners,
                tf.range(tf.shape(owners)[0])
                - tf.gather(tf.cumsum(counts, exclusive=True), owners),
            ],
            axis=-1,
        )
        capacity = tf.maximum(tf.reduce_max(counts), 1)
        valid = tf.sequence_mask(counts, capacity)
        query, key, value = tf.split(
            self.query_key_value(self.normalization(candidates)), 3, axis=-1
        )
        query = tf.nn.gelu(query)
        key = tf.nn.gelu(key)
        dense_query, dense_key, dense_value = tf.split(
            tf.scatter_nd(
                indices,
                tf.concat([query, key, value], axis=-1),
                [batch_size, capacity, 3 * self.width],
            ),
            3,
            axis=-1,
        )
        weights = tf.nn.softmax(
            tf.where(
                valid[:, None, :],
                tf.matmul(dense_query, dense_key, transpose_b=True)
                / tf.sqrt(tf.cast(self.width, tf.float32)),
                -1e9,
            ),
            axis=-1,
        )
        context = tf.gather_nd(tf.matmul(weights, dense_value), indices)
        return candidates + self.correction(
            tf.concat([query, key, context], axis=-1)
        )
