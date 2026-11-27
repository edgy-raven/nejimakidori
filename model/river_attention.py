"""Observed-event attention across opponent seats and riichi segments."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class RiverAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.position = tf.keras.layers.Embedding(
            64, 64, dtype=self.dtype_policy
        )
        self.seat = tf.keras.layers.Embedding(3, 64, dtype=self.dtype_policy)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16, dtype=self.dtype_policy
        )
        self.correction = tf.keras.layers.Dense(
            64,
            use_bias=False,
            kernel_initializer="zeros",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.position.build((192,))
        self.seat.build((192,))
        self.attention.build((None, 192, 64), (None, 192, 64))
        self.correction.build((None, 192, 64))
        super().build(input_shape)

    def call(self, values, mask):
        events = tf.reshape(values, (-1, 192, 64))
        observed = tf.reshape(mask > 0, (-1, 192))
        tokens = (
            events
            + self.position(tf.range(192) % 64)[None]
            + self.seat(tf.range(192) // 64)[None]
        )
        update = self.attention(
            tokens,
            tokens,
            attention_mask=observed[:, :, None] & observed[:, None, :],
        )
        return values + tf.reshape(self.correction(update), (-1, 32, 64)) * mask
