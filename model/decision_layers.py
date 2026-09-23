"""Shared candidate representations and learned attack/defense fusion."""

import tensorflow as tf

import feature_vector


@tf.custom_gradient
def gather_encodings(values, indices):
    """Reuse activations while accumulating repeated gradients in float32."""

    def gradient(upstream):
        return (
            tf.cast(
                tf.math.unsorted_segment_sum(
                    tf.cast(upstream, tf.float32), indices, tf.shape(values)[0]
                ),
                values.dtype,
            ),
            None,
        )

    return tf.gather(values, indices), gradient


def legal_choices(inputs):
    draw = tf.where(
        tf.cast(inputs["current_draw_is_red"], tf.bool),
        34 + tf.cast(inputs["current_draw_tile_id"], tf.int32) // 9,
        tf.cast(inputs["current_draw_tile_id"], tf.int32),
    )
    discard = tf.concat(
        [
            tf.cast(inputs["discard_tedashi_legal"], tf.bool),
            (
                tf.cast(inputs["current_draw_valid"], tf.bool)
                & tf.cast(
                    tf.gather(
                        inputs["discard_action_legal"], draw, batch_dims=1
                    ),
                    tf.bool,
                )
            )[:, None],
        ],
        axis=1,
    )
    declaration = tf.concat(
        [
            tf.cast(inputs["riichi_discard_legal_mask"], tf.bool),
            tf.cast(
                tf.gather(
                    inputs["riichi_discard_legal_mask"], draw, batch_dims=1
                ),
                tf.bool,
            )[:, None],
        ],
        axis=1,
    )
    return {
        "win_decision_logits": tf.stack(
            [
                tf.ones_like(inputs["win_kind"], dtype=tf.bool),
                inputs["win_kind"] > 0,
            ],
            axis=-1,
        ),
        "discard_policy_logits": discard,
        "riichi_policy_logits": tf.concat([discard, discard & declaration], 1),
        "response_policy_logits": tf.concat(
            [
                tf.cast(inputs["response_legal_mask"][:, :1], tf.bool),
                tf.reshape(
                    tf.cast(inputs["chi_discard_legal"], tf.bool), (-1, 111)
                ),
                tf.reshape(
                    tf.cast(inputs["pon_discard_legal"], tf.bool), (-1, 37)
                ),
                tf.cast(inputs["response_legal_mask"][:, 3:], tf.bool),
            ],
            axis=1,
        ),
        "kan_action_logits": tf.cast(inputs["kan_legal_mask"], tf.bool),
    }


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class SuitResidual(tf.keras.layers.Layer):
    def __init__(self, dilation, **kwargs):
        super().__init__(**kwargs)
        self.dilation = dilation
        self.first = tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            dilation_rate=dilation,
            dtype=self.dtype_policy,
            activation="gelu",
        )
        self.second = tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            dilation_rate=dilation,
            dtype=self.dtype_policy,
        )
        self.normalization = tf.keras.layers.LayerNormalization(
            dtype=self.dtype_policy
        )

    def build(self, input_shape):
        self.first.build(input_shape)
        self.second.build((*input_shape[:-1], 128))
        self.normalization.build((*input_shape[:-1], 128))
        super().build(input_shape)

    def call(self, values):
        return tf.nn.gelu(
            self.normalization(values + self.second(self.first(values)))
        )

    def get_config(self):
        return {**super().get_config(), "dilation": self.dilation}


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class HandBank(tf.keras.layers.Layer):
    """Encode occupied candidate hands with shared, suit-local convolutions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(
            34, 16, dtype=self.dtype_policy
        )
        self.tiles = tf.keras.layers.Dense(
            128, activation="gelu", dtype=self.dtype_policy
        )
        self.suits = [
            SuitResidual(dilation, dtype=self.dtype_policy)
            for dilation in (1, 2, 4)
        ]
        self.honors = tf.keras.layers.Dense(
            128, activation="gelu", dtype=self.dtype_policy
        )
        self.projection = tf.keras.layers.Dense(
            512, activation="gelu", dtype=self.dtype_policy
        )
        self.normalization = tf.keras.layers.LayerNormalization(
            dtype=self.dtype_policy
        )

    def build(self, input_shape):
        self.embedding.build((34,))
        self.tiles.build((None, 34, 28))
        for block in self.suits:
            block.build((None, 9, 128))
        self.honors.build((None, 7, 128))
        self.projection.build((None, 34 * 128))
        self.normalization.build((None, 512))
        super().build(input_shape)

    def call(self, inputs):
        selected = tf.where(inputs["candidate_hand_valid"] > 0)
        counts = tf.gather_nd(inputs["candidate_hand_counts"], selected)
        red = tf.linalg.matmul(
            tf.cast(
                tf.gather_nd(inputs["candidate_hand_red"], selected),
                self.compute_dtype,
            ),
            tf.one_hot([4, 13, 22], 34, dtype=self.compute_dtype),
        )
        melds = tf.gather_nd(inputs["candidate_melds"], selected)
        # Exact raw keys include suit identity: tile embeddings differ by suit.
        # Only the integer deduplication runs on CPU; encodings stay on device.
        with tf.device("/CPU:0"):
            keys = tf.reshape(
                tf.concat(
                    [
                        tf.cast(counts[:, :27, None], tf.int32),
                        tf.cast(red[:, :27, None], tf.int32),
                        tf.cast(melds[:, :27], tf.int32),
                    ],
                    axis=-1,
                ),
                (-1, 72),
            )
            keys = tf.concat(
                [keys, tf.tile(tf.range(3), [tf.shape(counts)[0]])[:, None]],
                axis=-1,
            )
            unique, inverse = tf.raw_ops.UniqueV2(x=keys, axis=[0])
            first = tf.math.unsorted_segment_min(
                tf.range(tf.shape(keys)[0]), inverse, tf.shape(unique)[0]
            )
        values = tf.concat(
            [
                tf.one_hot(
                    tf.cast(counts, tf.int32), 5, dtype=self.compute_dtype
                ),
                red[..., None],
                tf.cast(melds, self.compute_dtype),
                tf.broadcast_to(
                    self.embedding(tf.range(34))[None],
                    (tf.shape(counts)[0], 34, 16),
                ),
            ],
            axis=-1,
        )
        # Keep convolution batch shapes stable as occupied hand counts vary.
        # Small inference requests use smaller buckets to limit padding cost.
        size = tf.shape(counts)[0]
        multiple = tf.where(size > 128, 128, 16)
        values = tf.pad(values, ((0, (-size) % multiple), (0, 0), (0, 0)))
        values = self.tiles(values)
        suits = tf.gather(tf.reshape(values[:size, :27], (-1, 9, 128)), first)
        suit_size = tf.shape(suits)[0]
        suit_multiple = tf.where(suit_size > 128, 128, 16)
        suits = tf.pad(
            suits, ((0, (-suit_size) % suit_multiple), (0, 0), (0, 0))
        )
        for block in self.suits:
            suits = block(suits)
        suits = tf.reshape(gather_encodings(suits, inverse), (-1, 27, 128))
        suits = tf.pad(suits, ((0, (-size) % multiple), (0, 0), (0, 0)))
        values = tf.concat(
            [suits, self.honors(values[:, 27:])],
            axis=1,
        )
        values = self.normalization(
            self.projection(tf.reshape(values, (-1, 34 * 128)))
        )
        return tf.scatter_nd(
            selected,
            values[:size],
            tf.cast(
                (tf.shape(inputs["candidate_hand_valid"])[0], 96, 512), tf.int64
            ),
        )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class RiverEncoding(tf.keras.layers.Layer):
    """Shared temporal patterns, ordered positions and masked pattern maxima."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = tuple(
            name
            for name in feature_vector.INPUTS
            if name.startswith("river_")
            and name not in ("river_tile_id", "river_valid")
        )
        self.embedding = tf.keras.layers.Embedding(
            34, 16, dtype=self.dtype_policy
        )
        self.tokens = tf.keras.layers.Dense(
            64, activation="gelu", dtype=self.dtype_policy
        )
        self.convolutions = [
            tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                padding="same",
                dilation_rate=dilation,
                dtype=self.dtype_policy,
            )
            for dilation in (1, 2, 4)
        ]
        self.normalizations = [
            tf.keras.layers.LayerNormalization(dtype=self.dtype_policy)
            for _ in self.convolutions
        ]
        self.projection = tf.keras.layers.Dense(
            64, activation="gelu", dtype=self.dtype_policy
        )

    def build(self, input_shape):
        self.embedding.build(input_shape["river_tile_id"])
        self.tokens.build((None, 32, 16 + len(self.fields)))
        for convolution, normalization in zip(
            self.convolutions, self.normalizations
        ):
            convolution.build((None, 32, 64))
            normalization.build((None, 32, 64))
        self.projection.build((None, 33 * 64))
        super().build(input_shape)

    def call(self, inputs):
        mask = tf.reshape(
            tf.cast(inputs["river_valid"], self.compute_dtype), (-1, 32, 1)
        )
        metadata = tf.stack(
            [tf.cast(inputs[name], tf.float32) for name in self.fields], -1
        )
        values = tf.reshape(
            tf.concat(
                [
                    self.embedding(inputs["river_tile_id"]),
                    tf.cast(tf.math.asinh(metadata), self.compute_dtype),
                ],
                axis=-1,
            ),
            (-1, 32, 16 + len(self.fields)),
        )
        values = self.tokens(values) * mask
        # All positions are already observed. Mask every block so padding
        # cannot feed activations back into the observed river boundary.
        for convolution, normalization in zip(
            self.convolutions, self.normalizations
        ):
            values = (
                tf.nn.gelu(normalization(values + convolution(values))) * mask
            )
        occupied = tf.reduce_any(mask > 0, axis=1)
        maximum = tf.where(
            occupied,
            tf.reduce_max(tf.where(mask > 0, values, -1e4), axis=1),
            tf.zeros_like(values[:, 0]),
        )
        encoded = self.projection(
            tf.concat([tf.reshape(values, (-1, 32 * 64)), maximum], axis=-1)
        ) * tf.cast(occupied, self.compute_dtype)
        return tf.reshape(encoded, (-1, 256))


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class DecisionCandidates(tf.keras.layers.Layer):
    """298 choices: discard 38, riichi 76, response 149, kan 35."""

    def __init__(self, **kwargs):
        super().__init__(**{**kwargs, "autocast": False})
        self.context = tf.keras.layers.Dense(
            192, activation="gelu", dtype=self.dtype_policy
        )
        self.projection = tf.keras.layers.Dense(
            384, activation="gelu", dtype=self.dtype_policy
        )

    def build(self, input_shape):
        self.context.build(input_shape[2])
        self.projection.build((None, input_shape[1][-1] + 268))
        super().build(input_shape)

    def call(self, values):
        inputs, bank, board = values
        batch = tf.shape(board)[0]
        draw = tf.where(
            tf.cast(inputs["current_draw_is_red"], tf.bool),
            34 + tf.cast(inputs["current_draw_tile_id"], tf.int32) // 9,
            tf.cast(inputs["current_draw_tile_id"], tf.int32),
        )
        discard = tf.concat(
            [
                inputs["discard_candidate_hand"],
                tf.gather(inputs["discard_candidate_hand"], draw, batch_dims=1)[
                    :, None
                ],
            ],
            axis=1,
        )
        indices = tf.concat(
            [
                discard,
                discard,
                discard,
                tf.zeros((batch, 1), discard.dtype),
                tf.reshape(inputs["chi_candidate_hand"], (-1, 111)),
                tf.reshape(inputs["pon_candidate_hand"], (-1, 37)),
                tf.zeros((batch, 1), discard.dtype),
                inputs["kan_candidate_hand"],
            ],
            axis=1,
        )
        cuts = tf.concat(
            [tf.broadcast_to(tf.range(37)[None], (batch, 37)), draw[:, None]],
            axis=1,
        )
        tiles = tf.concat(
            [
                cuts,
                cuts,
                cuts,
                tf.fill((batch, 1), 37),
                tf.broadcast_to(tf.tile(tf.range(37), [4])[None], (batch, 148)),
                tf.fill((batch, 1), 37),
                tf.broadcast_to(tf.range(34)[None], (batch, 34)),
            ],
            axis=1,
        )
        discard_values = tf.concat(
            [
                tf.cast(
                    inputs["discard_action_all_shanten"][..., None], tf.float32
                ),
                tf.cast(
                    inputs["discard_action_all_ukeire_count"][..., None],
                    tf.float32,
                ),
                tf.reshape(inputs["discard_tenpai_values"], (-1, 37, 32)),
            ],
            axis=-1,
        )
        discard_values = tf.concat(
            [
                discard_values,
                tf.gather(discard_values, draw, batch_dims=1)[:, None],
            ],
            axis=1,
        )
        calls = []
        for kind, count in (("chi", 111), ("pon", 37)):
            calls.append(
                tf.concat(
                    [
                        tf.cast(
                            tf.reshape(
                                inputs[kind + "_discard_shanten"],
                                (-1, count, 1),
                            ),
                            tf.float32,
                        ),
                        tf.cast(
                            tf.reshape(
                                inputs[kind + "_discard_ukeire_count"],
                                (-1, count, 1),
                            ),
                            tf.float32,
                        ),
                        tf.reshape(
                            inputs[kind + "_call_tenpai_values"],
                            (-1, count, 16),
                        ),
                        tf.zeros((batch, count, 16)),
                    ],
                    axis=-1,
                )
            )
        structural = tf.concat(
            [
                discard_values,
                discard_values,
                discard_values,
                tf.zeros((batch, 1, 34)),
                *calls,
                tf.zeros((batch, 35, 34)),
            ],
            axis=1,
        )
        # Flags distinguish declaration, origin, call kind and call option.
        flags = tf.constant(
            [[0, 0, int(cut == 37), 0] for cut in range(38)]
            + [
                [1, declare, int(cut == 37), 0]
                for declare in range(2)
                for cut in range(38)
            ]
            + [[2, 0, 0, 0]]
            + [[3, 0, 0, option] for option in range(3) for _ in range(37)]
            + [[4, 0, 0, option] for option in range(1) for _ in range(37)]
            + [[6, 0, 0, 0]]
            + [[5, 0, 0, 0]] * 34,
            dtype=tf.float32,
        )
        masks = legal_choices(inputs)
        selected = tf.cast(
            tf.where(
                tf.concat(
                    [
                        masks["discard_policy_logits"],
                        masks["riichi_policy_logits"],
                        masks["response_policy_logits"][:, :149],
                        masks["kan_action_logits"],
                    ],
                    axis=1,
                )
            ),
            tf.int32,
        )
        return (
            self.projection(
                tf.concat(
                    [
                        tf.cast(value, self.compute_dtype)
                        for value in [
                            tf.gather_nd(
                                bank,
                                tf.stack(
                                    [
                                        selected[:, 0],
                                        tf.cast(
                                            tf.gather_nd(indices, selected),
                                            tf.int32,
                                        ),
                                    ],
                                    axis=1,
                                ),
                            ),
                            tf.one_hot(tf.gather_nd(tiles, selected), 38),
                            tf.math.asinh(tf.gather_nd(structural, selected)),
                            tf.gather(flags, selected[:, 1]),
                            tf.gather(self.context(board), selected[:, 0]),
                        ]
                    ],
                    axis=-1,
                )
            ),
            selected,
            tf.concat(
                [
                    tf.gather_nd(structural, selected),
                    tf.cast(tf.gather_nd(tiles, selected), tf.float32)[:, None],
                ],
                axis=-1,
            ),
        )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class FinalCandidates(tf.keras.layers.Layer):
    def call(self, values):
        candidates, context, selected = values
        return tf.concat(
            [
                candidates,
                tf.gather(context, selected[:, 0]),
            ],
            axis=-1,
        )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class UnpackCandidates(tf.keras.layers.Layer):
    def call(self, values):
        packed, selected, board = values
        return tf.scatter_nd(
            selected,
            packed,
            (tf.shape(board)[0], 298, packed.shape[-1]),
        )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class ResponseKan(tf.keras.layers.Layer):
    def call(self, values):
        scores, inputs = values
        # A response kan refers to the triggering discard's base tile.
        return tf.concat(
            [
                scores[:, 114:263],
                tf.gather(
                    scores[:, 264:298],
                    tf.cast(inputs["trigger_tile_id"], tf.int32),
                    batch_dims=1,
                )[:, None],
            ],
            axis=-1,
        )
