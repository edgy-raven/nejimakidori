"""Shared candidate representations and learned attack/defense fusion."""

import math

import tensorflow as tf

from model import feature_vector

HAND_BATCH_BUCKET = 512


def numeric_features(values):
    """Compress a fixed field list in one GPU operation."""
    return tf.math.asinh(
        tf.concat(
            [
                tf.reshape(
                    tf.cast(value, tf.float32),
                    (-1, math.prod(value.shape[1:])),
                )
                for value in values
            ],
            axis=-1,
        )
    )


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
    discard = tf.cast(inputs["discard_action_legal"], tf.bool)
    declaration = tf.cast(inputs["riichi_discard_legal_mask"], tf.bool)
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


def opening_discard_symmetry(logits, inputs, legal):
    """Pool opening symmetry classes, fading with opponent discards."""
    counts = inputs["candidate_hand_counts"][:, 0]
    suited = tf.reshape(counts[:, :27], (-1, 3, 9))
    non_dora = ~tf.reduce_any(
        tf.reshape(inputs["dora_multiplicity"][:, :27] > 0, (-1, 3, 9)),
        axis=-1,
    )
    terminals = tf.reshape(
        tf.stack(
            [
                non_dora
                & (suited[:, :, 0] == 1)
                & tf.reduce_all(suited[:, :, 1:5] == 0, -1),
                non_dora
                & (suited[:, :, 8] == 1)
                & tf.reduce_all(suited[:, :, 4:8] == 0, -1),
            ],
            axis=-1,
        ),
        (-1, 6),
    )
    floating = (
        tf.linalg.matmul(
            tf.cast(terminals, tf.float32),
            tf.one_hot([0, 8, 9, 17, 18, 26], 37),
        )
        > 0
    ) & legal
    symmetric = (
        non_dora
        & (suited[:, :, 0] == 1)
        & tf.reduce_all(suited == tf.reverse(suited, axis=[-1]), axis=-1)
        & tf.reduce_all(suited[:, :, 1:3] == 0, axis=-1)
        & ~tf.reduce_any(floating, axis=-1, keepdims=True)
    )
    winds = tf.constant([[0, 1, 2, 3]], dtype=tf.int8)
    groups = (
        tf.concat(
            [
                floating[:, None, :],
                tf.pad(
                    (counts[:, 31:34] == 1)
                    & (inputs["dora_multiplicity"][:, 31:34] == 0),
                    [[0, 0], [31, 3]],
                )[:, None, :],
                tf.pad(
                    (counts[:, 27:31] == 1)
                    & (inputs["dora_multiplicity"][:, 27:31] == 0)
                    & (winds != inputs["seat_wind"][:, 0, None])
                    & (winds != inputs["seat_wind"][:, 1, None])
                    & (winds != inputs["prevailing_wind"][:, None]),
                    [[0, 0], [27, 6]],
                )[:, None, :],
                symmetric[:, :, None]
                & tf.cast(
                    tf.one_hot([0, 9, 18], 37) + tf.one_hot([8, 17, 26], 37),
                    tf.bool,
                )[None],
            ],
            axis=1,
        )
        & legal[:, None, :]
    )
    strength = tf.maximum(
        0.0,
        1.0
        - tf.reduce_sum(
            tf.cast(inputs["opponent_discard_count"][:, 1:], tf.float32),
            axis=-1,
        )
        / 6.0,
    )
    # Log-space arithmetic preserves tiny probabilities and group mass.
    group_mean = tf.reduce_logsumexp(
        tf.where(groups, logits[:, None, :], -1e9), axis=-1
    ) - tf.math.log(
        tf.maximum(1.0, tf.reduce_sum(tf.cast(groups, tf.float32), axis=-1))
    )
    pooled = tf.reduce_sum(
        tf.where(groups, group_mean[:, :, None], 0.0), axis=1
    )
    mixed = tf.reduce_logsumexp(
        tf.stack(
            [
                logits + tf.math.log(1.0 - strength[:, None]),
                pooled + tf.math.log(strength[:, None]),
            ],
            axis=-1,
        ),
        axis=-1,
    )
    return tf.where(
        tf.reduce_any(groups, axis=1) & (strength[:, None] > 0), mixed, logits
    )


def candidate_hand_indices(inputs):
    """Map the shared 295 action slots to their post-action hand banks."""
    discard = inputs["discard_candidate_hand"]
    batch = tf.shape(discard)[0]
    return tf.concat(
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

    def __init__(self, board_context=False, **kwargs):
        super().__init__(**kwargs)
        self.board_context = board_context
        if board_context:
            self.board_fields = tuple(
                name
                for name in feature_vector.INPUTS
                if not name.startswith(
                    ("candidate_", "discard_", "chi_", "pon_", "kan_")
                )
                and name not in ("river_pre_tile_id", "river_post_tile_id")
            )
            self.board_projection = tf.keras.layers.Dense(
                128, use_bias=False, dtype=self.dtype_policy
            )
        self.embedding = tf.keras.layers.Embedding(
            34, 16, dtype=self.dtype_policy
        )
        self.tiles = tf.keras.layers.Dense(
            128,
            activation=None if board_context else "gelu",
            dtype=self.dtype_policy,
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
        self.tiles.build((None, 34, 47 if self.board_context else 28))
        if self.board_context:
            self.board_projection.build(
                (
                    None,
                    6 * 32 * 34
                    + sum(
                        math.prod(feature_vector.INPUTS[name]["shape"])
                        for name in self.board_fields
                    ),
                )
            )
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
            if self.board_context:
                # Board-conditioned suits can only share within one board.
                keys = tf.concat(
                    [
                        keys,
                        tf.repeat(tf.cast(selected[:, 0], tf.int32), 3)[
                            :, None
                        ],
                    ],
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
        if self.board_context:
            values = tf.concat(
                [
                    values,
                    tf.gather(
                        tf.cast(
                            tf.math.asinh(
                                tf.concat(
                                    [
                                        tf.cast(
                                            inputs[name][..., None],
                                            tf.float32,
                                        )
                                        for name in (
                                            "dora_multiplicity",
                                            "known_unavailable_counts",
                                        )
                                    ]
                                    + [
                                        tf.cast(
                                            inputs["blocker_counts"],
                                            tf.float32,
                                        ),
                                        *[
                                            tf.transpose(
                                                tf.cast(
                                                    inputs[name], tf.float32
                                                ),
                                                (0, 2, 1),
                                            )
                                            for name in (
                                                "genbutsu_to_seat",
                                                "passed_unchanged_to_seat",
                                                "sotogawa_earliest_turn_to_seat",
                                            )
                                        ],
                                    ],
                                    axis=-1,
                                )
                            ),
                            self.compute_dtype,
                        ),
                        selected[:, 0],
                    ),
                ],
                axis=-1,
            )
        # Keep convolution batch shapes stable as occupied hand counts vary.
        # Small inference requests use smaller buckets to limit padding cost.
        size = tf.shape(counts)[0]
        multiple = tf.where(size > 128, HAND_BATCH_BUCKET, 16)
        values = tf.pad(values, ((0, (-size) % multiple), (0, 0), (0, 0)))
        values = self.tiles(values)
        if self.board_context:
            # Sum of affine maps is one joint affine map before the first
            # nonlinearity. No hand representation is compressed beforehand.
            board = self.board_projection(
                tf.concat(
                    [
                        tf.reshape(
                            tf.one_hot(
                                tf.cast(
                                    tf.stack(
                                        [
                                            inputs["river_pre_tile_id"],
                                            inputs["river_post_tile_id"],
                                        ],
                                        axis=2,
                                    ),
                                    tf.int32,
                                ),
                                34,
                            )
                            * tf.cast(
                                tf.stack(
                                    [
                                        inputs["river_pre_valid"],
                                        inputs["river_post_valid"],
                                    ],
                                    axis=2,
                                )[..., None],
                                tf.float32,
                            ),
                            (-1, 6 * 32 * 34),
                        ),
                        numeric_features(
                            [inputs[name] for name in self.board_fields]
                        ),
                    ],
                    axis=-1,
                )
            )
            values = tf.nn.gelu(
                values
                + tf.pad(
                    tf.gather(board, selected[:, 0])[:, None],
                    ((0, (-size) % multiple), (0, 0), (0, 0)),
                )
            )
        suits = tf.gather(tf.reshape(values[:size, :27], (-1, 9, 128)), first)
        suit_size = tf.shape(suits)[0]
        # Coarser suit padding reduces compiled convolution shape variants.
        suit_multiple = tf.where(suit_size > 128, 4096, 16)
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

    def get_config(self):
        return {**super().get_config(), "board_context": self.board_context}


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class RiverEncoding(tf.keras.layers.Layer):
    """Encode separately padded pre/post-riichi rivers with shared weights."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = tuple(
            name.removeprefix("river_pre_")
            for name in feature_vector.INPUTS
            if name.startswith("river_pre_")
            and name not in ("river_pre_tile_id", "river_pre_valid")
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
        self.embedding.build(input_shape["river_pre_tile_id"])
        self.tokens.build((None, 32, 16 + len(self.fields)))
        for convolution, normalization in zip(
            self.convolutions, self.normalizations
        ):
            convolution.build((None, 32, 64))
            normalization.build((None, 32, 64))
        self.projection.build((None, 33 * 64))
        super().build(input_shape)

    def call(self, inputs):
        return self.encode(
            {
                name: value
                for name, value in inputs.items()
                if name.startswith(("river_pre_", "river_post_"))
            }
        )

    def encode(self, inputs):
        rivers = {
            name: tf.stack(
                [inputs["river_pre_" + name], inputs["river_post_" + name]],
                axis=2,
            )
            for name in ("tile_id", "valid", *self.fields)
        }
        mask = tf.reshape(
            tf.cast(rivers["valid"], self.compute_dtype), (-1, 32, 1)
        )
        metadata = tf.stack(
            [tf.cast(rivers[name], tf.float32) for name in self.fields], -1
        )
        values = tf.reshape(
            tf.concat(
                [
                    self.embedding(rivers["tile_id"]),
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
        return tf.reshape(encoded, (-1, 384))


def last_draw_context(inputs, selected):
    """Current last-tile flags and projected seats without further calls.

    Seat projections describe ordinary live-wall draws, excluding rinshan.
    A kan consumes one live-wall tile for its replacement draw. A response
    pass continues after the discarder; our chi/pon continues after us.
    """
    wall = tf.cast(
        tf.gather(inputs["live_wall_count"], selected[:, 0]), tf.int32
    )
    response = tf.gather(inputs["decision_phase"], selected[:, 0]) == 1
    source = tf.cast(
        tf.gather(inputs["trigger_source"], selected[:, 0]), tf.int32
    )
    remaining = wall - tf.cast(selected[:, 1] >= 261, tf.int32)
    return tf.concat(
        [
            tf.cast(
                tf.stack(
                    [
                        (wall == 0)
                        & (
                            tf.gather(
                                inputs["current_draw_valid"], selected[:, 0]
                            )
                            > 0
                        )
                        & (
                            tf.gather(inputs["rinshan_state"], selected[:, 0])
                            == 0
                        ),
                        (wall == 0) & response,
                    ],
                    axis=-1,
                ),
                tf.float32,
            ),
            tf.one_hot((tf.where(response, source, 0) + wall) % 4, 4)
            * tf.cast(wall > 0, tf.float32)[:, None],
            tf.one_hot(
                (
                    tf.where(response & (selected[:, 1] == 111), source, 0)
                    + remaining
                )
                % 4,
                4,
            )
            * tf.cast(remaining > 0, tf.float32)[:, None],
        ],
        axis=-1,
    )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class DecisionCandidates(tf.keras.layers.Layer):
    """295 choices: discard 37, riichi 74, response 149, kan 35."""

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
        self.projection.build((None, input_shape[1][-1] + 277))
        super().build(input_shape)

    def call(self, values):
        inputs, bank, board = values
        batch = tf.shape(board)[0]
        indices = candidate_hand_indices(inputs)
        cuts = tf.broadcast_to(tf.range(37)[None], (batch, 37))
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
        # Flags distinguish declaration, call kind and call option.
        flags = tf.constant(
            [[0, 0, 0]] * 37
            + [[1, declare, 0] for declare in range(2) for _ in range(37)]
            + [[2, 0, 0]]
            + [[3, 0, option] for option in range(3) for _ in range(37)]
            + [[4, 0, option] for option in range(1) for _ in range(37)]
            + [[6, 0, 0]]
            + [[5, 0, 0]] * 34,
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
                            last_draw_context(inputs, selected),
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
            (tf.shape(board)[0], 295, packed.shape[-1]),
        )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class ResponseKan(tf.keras.layers.Layer):
    def call(self, values):
        scores, inputs = values
        # A response kan refers to the triggering discard's base tile.
        return tf.concat(
            [
                scores[:, 111:260],
                tf.gather(
                    scores[:, 261:295],
                    tf.cast(inputs["trigger_tile_id"], tf.int32),
                    batch_dims=1,
                )[:, None],
            ],
            axis=-1,
        )
