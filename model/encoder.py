"""Shared board, hand and river representations."""

import math

import tensorflow as tf

from log_dataset import _native_collection
from model import actions

HAND_BATCH_BUCKET = 512
# Scoring constraints bypass learned projections to preserve weights.
SCORING_INPUTS = frozenset(
    {
        "public_fu",
        "public_bonus",
        "candidate_public_fu",
        "candidate_public_bonus",
    }
)


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class BoardProjection(tf.keras.layers.Layer):
    """Shared score/yaku axes with distinct, ordered action slots."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        groups = {}
        offset = 1408
        for name, spec in _native_collection.INPUTS.items():
            if (
                name in SCORING_INPUTS
                or name.startswith("river_")
                or name
                in {
                    "candidate_hand_counts",
                    "candidate_hand_red",
                    "candidate_melds",
                    "candidate_hand_valid",
                }
            ):
                continue
            size = math.prod(spec["shape"])
            if name.endswith(("yaku_possibility",)):
                groups.setdefault(spec["shape"][-1], []).extend(
                    range(offset, offset + size)
                )
            offset += size
        selected = {index for indices in groups.values() for index in indices}
        self.retained = [
            index for index in range(offset) if index not in selected
        ]
        self.indices = list(groups.values())
        self.widths = list(groups)
        self.bases = [
            tf.keras.layers.Dense(
                max(4, width // 2), use_bias=False, dtype=self.dtype_policy
            )
            for width in self.widths
        ]
        self.project = tf.keras.layers.Dense(
            512, activation="gelu", dtype=self.dtype_policy
        )

    def build(self, input_shape):
        for width, basis in zip(self.widths, self.bases):
            basis.build((None, width))
        self.project.build(
            (
                None,
                len(self.retained)
                + sum(
                    len(indices) // width * basis.units
                    for width, indices, basis in zip(
                        self.widths, self.indices, self.bases
                    )
                ),
            )
        )
        super().build(input_shape)

    def call(self, values):
        encoded = [tf.gather(values, self.retained, axis=-1)]
        for width, indices, basis in zip(self.widths, self.indices, self.bases):
            encoded.append(
                tf.reshape(
                    basis(
                        tf.reshape(
                            tf.gather(values, indices, axis=-1),
                            (-1, len(indices) // width, width),
                        )
                    ),
                    (-1, len(indices) // width * basis.units),
                )
            )
        return self.project(tf.concat(encoded, -1))


def numeric_features(values):
    """Flatten numeric fields and compress magnitudes in float32."""
    return tf.keras.ops.arcsinh(
        tf.keras.ops.concatenate(
            [
                tf.keras.ops.reshape(
                    tf.keras.ops.cast(value, "float32"),
                    (-1, math.prod(value.shape[1:])),
                )
                for value in values
            ],
            axis=-1,
        )
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.board_fields = tuple(
            name
            for name in _native_collection.INPUTS
            if not name.startswith(
                ("candidate_", "discard_", "chi_", "pon_", "kan_")
            )
            and name not in ("river_pre_tile_id", "river_post_tile_id")
            and name not in SCORING_INPUTS
        )
        self.board_projection = tf.keras.layers.Dense(
            128, use_bias=False, dtype=self.dtype_policy
        )
        self.embedding = tf.keras.layers.Embedding(
            34, 16, dtype=self.dtype_policy
        )
        self.tiles = tf.keras.layers.Dense(
            128,
            activation=None,
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
        self.tiles.build((None, 34, 47))
        self.board_projection.build(
            (
                None,
                6 * 32 * 34
                + sum(
                    math.prod(_native_collection.INPUTS[name]["shape"])
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
        # GatherNd has no int8 GPU kernel. Cast before selecting so the
        # full candidate bank does not take a round trip through the host.
        counts = tf.gather_nd(
            tf.cast(inputs["candidate_hand_counts"], tf.int32), selected
        )
        red = tf.linalg.matmul(
            tf.gather_nd(
                tf.cast(inputs["candidate_hand_red"], self.compute_dtype),
                selected,
            ),
            tf.one_hot([4, 13, 22], 34, dtype=self.compute_dtype),
        )
        melds = tf.gather_nd(
            tf.cast(inputs["candidate_melds"], tf.int32), selected
        )
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
            # Board-conditioned suits can only share within one board.
            keys = tf.concat(
                [
                    keys,
                    tf.repeat(tf.cast(selected[:, 0], tf.int32), 3)[:, None],
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
                                            tf.cast(inputs[name], tf.float32),
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
        suits = tf.reshape(
            tf.cast(
                tf.gather(tf.cast(suits, tf.float32), inverse), suits.dtype
            ),
            (-1, 27, 128),
        )
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
    """Encode separately padded pre/post-riichi rivers with shared weights."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = tuple(
            name.removeprefix("river_pre_")
            for name in _native_collection.INPUTS
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
    pass continues after the discarder; our chii/pon continues after us.
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
        self.projection.build((None, input_shape[1][-1] + 260))
        super().build(input_shape)

    def call(self, values):
        inputs, bank, board = values
        batch = tf.shape(board)[0]
        indices = actions.candidate_hand_indices(inputs)
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
                    ],
                    axis=-1,
                )
            )
        structural = tf.concat(
            [
                discard_values,
                discard_values,
                discard_values,
                tf.zeros((batch, 1, 2)),
                *calls,
                tf.zeros((batch, 35, 2)),
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
        structural = tf.concat(
            [
                structural,
                tf.reshape(inputs["action_value_range"], (batch, 295, 8)),
                tf.cast(
                    tf.reshape(inputs["action_value_valid"], (batch, 295, 4)),
                    tf.float32,
                ),
                tf.cast(inputs["action_value_eligible"][..., None], tf.float32),
                tf.cast(inputs["action_open_yaku_restricted"], tf.float32),
            ],
            axis=-1,
        )
        masks = actions.legal_choices(inputs)
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


def encode_inputs():
    inputs = {
        name: tf.keras.Input(
            shape=definition["shape"], dtype=definition["dtype"], name=name
        )
        for name, definition in _native_collection.INPUTS.items()
    }
    bank = HandBank(name="hand_bank")(inputs)
    encoded = [
        bank[:, 0],
        tf.keras.layers.GlobalAveragePooling1D()(bank),
        RiverEncoding(name="river_encoder")(inputs),
        numeric_features(
            [
                # Ignore retired inputs, including values in cached records.
                (
                    tf.keras.ops.zeros_like(value)
                    if name in _native_collection.IGNORED_INPUTS
                    else value
                )
                for name, value in inputs.items()
                if name
                not in {
                    "candidate_hand_counts",
                    "candidate_hand_red",
                    "candidate_melds",
                    "candidate_hand_valid",
                }
                and not name.startswith("river_")
                and name not in SCORING_INPUTS
            ]
        ),
    ]
    hidden = BoardProjection(name="board_projection")(
        tf.keras.layers.Concatenate()(encoded)
    )
    for block in range(4):
        residual = tf.keras.layers.LayerNormalization(
            name=f"board_block_{block}_norm"
        )(hidden)
        residual = tf.keras.layers.Dense(
            1536, activation="gelu", name=f"board_block_{block}_expand"
        )(residual)
        residual = tf.keras.layers.Dense(
            512, name=f"board_block_{block}_contract"
        )(residual)
        hidden = tf.keras.layers.Add(name=f"board_block_{block}_add")(
            [hidden, residual]
        )
    hidden = tf.keras.layers.Dense(512, activation="gelu", name="board_state")(
        hidden
    )
    hidden = tf.keras.layers.LayerNormalization(
        center=False,
        scale=False,
        epsilon=1e-5,
        dtype="float32",
        name="board_state_normalized",
    )(hidden)
    candidates, selected, structure = DecisionCandidates(
        name="decision_candidates"
    )([inputs, bank, hidden])
    candidates = tf.keras.layers.LayerNormalization(
        center=False,
        scale=False,
        epsilon=1e-5,
        dtype="float32",
        name="candidate_state_normalized",
    )(candidates)
    return inputs, hidden, candidates, selected, structure
