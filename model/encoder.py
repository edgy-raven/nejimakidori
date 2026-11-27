"""Shared board, hand and river representations."""

import math

import tensorflow as tf

from log_dataset import _native_collection
from model import actions, input_features, river_attention

HAND_BATCH_BUCKET = 512
# Deterministic scoring constraints bypass learned projections.
SCORING_INPUTS = frozenset(
    {
        "public_bonus",
        "candidate_fu",
    }
)
CANDIDATE_SCORING_INPUTS = frozenset(
    {
        "action_win_value_range",
        "action_win_value_valid",
        "action_value_eligible",
        "action_open_yaku_restricted",
    }
)
TILE_CONTEXT_INPUTS = (
    "dora_multiplicity",
    "known_unavailable_counts",
    "blocker_counts",
    "genbutsu_to_seat",
    "passed_unchanged_to_seat",
    "sotogawa_earliest_turn_to_seat",
)
TABLE_INPUTS = {
    name: spec
    for name, spec in input_features.NEURAL_INPUTS.items()
    if name not in SCORING_INPUTS
    and name not in CANDIDATE_SCORING_INPUTS
    and not name.startswith("river_")
    and not name.startswith(("candidate_", "discard_", "call_", "kan_"))
    and name not in TILE_CONTEXT_INPUTS
    and name
    not in {
        "riichi_legal_mask",
        "riichi_discard_legal_mask",
        "response_legal_mask",
    }
}


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class TableContext(tf.keras.layers.Layer):
    """Encode shared fields and river summaries once before hand fusion."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        groups = {}
        offset = 384
        for name, spec in TABLE_INPUTS.items():
            size = math.prod(spec["shape"]) * input_features.field_width(name)
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
        self.project = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    256, use_bias=False, dtype=self.dtype_policy
                ),
                tf.keras.layers.Dense(
                    512, activation="gelu", dtype=self.dtype_policy
                ),
            ],
            name="table_readout",
        )
        self.project.dtype_policy = self.dtype_policy

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


def semantic_features(values):
    """Flatten already encoded field coordinates in schema order."""
    return tf.keras.ops.concatenate(
        [
            tf.keras.ops.reshape(value, (-1, math.prod(value.shape[1:])))
            for value in values.values()
        ],
        axis=-1,
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
        super().__init__(**{**kwargs, "autocast": False})
        self.context = tf.keras.layers.Dense(
            128, use_bias=False, dtype=self.dtype_policy
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
        self.projection = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    256, use_bias=False, dtype=self.dtype_policy
                ),
                tf.keras.layers.Dense(
                    512, activation="gelu", dtype=self.dtype_policy
                ),
            ],
            name="hand_readout",
        )
        self.projection.dtype_policy = self.dtype_policy
        self.normalization = tf.keras.layers.LayerNormalization(
            dtype=self.dtype_policy
        )

    def build(self, input_shape):
        self.tiles.build((None, 34, 31 + input_features.TILE_WIDTH))
        self.context.build(input_shape[1])
        for block in self.suits:
            block.build((None, 9, 128))
        self.honors.build((None, 7, 128))
        self.projection.build((None, 34 * 128 + 19))
        self.normalization.build((None, 512))
        super().build(input_shape)

    def call(self, values):
        inputs, context = values
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
                    tf.cast(inputs["tile_identity"], self.compute_dtype)[None],
                    (tf.shape(selected)[0], 34, input_features.TILE_WIDTH),
                ),
            ],
            axis=-1,
        )
        values = tf.concat(
            [
                values,
                tf.gather(
                    tf.cast(
                        tf.concat(
                            [
                                input_features.field_features(
                                    name, inputs[name], inputs
                                )
                                for name in (
                                    "dora_multiplicity",
                                    "known_unavailable_counts",
                                )
                            ]
                            + [
                                tf.math.asinh(
                                    tf.cast(
                                        inputs["blocker_counts"], tf.float32
                                    )
                                ),
                                *[
                                    tf.transpose(
                                        input_features.field_features(
                                            name, inputs[name], inputs
                                        )[..., 0],
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
        size = tf.shape(selected)[0]
        multiple = tf.where(size > 128, HAND_BATCH_BUCKET, 16)
        values = tf.pad(values, ((0, (-size) % multiple), (0, 0), (0, 0)))
        values = self.tiles(values)
        # Every hand reads the same observed context. Only hand zero feeds
        # the board; hypothetical hands are consumed by candidate scoring.
        values = tf.nn.gelu(
            values
            + tf.pad(
                tf.gather(self.context(context), selected[:, 0])[:, None],
                ((0, (-size) % multiple), (0, 0), (0, 0)),
            )
        )
        suits = tf.gather(tf.reshape(values[:size, :27], (-1, 9, 128)), first)
        suit_size = tf.shape(suits)[0]
        # Coarser suit padding reduces compiled convolution shape variants.
        suit_multiple = tf.where(
            suit_size <= 128,
            16,
            tf.where(
                suit_size <= 512, 512, tf.where(suit_size <= 1024, 1024, 4096)
            ),
        )
        suits = tf.pad(
            suits, ((0, (-suit_size) % suit_multiple), (0, 0), (0, 0))
        )
        for block in self.suits:
            suits = block(suits)
        # A leading batch axis makes Gather's gradient dense, so repeated
        # suits sum in float32 before the cast back to the compute dtype.
        suits = tf.reshape(
            tf.cast(
                tf.gather(tf.cast(suits, tf.float32)[None], inverse, axis=1)[0],
                suits.dtype,
            ),
            (-1, 27, 128),
        )
        suits = tf.pad(suits, ((0, (-size) % multiple), (0, 0), (0, 0)))
        values = tf.concat([suits, self.honors(values[:, 27:])], axis=1)
        values = self.normalization(
            self.projection(
                tf.concat(
                    [
                        tf.reshape(values, (-1, 34 * 128)),
                        tf.pad(
                            tf.gather_nd(
                                tf.cast(
                                    inputs["candidate_hand_yaku_possibility"],
                                    self.compute_dtype,
                                ),
                                selected,
                            ),
                            ((0, (-size) % multiple), (0, 0)),
                        ),
                    ],
                    axis=-1,
                )
            )
        )
        return tf.scatter_nd(
            selected,
            values[:size],
            tf.cast(
                (tf.shape(inputs["candidate_hand_valid"])[0], 16, 512), tf.int64
            ),
        )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class RiverEncoding(tf.keras.layers.Layer):
    """Encode separately padded pre/post-riichi rivers with shared weights."""

    def __init__(self, **kwargs):
        super().__init__(**{**kwargs, "autocast": False})
        self.event_attention = river_attention.RiverAttention(
            dtype=self.dtype_policy
        )
        self.fields = tuple(
            name.removeprefix("river_pre_")
            for name in input_features.NEURAL_INPUTS
            if name.startswith("river_pre_")
            and name not in ("river_pre_tile_id", "river_pre_valid")
        )
        self.metadata_features = input_features.SemanticFields(
            ["river_pre_" + name for name in self.fields],
            dtype=self.dtype_policy,
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
        self.event_attention.build((None, 32, 64))
        self.tokens.build(
            (
                None,
                32,
                input_features.TILE_WIDTH
                + sum(
                    input_features.field_width("river_pre_" + name)
                    for name in self.fields
                ),
            )
        )
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
                or name == "tile_identity"
            }
        )

    def encode(self, inputs):
        rivers = {
            name: tf.stack(
                [
                    inputs["river_pre_" + name],
                    (
                        tf.zeros_like(inputs["river_pre_riichi"])
                        if name == "riichi"
                        else inputs["river_post_" + name]
                    ),
                ],
                axis=2,
            )
            for name in ("tile_id", "valid", *self.fields)
        }
        mask = tf.reshape(
            tf.cast(rivers["valid"], self.compute_dtype), (-1, 32, 1)
        )
        metadata = tf.concat(
            list(
                self.metadata_features(
                    {
                        "river_pre_" + name: value
                        for name, value in rivers.items()
                    }
                ).values()
            ),
            -1,
        )
        values = tf.reshape(
            tf.concat(
                [
                    tf.cast(
                        input_features.tile_features(
                            rivers["tile_id"], inputs["tile_identity"]
                        ),
                        self.compute_dtype,
                    ),
                    tf.cast(metadata, self.compute_dtype),
                ],
                axis=-1,
            ),
            (
                -1,
                32,
                input_features.TILE_WIDTH
                + sum(
                    input_features.field_width("river_pre_" + name)
                    for name in self.fields
                ),
            ),
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
        values = self.event_attention(values, mask)
        # Shared summaries retain each opponent's pre/post-riichi history.
        # Empty rivers remain identical regardless of their storage slot.
        occupied = tf.reduce_any(mask > 0, axis=1)
        maximum = tf.where(
            occupied,
            tf.reduce_max(tf.where(mask > 0, values, -1e4), axis=1),
            tf.zeros_like(values[:, 0]),
        )
        local = tf.concat(
            [
                tf.math.divide_no_nan(
                    tf.reduce_sum(values, axis=1),
                    tf.reduce_sum(mask, axis=1),
                ),
                maximum,
            ],
            axis=-1,
        )
        encoded = self.projection(
            tf.concat([tf.reshape(values, (-1, 32 * 64)), maximum], axis=-1)
        ) * tf.cast(occupied, self.compute_dtype)
        return (
            tf.reshape(encoded, (-1, 384)),
            tf.reshape(local, (-1, 3, 256)),
        )


def projected_draws(inputs, selected):
    """Ordinary draw counts and last drawer if this action takes effect."""
    remaining = tf.cast(
        tf.gather(inputs["live_wall_count"], selected[:, 0]), tf.int32
    ) - tf.cast(selected[:, 1] >= 261, tf.int32)
    source = tf.where(
        (tf.gather(inputs["decision_phase"], selected[:, 0]) == 1)
        & (selected[:, 1] == 111),
        tf.cast(tf.gather(inputs["trigger_source"], selected[:, 0]), tf.int32),
        0,
    )
    return tf.stack(
        [
            tf.cast(
                (
                    remaining[:, None]
                    + 3
                    - (tf.range(4)[None] - source[:, None] - 1) % 4
                )
                // 4,
                tf.float32,
            ),
            tf.one_hot((source + remaining) % 4, 4)
            * tf.cast(remaining > 0, tf.float32)[:, None],
        ],
        axis=-1,
    )


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
            projected_draws(inputs, selected)[..., 1],
        ],
        axis=-1,
    )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class DecisionCandidates(tf.keras.layers.Layer):
    """Local discard/riichi, pass and kan candidates in joint action IDs."""

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
        self.projection.build(
            (None, input_shape[1][-1] + 241 + input_features.TILE_WIDTH)
        )
        super().build(input_shape)

    def call(self, values):
        inputs, bank, board = values
        batch = tf.shape(board)[0]
        local_ids = tf.constant(list(range(112)) + list(range(260, 295)))
        indices = tf.gather(
            actions.candidate_hand_indices(inputs), local_ids, axis=1
        )
        cuts = tf.broadcast_to(tf.range(37)[None], (batch, 37))
        tiles = tf.concat(
            [
                cuts,
                cuts,
                cuts,
                tf.fill((batch, 1), 37),
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
        structural = tf.pad(
            tf.tile(discard_values, [1, 3, 1]), [[0, 0], [0, 36], [0, 0]]
        )
        # Local action kind and declaration flag; no call-discard slots.
        flags = tf.constant(
            [[0, 0]] * 37
            + [[1, declare] for declare in range(2) for _ in range(37)]
            + [[2, 0]]
            + [[6, 0]]
            + [[5, 0]] * 34,
            dtype=tf.float32,
        )
        scoring_rows = tf.constant(list(range(37)) + list(range(110)))
        local_open = (inputs["candidate_meld_tiles"] < 34) & (
            inputs["candidate_meld_types"] != 3
        )
        opened = tf.gather(
            tf.reduce_any(local_open, axis=-1),
            tf.cast(indices, tf.int32),
            batch_dims=1,
        )
        open_axis = tf.one_hot(tf.cast(opened, tf.int32), 2)
        structural = tf.concat(
            [
                structural,
                tf.reshape(
                    tf.gather(
                        inputs["action_win_value_range"], scoring_rows, axis=1
                    )[:, :, None]
                    * open_axis[..., None, None],
                    (batch, 147, 8),
                ),
                tf.cast(
                    tf.reshape(
                        tf.gather(
                            inputs["action_win_value_valid"],
                            scoring_rows,
                            axis=1,
                        )[:, :, None]
                        * tf.cast(open_axis[..., None], tf.int8),
                        (batch, 147, 4),
                    ),
                    tf.float32,
                ),
                tf.cast(
                    tf.gather(
                        inputs["action_value_eligible"], scoring_rows, axis=1
                    )[..., None],
                    tf.float32,
                ),
                tf.cast(
                    tf.gather(
                        inputs["action_open_yaku_restricted"],
                        scoring_rows,
                        axis=1,
                    ),
                    tf.float32,
                ),
            ],
            axis=-1,
        )
        # Action analysis belongs to each alternative, not observed-state
        # inference. Calls read these fields from their accepted child state.
        discard_analysis = tf.math.asinh(
            tf.concat(
                [
                    tf.cast(inputs[name], tf.float32)[..., None]
                    for name in (
                        "discard_action_normal_shanten",
                        "discard_action_normal_ukeire_count",
                    )
                ]
                + [tf.cast(inputs["discard_retained_genbutsu"], tf.float32)],
                axis=-1,
            )
        )
        action_analysis = tf.concat(
            [
                tf.pad(
                    tf.tile(discard_analysis, [1, 3, 1]),
                    [[0, 0], [0, 36], [0, 0]],
                ),
                tf.pad(inputs["kan_efficiency"], [[0, 0], [113, 0], [0, 0]]),
                tf.cast(
                    tf.concat(
                        [
                            inputs["discard_abort"][:, 0],
                            tf.reshape(inputs["discard_abort"], (batch, 74)),
                            tf.zeros((batch, 36), tf.int8),
                        ],
                        1,
                    ),
                    tf.float32,
                )[..., None],
            ],
            -1,
        )
        masks = actions.legal_choices(inputs)
        local_selected = tf.cast(
            tf.where(
                tf.concat(
                    [
                        masks["discard_policy_logits"],
                        masks["riichi_policy_logits"],
                        masks["response_policy_logits"][:, :1],
                        masks["kan_action_logits"],
                    ],
                    axis=1,
                )
            ),
            tf.int32,
        )
        selected = tf.stack(
            [local_selected[:, 0], tf.gather(local_ids, local_selected[:, 1])],
            axis=1,
        )
        hands = tf.stack(
            [
                selected[:, 0],
                tf.cast(tf.gather_nd(indices, local_selected), tf.int32),
            ],
            axis=1,
        )
        return (
            self.projection(
                tf.concat(
                    [
                        tf.cast(value, self.compute_dtype)
                        for value in [
                            tf.gather_nd(bank, hands),
                            tf.math.asinh(
                                tf.cast(
                                    tf.gather_nd(
                                        inputs["candidate_yaku_shanten"], hands
                                    ),
                                    tf.float32,
                                )
                            ),
                            tf.gather_nd(action_analysis, local_selected),
                            input_features.tile_features(
                                tf.gather_nd(tiles, local_selected),
                                inputs["tile_identity"],
                            ),
                            tf.cast(
                                (tf.gather_nd(tiles, local_selected) >= 34)
                                & (tf.gather_nd(tiles, local_selected) < 37),
                                tf.float32,
                            )[:, None],
                            tf.math.asinh(
                                tf.gather_nd(
                                    structural[..., :10], local_selected
                                )
                            ),
                            tf.gather_nd(structural[..., 10:], local_selected),
                            tf.one_hot(
                                tf.cast(
                                    tf.gather(
                                        flags[:, 0], local_selected[:, 1]
                                    ),
                                    tf.int32,
                                ),
                                7,
                            ),
                            tf.gather(flags[:, 1:2], local_selected[:, 1]),
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
                    tf.gather_nd(structural, local_selected),
                    tf.cast(tf.gather_nd(tiles, local_selected), tf.float32)[
                        :, None
                    ],
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
    neural = input_features.NeuralFeatures(
        name="neural_features", dtype="float32"
    )(inputs)
    rivers, critic_rivers = RiverEncoding(
        name="river_encoder", dtype="mixed_bfloat16"
    )(neural)
    context = TableContext(name="table_context", dtype="mixed_bfloat16")(
        tf.keras.layers.Concatenate(dtype="mixed_bfloat16")(
            [
                rivers,
                semantic_features(
                    input_features.SemanticFields(
                        TABLE_INPUTS,
                        dtype="mixed_bfloat16",
                        name="table_features",
                    )(neural)
                ),
            ]
        )
    )
    bank = HandBank(name="hand_bank", dtype="mixed_bfloat16")([neural, context])
    hidden = tf.keras.layers.Dense(
        512, activation="gelu", name="board_fusion", dtype="mixed_bfloat16"
    )(
        tf.keras.layers.Concatenate(dtype="mixed_bfloat16")(
            [context, bank[:, 0]]
        )
    )
    for block in range(6):
        residual = tf.keras.layers.LayerNormalization(
            name=f"board_block_{block}_norm", dtype="mixed_bfloat16"
        )(hidden)
        residual = tf.keras.layers.Dense(
            768,
            activation="gelu",
            name=f"board_block_{block}_expand",
            dtype="mixed_bfloat16",
        )(residual)
        residual = tf.keras.layers.Dense(
            768,
            activation="gelu",
            name=f"board_block_{block}_hidden",
            dtype="mixed_bfloat16",
        )(residual)
        residual = tf.keras.layers.Dense(
            512, name=f"board_block_{block}_contract", dtype="mixed_bfloat16"
        )(residual)
        hidden = tf.keras.layers.Add(
            name=f"board_block_{block}_add", dtype="mixed_bfloat16"
        )([hidden, residual])
    hidden = tf.keras.layers.Dense(
        512, activation="gelu", name="board_state", dtype="mixed_bfloat16"
    )(hidden)
    hidden = tf.keras.layers.LayerNormalization(
        center=False,
        scale=False,
        epsilon=1e-5,
        dtype="float32",
        name="board_state_normalized",
    )(hidden)
    candidates, selected, structure = DecisionCandidates(
        name="decision_candidates", dtype="mixed_bfloat16"
    )([neural, bank, hidden])
    candidates = tf.keras.layers.LayerNormalization(
        center=False,
        scale=False,
        epsilon=1e-5,
        dtype="float32",
        name="candidate_state_normalized",
    )(candidates)
    return inputs, hidden, candidates, selected, structure, critic_rivers
