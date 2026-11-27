"""Lossless semantic expansion of compact model inputs."""

import tensorflow as tf

from log_dataset import _native_collection

TILE_WIDTH = 8

# Projection axes retain their semantic order independently of wire packing.
NEURAL_INPUTS = {}
for name, spec in _native_collection.INPUTS.items():
    if name == "candidate_meld_tiles":
        NEURAL_INPUTS["candidate_melds"] = {
            "shape": [16, 34, 6],
            "dtype": "int8",
        }
    elif name in ("candidate_meld_types", "candidate_meld_red"):
        continue
    elif name == "last_tedashi_tile_id":
        NEURAL_INPUTS[name] = spec
        NEURAL_INPUTS["last_tedashi_is_red"] = {"shape": [4], "dtype": "int8"}
    elif name == "meld_tiles":
        for field in ("meld_tile_ids", "meld_tile_red", "meld_tile_valid"):
            NEURAL_INPUTS[field] = {"shape": [4, 4, 4], "dtype": "int8"}
    else:
        NEURAL_INPUTS[name] = spec
    if name == "passed_unchanged_to_seat":
        NEURAL_INPUTS["blocker_counts"] = {"shape": [34, 5], "dtype": "int8"}


def candidate_melds(inputs):
    groups = tf.reduce_sum(
        tf.one_hot(tf.cast(inputs["candidate_meld_tiles"], tf.int32), 34)[
            ..., None
        ]
        * tf.one_hot(tf.cast(inputs["candidate_meld_types"], tf.int32), 5)[
            ..., None, :
        ],
        axis=-3,
    )
    red = tf.linalg.matmul(
        tf.cast(inputs["candidate_meld_red"], tf.float32),
        tf.one_hot([4, 13, 22], 34),
    )
    return tf.cast(tf.concat([groups, red[..., None]], -1), tf.int8)


def meld_tile_ids(inputs):
    codes = tf.cast(inputs["meld_tiles"], tf.int32)
    return tf.cast(
        tf.where(
            codes < 37, tf.where(codes < 34, codes, 4 + 9 * (codes - 34)), 0
        ),
        tf.int8,
    )


def meld_tile_red(inputs):
    return tf.cast(
        (inputs["meld_tiles"] >= 34) & (inputs["meld_tiles"] < 37), tf.int8
    )


def meld_tile_valid(inputs):
    return tf.cast(inputs["meld_tiles"] < 37, tf.int8)


def last_tedashi_tile(inputs):
    return tf.cast(
        tf.one_hot(tf.cast(inputs["last_tedashi_tile_id"], tf.int32), 37),
        tf.int8,
    )


def blocker_counts(inputs):
    outside = ((), (0,), (0, 1), (1, 2), (2, 6), (6, 7), (7, 8), (8,), ())
    return tf.cast(
        tf.einsum(
            "st,bsc->btc",
            tf.constant(
                [
                    [
                        float(
                            source < 27
                            and target // 9 == source // 9
                            and target % 9 in outside[source % 9]
                        )
                        for target in range(34)
                    ]
                    for source in range(34)
                ]
            ),
            tf.one_hot(
                4 - tf.cast(inputs["known_unavailable_counts"], tf.int32), 5
            ),
        ),
        tf.int8,
    )


def neural_features(inputs):
    return {
        **inputs,
        "candidate_melds": candidate_melds(inputs),
        "last_tedashi_is_red": tf.cast(
            (inputs["last_tedashi_tile_id"] >= 34)
            & (inputs["last_tedashi_tile_id"] < 37),
            tf.int8,
        ),
        "meld_tile_ids": meld_tile_ids(inputs),
        "meld_tile_red": meld_tile_red(inputs),
        "meld_tile_valid": meld_tile_valid(inputs),
        "blocker_counts": blocker_counts(inputs),
    }


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class NeuralFeatures(tf.keras.layers.Layer):
    """Expand compact fields and project one shared one-hot tile vocabulary."""

    def __init__(self, **kwargs):
        super().__init__(**{**kwargs, "autocast": False})
        self.tile_projection = tf.keras.layers.Dense(
            TILE_WIDTH, use_bias=False, dtype="float32", name="tile_projection"
        )

    def build(self, input_shape):
        self.tile_projection.build((None, 34))
        super().build(input_shape)

    def call(self, inputs):
        return {
            **neural_features(inputs),
            "tile_identity": self.tile_projection(tf.eye(34)),
        }


def tile_features(codes, identity):
    """Project base-34 one-hot identity; red codes share their five's row."""
    codes = tf.cast(codes, tf.int32)
    base = tf.where(codes < 34, codes, 4 + 9 * (codes - 34))
    return tf.linalg.matmul(
        tf.one_hot(tf.where(codes < 37, base, 34), 34, dtype=identity.dtype),
        identity,
    )


# Categorical coordinates have no meaningful numeric distance. Tile identity
# uses a shared learned vector for every base tile; red flags remain separate.
RIVER_CATEGORIES = {
    "origin": 6,  # Padding, dealer start, draw, hand, chii, pon.
    "caller": 6,  # No caller, pending, then four actor-relative seats.
    "call_resolution": 6,  # Padding, pending, no call, chii, pon, open kan.
}
CATEGORIES = {
    "prevailing_wind": 4,
    "dealer_relative_seat": 4,
    "seat_wind": 4,
    "win_kind": 3,
    "decision_phase": 5,
    "trigger_source": 4,
    "riichi_state": 3,
    "meld_type": 6,
    "meld_source": 5,
    **{
        f"river_{phase}_{field}": width
        for phase in ("pre", "post")
        for field, width in RIVER_CATEGORIES.items()
    },
}
TILE_FIELDS = frozenset(
    {
        "current_draw_tile_id",
        "trigger_tile_id",
        "meld_tile_ids",
        "last_tedashi_tile_id",
    }
)
BINARY_FIELDS = frozenset(
    {
        "candidate_hand_valid",
        "meld_tile_valid",
        "meld_tile_red",
        "current_draw_valid",
        "current_draw_is_red",
        "closed_hand",
        "candidate_hand_yaku_possibility",
        "open_opponent_yaku_possibility",
        "ippatsu_alive",
        "rinshan_state",
        "genbutsu_to_seat",
        "passed_unchanged_to_seat",
        "last_tedashi_is_red",
        "trigger_valid",
        "trigger_is_red",
        "riichi_legal_mask",
        "riichi_discard_legal_mask",
        "response_legal_mask",
        "call_discard_legal",
        "kan_legal_mask",
        "discard_abort",
        "discard_action_legal",
        "discard_tedashi_legal",
        "river_pre_riichi",
        *[
            f"river_{phase}_{field}"
            for phase in ("pre", "post")
            for field in ("is_red", "tsumogiri", "valid")
        ],
    }
)
SCALED_FIELDS = frozenset(
    {
        "self_efficiency",
        "call_efficiency",
        "kan_efficiency",
        "meld_call_age",
        "meld_upgrade_age",
        *[
            f"river_{phase}_{field}"
            for phase in ("pre", "post")
            for field in ("event_age", "own_gap", "live_wall")
        ],
    }
)


def field_width(name):
    if name in TILE_FIELDS:
        return TILE_WIDTH
    if name.startswith("river_") and name in CATEGORIES:
        return 2
    return CATEGORIES[name] if name in CATEGORIES else 1


def field_features(name, value, inputs):
    """Encode learned evidence; deterministic consumers retain native values."""
    ops = tf.keras.ops
    value = ops.cast(value, "float32")
    if name == "scores":
        encoded = value[..., None] / 100000
    elif name == "ranks":
        encoded = value[..., None] / 4
    elif name in CATEGORIES:
        encoded = ops.one_hot(ops.cast(value, "int32"), CATEGORIES[name])
    elif name in BINARY_FIELDS or name in SCALED_FIELDS:
        encoded = value[..., None]
    else:
        encoded = ops.arcsinh(value)[..., None]
    if name.startswith("river_"):
        encoded = encoded * ops.cast(
            inputs["river_" + name.split("_")[1] + "_valid"][..., None],
            "float32",
        )
    elif name in ("trigger_source", "trigger_is_red"):
        encoded = encoded * ops.cast(
            inputs["trigger_valid"][..., None], "float32"
        )
    elif name == "current_draw_is_red":
        encoded = encoded * ops.cast(
            inputs["current_draw_valid"][..., None], "float32"
        )
    elif name.startswith("meld_"):
        valid = inputs["meld_tile_valid"][..., 0]
        while len(valid.shape) < len(value.shape):
            valid = valid[..., None]
        encoded = encoded * ops.cast(valid[..., None], "float32")
    return encoded


class SemanticFields(tf.keras.layers.Layer):
    """Shared categorical embeddings, masks, and compressed numeric evidence."""

    def __init__(self, names, **kwargs):
        super().__init__(**{**kwargs, "autocast": False})
        self.names = tuple(names)
        self.river_categories = {
            field: tf.keras.layers.Embedding(width, 2, dtype=self.dtype_policy)
            for field, width in RIVER_CATEGORIES.items()
            if any(name.endswith("_" + field) for name in self.names)
        }

    def call(self, inputs):
        encoded = {}
        for name in self.names:
            if name in TILE_FIELDS:
                valid = (
                    inputs[name] < 37
                    if name == "last_tedashi_tile_id"
                    else inputs[
                        (
                            "meld_tile_valid"
                            if name == "meld_tile_ids"
                            else name.removesuffix("_tile_id") + "_valid"
                        )
                    ]
                )
                encoded[name] = tf.cast(
                    tile_features(inputs[name], inputs["tile_identity"]),
                    self.compute_dtype,
                ) * tf.cast(valid[..., None], self.compute_dtype)
            elif name.startswith("river_") and name in CATEGORIES:
                field = name.split("_", 2)[2]
                encoded[name] = self.river_categories[field](
                    tf.cast(inputs[name], tf.int32)
                ) * tf.cast(
                    inputs["river_" + name.split("_")[1] + "_valid"][..., None],
                    self.compute_dtype,
                )
            else:
                encoded[name] = tf.cast(
                    field_features(name, inputs[name], inputs),
                    self.compute_dtype,
                )
        return encoded
