"""Action-conditioned settlements and policy values."""

import math

import numpy
import tensorflow as tf

from log_dataset import scoring
from model import (
    actions,
    beliefs,
    encoder,
    input_features,
    joint_settlement,
    losses,
    outcomes,
    payment_scoring,
)

STATISTICS_CHUNK = 128
BELIEF_SHAPES = tuple(
    actions.ACTOR_OUTPUTS[name] for name in beliefs.BELIEF_HEADS
)
BELIEF_WIDTH = sum(math.prod(shape) for shape in BELIEF_SHAPES)
ROUTE_WIDTH = 3
PREDICTED_TAIL_WIDTH = ROUTE_WIDTH + sum(
    math.prod(actions.ACTOR_OUTPUTS[name]) for name in beliefs.COMPLETION_HEADS
)


def meld_tile_counts(melds):
    """Expand chi starts, triplets and kans to their physical tile counts."""
    chi = melds[..., 0]
    return (
        chi
        + tf.roll(chi, 1, axis=-1)
        + tf.roll(chi, 2, axis=-1)
        + 3 * melds[..., 1]
        + 4 * tf.reduce_sum(melds[..., 2:5], axis=-1)
    )


def public_inputs(features, selected=None):
    """Guaranteed scoring constraints given an attempted action.

    Scores and the existing pot remain pre-action: settlement deposits account
    for a declaration only after its discard survives ron.
    """
    opened = tf.reduce_any(
        (input_features.meld_tile_valid(features)[..., 0] > 0)
        & (features["meld_type"] != 4),
        axis=-1,
    )
    public = {
        "opened": opened,
        "deposit_allowed": ~opened
        & (features["riichi_state"] != 2)
        & (
            tf.gather(
                features["scores"],
                tf.cast(features["seat_wind"], tf.int32),
                batch_dims=1,
            )
            >= 1000
        ),
        "deposit_required": features["riichi_state"] == 1,
        "riichi_required": features["riichi_state"] > 0,
        "tenpai_required": (features["riichi_state"] > 0)
        | (
            tf.reduce_sum(
                tf.cast(
                    input_features.meld_tile_valid(features)[..., 0] > 0,
                    tf.int32,
                ),
                -1,
            )
            == 4
        ),
        "action_tile": tf.fill(tf.shape(features["dealer_relative_seat"]), 37),
        "ending_draw_kind": tf.zeros_like(features["dealer_relative_seat"]),
        "ending_tenpai": tf.fill(tf.shape(features["riichi_state"]), -1),
        "ending_safe": tf.zeros_like(features["riichi_state"][:, 1:], tf.bool),
        "dealer": features["dealer_relative_seat"],
        "public_fu": payment_scoring.fixed_meld_fu(features),
        "public_bonus": tf.cast(features["public_bonus"], tf.int32),
        "yakuman_allowed": payment_scoring.support(
            payment_scoring.public_melds(features)
        )[..., -1],
    }
    if selected is None:
        return public
    public = {
        name: tf.gather(value, selected[:, 0]) for name, value in public.items()
    }
    response = tf.gather(features["decision_phase"], selected[:, 0]) == 1
    hands = tf.stack(
        [
            selected[:, 0],
            tf.cast(
                tf.gather_nd(
                    actions.candidate_hand_indices(features), selected
                ),
                tf.int32,
            ),
        ],
        axis=1,
    )
    # Float32 keeps both gathers and slices on device; these small integer
    # counts are exact. TensorFlow's int32 StridedSlice uses host memory.
    melds = tf.gather_nd(
        tf.cast(input_features.candidate_melds(features), tf.float32), hands
    )
    own_opened = tf.reduce_any(
        tf.gather(melds, [0, 1, 2, 4], axis=2) > 0, axis=(1, 2)
    )
    declaration = (selected[:, 1] >= 74) & (selected[:, 1] < 111)
    own = tf.stack(
        [
            tf.range(tf.shape(selected)[0]),
            tf.zeros(tf.shape(selected)[0], tf.int32),
        ],
        axis=1,
    )
    public["opened"] = tf.tensor_scatter_nd_update(
        public["opened"],
        own,
        tf.where(response, public["opened"][:, 0], own_opened),
    )
    candidate_bonus = tf.reduce_sum(
        meld_tile_counts(melds)
        * tf.gather(
            tf.cast(features["dora_multiplicity"], tf.float32), selected[:, 0]
        )
        + melds[..., 5],
        axis=1,
    )
    for name, values in (
        (
            "public_fu",
            tf.gather_nd(tf.cast(features["candidate_fu"], tf.float32), hands),
        ),
        ("public_bonus", candidate_bonus),
    ):
        public[name] = tf.tensor_scatter_nd_update(
            public[name],
            own,
            tf.where(response, public[name][:, 0], tf.cast(values, tf.int32)),
        )
    public["yakuman_allowed"] = tf.tensor_scatter_nd_update(
        public["yakuman_allowed"],
        own,
        tf.where(
            response,
            public["yakuman_allowed"][:, 0],
            payment_scoring.support(melds)[..., -1],
        ),
    )
    public["deposit_allowed"] &= ~public["opened"]
    for name in ("deposit_required", "riichi_required"):
        public[name] = tf.tensor_scatter_nd_update(
            public[name], own, public[name][:, 0] | declaration
        )
        # A competing ron can end the hand before a pending declaration
        # is accepted. Hypothetical call features remain in learned context.
        public[name] &= ~(
            response[:, None]
            & (tf.gather(features["riichi_state"], selected[:, 0]) == 1)
        )
    # Only an unconditional own discard can force a draw at this boundary.
    # Response passes still await opponents' claims, including last-tile ron.
    discard = actions.discard_decisions(features, selected) & ~response
    public["tenpai_required"] |= public["riichi_required"]
    public["tenpai_required"] = tf.tensor_scatter_nd_update(
        public["tenpai_required"],
        own,
        public["tenpai_required"][:, 0]
        | (~response & (tf.reduce_sum(melds[:, :, :5], (1, 2)) == 4)),
    )
    public["action_tile"] = tf.where(
        selected[:, 1] < 111,
        selected[:, 1] % 37,
        tf.where(
            (selected[:, 1] >= 112) & (selected[:, 1] < 260),
            (selected[:, 1] - 112) % 37,
            tf.where(selected[:, 1] >= 261, selected[:, 1] - 261, 37),
        ),
    )
    tile = tf.where(selected[:, 1] < 111, selected[:, 1] % 37, 0)
    abort = (
        tf.gather_nd(
            features["discard_abort"],
            tf.stack(
                [selected[:, 0], tf.cast(declaration, tf.int32), tile],
                axis=1,
            ),
        )
        > 0
    )
    public["ending_safe"] = tf.gather_nd(
        tf.transpose(actions.guaranteed_safe_tiles(features), [0, 2, 1]),
        tf.stack(
            [
                selected[:, 0],
                tf.where(tile < 34, tile, 4 + 9 * (tile - 34)),
            ],
            axis=1,
        ),
    )
    public["ending_draw_kind"] = tf.where(
        discard & abort,
        2,
        tf.where(
            discard
            & (tf.gather(features["live_wall_count"], selected[:, 0]) == 0),
            1,
            0,
        ),
    )
    public["ending_tenpai"] = tf.concat(
        [
            tf.cast(
                tf.gather_nd(
                    features["discard_action_all_shanten"],
                    tf.stack([selected[:, 0], tile], axis=1),
                )
                == 1,
                tf.int32,
            )[:, None],
            tf.where(public["tenpai_required"][:, 1:], 1, -1),
        ],
        axis=1,
    )
    return public


class PublicContext(tf.keras.layers.Layer):
    def call(self, values):
        inputs, selected, structural = values
        tiles = tf.cast(structural[:, -1], tf.int32)
        # Code 37 denotes an action without a tile. Red fives retain their
        # identity in the one-hot feature, but share public safety with fives.
        base = tf.where(
            tiles == 37, 0, tf.where(tiles < 34, tiles, 4 + 9 * (tiles - 34))
        )
        tile_indices = tf.stack([selected[:, 0], base], axis=-1)
        has_tile = tf.cast(tiles != 37, tf.float32)[:, None]
        # Eighteen two-sided wait lines: 14, 25, 36, 47, 58, 69 in each suit.
        # Genbutsu at either endpoint rules out that entire ryanmen line.
        lower = tf.constant(
            [9 * suit + rank for suit in range(3) for rank in range(6)]
        )
        live_lines = ~(
            tf.gather(inputs["genbutsu_to_seat"], lower, axis=2) > 0
        ) & ~(tf.gather(inputs["genbutsu_to_seat"], lower + 3, axis=2) > 0)
        remaining = tf.gather(
            tf.reduce_sum(tf.cast(live_lines, tf.float32), axis=2),
            selected[:, 0],
        )
        tile_lines = (
            tf.reduce_sum(
                tf.cast(tf.gather(live_lines, selected[:, 0]), tf.float32)
                * tf.cast(
                    (base[:, None] == lower) | (base[:, None] == lower + 3),
                    tf.float32,
                )[:, None],
                axis=2,
            )
            * has_tile
        )
        return tf.concat(
            [
                tf.one_hot(tiles, 38),
                tf.one_hot(
                    tf.searchsorted(
                        [37, 74, 111, 112, 223, 260, 261],
                        selected[:, 1],
                        side="right",
                    ),
                    8,
                ),
                tf.cast(
                    tf.gather(inputs["riichi_state"], selected[:, 0]),
                    tf.float32,
                )
                / 2,
                tf.cast(
                    tf.gather(inputs["ippatsu_alive"], selected[:, 0]),
                    tf.float32,
                ),
                tf.cast(
                    tf.gather(
                        inputs["opponent_called_tile_count"], selected[:, 0]
                    ),
                    tf.float32,
                )
                / 16,
                tf.cast(
                    tf.gather(inputs["opponent_discard_count"], selected[:, 0]),
                    tf.float32,
                )
                / 24,
                tf.one_hot(
                    tf.cast(
                        tf.gather(
                            inputs["dealer_relative_seat"], selected[:, 0]
                        ),
                        tf.int32,
                    ),
                    4,
                ),
                *[
                    tf.cast(
                        tf.gather_nd(
                            tf.transpose(inputs[name], [0, 2, 1]), tile_indices
                        ),
                        tf.float32,
                    )
                    * has_tile
                    / scale
                    for name, scale in (
                        ("genbutsu_to_seat", 1),
                        ("sotogawa_earliest_turn_to_seat", 6),
                    )
                ],
                tf.cast(
                    tf.gather_nd(
                        input_features.blocker_counts(inputs), tile_indices
                    ),
                    tf.float32,
                )
                * has_tile
                / 4,
                *[
                    tf.cast(
                        tf.gather_nd(inputs[name], tile_indices), tf.float32
                    )[:, None]
                    * has_tile
                    / 4
                    for name in (
                        "known_unavailable_counts",
                        "dora_multiplicity",
                    )
                ],
                remaining / 18,
                tile_lines / 2,
                tf.math.divide_no_nan(tile_lines, remaining),
                tf.cast(
                    tf.gather_nd(
                        tf.transpose(
                            inputs["passed_unchanged_to_seat"], [0, 2, 1]
                        ),
                        tile_indices,
                    ),
                    tf.float32,
                )
                * has_tile,
                tf.reshape(
                    tf.cast(
                        tf.gather(inputs["riichi_push_count"], selected[:, 0]),
                        tf.float32,
                    ),
                    (-1, 16),
                )
                / 24,
            ],
            axis=-1,
        )


class HandContext(tf.keras.layers.Layer):

    def call(self, values):
        inputs, selected, structural = values
        hands = tf.stack(
            [
                selected[:, 0],
                tf.cast(
                    tf.gather_nd(
                        actions.candidate_hand_indices(inputs),
                        selected,
                    ),
                    tf.int32,
                ),
            ],
            axis=1,
        )
        melds = tf.gather_nd(
            tf.cast(input_features.candidate_melds(inputs), tf.float32), hands
        )
        red = tf.reduce_sum(
            tf.gather_nd(
                tf.cast(inputs["candidate_hand_red"], tf.float32), hands
            ),
            axis=-1,
        ) + tf.reduce_sum(melds[:, :, 5], axis=-1)
        counts = tf.gather_nd(
            tf.cast(inputs["candidate_hand_counts"], tf.float32), hands
        ) + meld_tile_counts(melds)
        dora = (
            tf.reduce_sum(
                counts
                * tf.cast(
                    tf.gather(inputs["dora_multiplicity"], selected[:, 0]),
                    tf.float32,
                ),
                axis=-1,
            )
            + red
        )
        return tf.concat(
            [
                tf.stack(
                    [
                        dora,
                        red,
                        tf.cast(
                            tf.gather(
                                inputs["live_wall_count"], selected[:, 0]
                            ),
                            tf.float32,
                        )
                        / 70,
                        tf.reduce_sum(
                            4
                            - tf.cast(
                                tf.gather(
                                    inputs["known_unavailable_counts"],
                                    selected[:, 0],
                                ),
                                tf.float32,
                            ),
                            axis=-1,
                        )
                        / 136,
                    ],
                    axis=-1,
                ),
                tf.math.asinh(tf.cast(structural[:, :-1], tf.float32)),
            ],
            axis=-1,
        )


class PlayerContext(tf.keras.layers.Layer):
    """Public player descriptors, without learned opponent-slot predictions."""

    def call(self, values):
        inputs, rivers, selected = values
        players = tf.concat(
            [
                tf.broadcast_to(
                    tf.constant([1.0, 0.0, 0.0, 0.0])[None, :, None],
                    (tf.shape(rivers)[0], 4, 1),
                ),
                tf.concat(
                    [tf.zeros_like(rivers[:, :1]), tf.cast(rivers, tf.float32)],
                    axis=1,
                ),
                tf.concat(
                    [
                        tf.gather(
                            input_features.field_features(
                                name, inputs[name], inputs
                            ),
                            tf.cast(inputs["seat_wind"], tf.int32),
                            batch_dims=1,
                        )
                        for name in ("scores", "ranks")
                    ],
                    axis=-1,
                ),
                # Wind identity is not a player ID. Expose its scoring and
                # tile-availability consequences instead.
                tf.stack(
                    [
                        tf.cast(inputs["seat_wind"] == 0, tf.float32),
                        tf.cast(
                            inputs["seat_wind"]
                            == inputs["prevailing_wind"][:, None],
                            tf.float32,
                        ),
                        *[
                            tf.cast(
                                tf.gather(
                                    inputs[name],
                                    tf.cast(inputs["seat_wind"], tf.int32) + 27,
                                    axis=1,
                                    batch_dims=1,
                                ),
                                tf.float32,
                            )
                            / scale
                            for name, scale in (
                                ("known_unavailable_counts", 4),
                                ("dora_multiplicity", 5),
                            )
                        ],
                    ],
                    axis=-1,
                ),
                *[
                    tf.cast(inputs[name], tf.float32)[..., None] / scale
                    for name, scale in (
                        ("riichi_state", 2),
                        ("ippatsu_alive", 1),
                        ("opponent_called_tile_count", 16),
                        ("opponent_discard_count", 24),
                    )
                ],
                tf.math.asinh(
                    tf.cast(
                        input_features.last_tedashi_tile(inputs), tf.float32
                    )
                ),
                *[
                    tf.math.asinh(tf.cast(inputs[name], tf.float32))
                    for name in (
                        "open_opponent_yaku_possibility",
                        "genbutsu_to_seat",
                        "passed_unchanged_to_seat",
                        "sotogawa_earliest_turn_to_seat",
                        "meld_type",
                    )
                ],
                tf.reduce_sum(
                    tf.one_hot(
                        tf.cast(input_features.meld_tile_ids(inputs), tf.int32),
                        34,
                    )
                    * tf.cast(
                        input_features.meld_tile_valid(inputs), tf.float32
                    )[..., None],
                    axis=(2, 3),
                )
                / 4,
                tf.reduce_sum(
                    tf.cast(input_features.meld_tile_red(inputs), tf.float32)
                    * tf.cast(
                        input_features.meld_tile_valid(inputs), tf.float32
                    ),
                    axis=(2, 3),
                )[..., None]
                / 3,
            ],
            axis=-1,
        )
        response = tf.gather(inputs["decision_phase"], selected[:, 0]) == 1
        source = tf.where(
            response,
            tf.cast(
                tf.gather(inputs["trigger_source"], selected[:, 0]), tf.int32
            ),
            0,
        )
        active_discard = response | actions.discard_decisions(inputs, selected)
        return tf.concat(
            [
                tf.gather(players, selected[:, 0]),
                encoder.projected_draws(inputs, selected)[..., :1] / 18,
                tf.cast(
                    active_discard[:, None]
                    & (
                        tf.gather(inputs["live_wall_count"], selected[:, 0]) > 0
                    )[:, None]
                    & (tf.range(4)[None] == (source[:, None] + 1) % 4)
                    & (tf.gather(inputs["riichi_state"], selected[:, 0]) == 0),
                    tf.float32,
                )[..., None],
                tf.cast(
                    active_discard[:, None]
                    & (tf.range(4)[None] == source[:, None]),
                    tf.float32,
                )[..., None],
            ],
            axis=-1,
        )


class BeliefContext(tf.keras.layers.Layer):
    """Small shared residual from an opponent token and belief moments."""

    def __init__(self, belief_offset):
        super().__init__(name="belief_context", dtype="float32", autocast=False)
        self.belief_offset = belief_offset
        self.hidden = tf.keras.layers.Dense(
            128, activation="gelu", dtype="float32"
        )
        self.projection = tf.keras.layers.Dense(
            64, kernel_initializer="zeros", dtype="float32"
        )

    def moments(self, probability):
        probability = tf.cast(probability, tf.float32)
        shanten, ukeire, counts = [
            tf.reshape(value, [-1, *shape])
            for value, shape in zip(
                tf.split(
                    probability,
                    [math.prod(shape) for shape in BELIEF_SHAPES],
                    -1,
                ),
                BELIEF_SHAPES,
            )
        ]
        classes = tf.cast(tf.range(BELIEF_SHAPES[2][-1]), tf.float32)
        mean = tf.reduce_sum(counts * classes, -1)
        variance = tf.maximum(
            0.0,
            tf.reduce_sum(counts * tf.square(classes), -1) - tf.square(mean),
        )
        return tf.concat(
            [
                shanten,
                shanten[..., :1] * ukeire[:, :, 0],
                tf.reduce_sum(shanten[..., None] * ukeire, axis=2),
                mean / 4,
                variance / 4,
                -tf.reduce_sum(
                    tf.math.xlogy(shanten, tf.maximum(shanten, 1e-6)),
                    -1,
                    keepdims=True,
                )
                / tf.math.log(float(BELIEF_SHAPES[0][-1])),
            ],
            axis=-1,
        )

    def call(self, values):
        state, tokens = values
        probability = state[
            :, self.belief_offset : self.belief_offset + BELIEF_WIDTH
        ]
        return self.projection(
            self.hidden(
                tf.concat(
                    [tf.cast(tokens, tf.float32), self.moments(probability)], -1
                )
            )
        )


class PaymentCritic(tf.keras.Model):
    """Predict action-conditioned hand events, scores and deposits."""

    def __init__(self, belief_offset, shanten_offset, player_width):
        super().__init__(dtype="float32", autocast=False)
        self.belief_offset = belief_offset
        self.player_width = player_width
        self.shanten_offset = shanten_offset
        self.state = tf.keras.layers.Dense(
            256, activation="gelu", dtype="float32"
        )
        self.refinement = [
            tf.keras.Sequential(
                [
                    tf.keras.layers.LayerNormalization(dtype="float32"),
                    tf.keras.layers.Dense(
                        896, activation="gelu", dtype="float32"
                    ),
                    tf.keras.layers.Dense(
                        256, kernel_initializer="zeros", dtype="float32"
                    ),
                ],
                name=f"critic_refinement_{block}",
            )
            for block in range(6)
        ]
        for layer in self.refinement:
            layer.dtype_policy = "float32"
        self.belief_context = BeliefContext(belief_offset)
        self.self_player = tf.keras.layers.Dense(
            64, activation="gelu", dtype="float32"
        )
        self.opponent_players = tf.keras.layers.Dense(
            64, activation="gelu", dtype="float32"
        )
        self.cell_state = tf.keras.layers.Dense(
            64, activation="gelu", dtype="float32"
        )
        self.event_context = tf.keras.layers.Dense(16, dtype="float32")
        self.event_roles = [
            tf.keras.layers.Dense(16, dtype="float32") for _ in range(3)
        ]
        self.event_kind = tf.keras.layers.Embedding(4, 16, dtype="float32")
        self.event_kind.build((len(joint_settlement.EVENTS),))
        self.event = tf.keras.layers.Dense(1, dtype="float32")
        self.deposit_context = tf.keras.layers.Dense(8, dtype="float32")
        self.deposit_players = tf.keras.layers.Dense(8, dtype="float32")
        self.deposit_roles = tf.keras.layers.Dense(8, dtype="float32")
        self.deposit = tf.keras.layers.Dense(1, dtype="float32")
        self.total_score = tf.keras.layers.Dense(
            len(payment_scoring.SCORES), dtype="float32", name="total_score"
        )

    def context(self, state):
        state = tf.cast(state, tf.float32)
        hidden = self.state(state[:, : -4 * self.player_width])
        for layer in self.refinement:
            hidden = hidden + layer(hidden)
        descriptors = tf.concat(
            [
                tf.reshape(
                    state[:, -4 * self.player_width :],
                    (-1, 4, self.player_width),
                ),
                tf.broadcast_to(hidden[:, None], (tf.shape(state)[0], 4, 256)),
            ],
            axis=-1,
        )
        opponents = self.opponent_players(descriptors[:, 1:])
        players = tf.concat(
            [
                self.self_player(descriptors[:, :1]),
                opponents + self.belief_context([state, opponents]),
            ],
            axis=1,
        )
        return hidden, players

    def event_contexts(self, hidden, players):
        # Joint masks remain whole events. Shared role pooling is followed by
        # a nonlinearity, not a product of independent player probabilities.
        roles = tf.constant(
            [
                [
                    [
                        float(bool(winners & (1 << seat))),
                        float(source == seat),
                        float(bool(ready & (1 << seat))),
                    ]
                    for seat in range(4)
                ]
                for _, source, winners, ready in joint_settlement.EVENTS
            ],
            tf.float32,
        )
        return (
            tf.nn.gelu(
                self.event_context(hidden)[:, None]
                + tf.linalg.matmul(
                    tf.one_hot([row[0] for row in joint_settlement.EVENTS], 4),
                    self.event_kind.embeddings,
                )[None]
                + tf.add_n(
                    [
                        tf.einsum(
                            "es,nsh->neh",
                            roles[..., role],
                            projection(players),
                        )
                        for role, projection in enumerate(self.event_roles)
                    ]
                )
            ),
            roles,
        )

    def event_logits(self, state):
        hidden, players = self.context(state)
        event, _ = self.event_contexts(hidden, players)
        return self.event(event)[..., 0]

    def distributions(self, inputs):
        """Scored ron/tsumo events, exhaustive masks, abort and bank deposits.

        Twelve ron claimant slots plus four tsumo winners. All three tsumo
        payer shares use exactly the same total han/fu distribution.
        """
        hidden, players = self.context(inputs["state"])
        shares = [
            cell if cell < 12 else 12 + row[2]
            for cell, row in enumerate(scoring.CELLS[:24])
        ]
        cells = self.cell_state(
            tf.concat(
                [
                    tf.gather(
                        players,
                        [
                            scoring.CELLS[cell][2]
                            for cell in joint_settlement.SCORE_HEADS
                        ],
                        axis=1,
                    ),
                    tf.gather(
                        players,
                        [
                            scoring.CELLS[cell][1]
                            for cell in joint_settlement.SCORE_HEADS
                        ],
                        axis=1,
                    )
                    * tf.constant(
                        [cell < 12 for cell in joint_settlement.SCORE_HEADS],
                        tf.float32,
                    )[None, :, None],
                    *[
                        tf.gather(
                            tf.cast(inputs[name], tf.float32),
                            [
                                scoring.CELLS[cell][2]
                                for cell in joint_settlement.SCORE_HEADS
                            ],
                            axis=1,
                        )[..., None]
                        / scale
                        for name, scale in (
                            ("public_fu", 128),
                            ("public_bonus", 5),
                            ("opened", 1),
                            ("riichi_required", 1),
                        )
                    ],
                    tf.broadcast_to(
                        tf.one_hot(
                            [
                                scoring.CELLS[cell][0]
                                for cell in joint_settlement.SCORE_HEADS
                            ],
                            2,
                        )[None],
                        (tf.shape(players)[0], 16, 2),
                    ),
                ],
                axis=-1,
            )
        )
        # One total han/fu distribution per claim; tsumo payer shares use
        # the same winning-hand prediction and exact payment conversion.
        raw_score = tf.gather(self.total_score(cells), shares, axis=1)
        score_logits = payment_scoring.masked_score_logits(
            logits=raw_score,
            dealer=inputs["dealer"],
            meld_fu=inputs["public_fu"],
            opened=inputs["opened"],
            riichi_required=inputs["riichi_required"],
            bonus=inputs["public_bonus"],
            yakuman=inputs["yakuman_allowed"],
        )
        event, roles = self.event_contexts(hidden, players)
        deposit = tf.nn.gelu(
            self.deposit_context(event)[:, :, None]
            + tf.einsum(
                "ds,nsh->ndh",
                tf.constant(joint_settlement.DEPOSITS),
                self.deposit_players(players),
            )[:, None]
            + self.deposit_roles(
                tf.einsum(
                    "ds,esr->edr", tf.constant(joint_settlement.DEPOSITS), roles
                )
            )[None]
        )
        result = {
            "score_logits": score_logits,
            "score_probability": tf.nn.softmax(score_logits),
            "event_logits": self.event(event)[..., 0],
            "deposit_logits": self.deposit(deposit)[..., 0],
        }

        _, _, legal = joint_settlement.event_probabilities({**inputs, **result})
        possible = tf.reduce_any(legal, -1)
        result["critic_constraint_loss"] = (
            tf.reduce_mean(
                losses.constraint_divergence(raw_score, score_logits), -1
            )
            + losses.constraint_divergence(
                result["event_logits"],
                tf.where(possible, result["event_logits"], -1e9),
            )
            + tf.math.divide_no_nan(
                tf.reduce_sum(
                    losses.constraint_divergence(
                        result["deposit_logits"],
                        tf.where(legal, result["deposit_logits"], -1e9),
                    )
                    * tf.cast(possible, tf.float32),
                    -1,
                ),
                tf.reduce_sum(tf.cast(possible, tf.float32), -1),
            )
        )
        return result

    def joint_distributions(self, inputs):
        factors = self.distributions(inputs)
        result = {**factors, **joint_settlement.decode({**inputs, **factors})}
        result["payment_probability"] = joint_settlement.marginals(result)
        return result

    def candidate_chunk(self, inputs, payoff, weights, integer_fields):
        """Bounded, detached forecasts for actor objectives."""
        inputs = {
            name: (
                tf.bitcast(value, tf.int32) if name in integer_fields else value
            )
            for name, value in inputs.items()
        }
        with tf.keras.StatelessScope(
            state_mapping=zip(self.trainable_variables, weights)
        ):
            outputs = PaymentCritic.distributions(self, inputs)
            probability, _ = joint_settlement.probabilities(
                {**inputs, **outputs}
            )
        return tf.stop_gradient(
            joint_settlement.expectation(probability, payoff)
        )

    def direction_losses(self, inputs):
        """Train monotone belief-odds responses with other inputs fixed."""
        state = tf.stop_gradient(inputs["state"])
        shanten = tf.reshape(
            state[:, self.belief_offset : self.belief_offset + 12], [-1, 3, 4]
        )
        waits = tf.reshape(
            state[:, self.belief_offset + 12 : self.belief_offset + 420],
            [-1, 3, 4, 34],
        )
        odds = tf.constant([0.5, 2.0])[None, None, :, None, None]
        perturbed_shanten = shanten[:, None, None] * tf.where(
            tf.eye(3, dtype=tf.bool)[None, :, None, :, None]
            & (tf.range(4)[None, None, None, None] == 0),
            odds,
            1.0,
        )
        perturbed_shanten /= tf.reduce_sum(perturbed_shanten, -1, keepdims=True)
        tile = tf.where(
            inputs["action_tile"] < 34,
            inputs["action_tile"],
            4 + 9 * (inputs["action_tile"] - 34),
        )
        perturbed_waits = tf.where(
            tf.eye(3, dtype=tf.bool)[None, :, None, :, None, None]
            & (inputs["action_tile"] < 37)[:, None, None, None, None, None]
            & (tf.range(4)[None, None, None, None, :, None] == 0)
            & (
                tf.range(34)[None, None, None, None, None]
                == tile[:, None, None, None, None, None]
            ),
            waits[:, None, None]
            * odds[..., None]
            / (
                1
                - waits[:, None, None]
                + waits[:, None, None] * odds[..., None]
            ),
            waits[:, None, None],
        )
        wins = tf.constant(
            [
                [
                    bool(mask & (1 << seat))
                    and (kind == 1 or (kind == 0 and mask.bit_count() < 3))
                    for kind, _, mask, _ in joint_settlement.EVENTS
                ]
                for seat in range(1, 4)
            ],
            tf.float32,
        )
        ron = wins * tf.constant(
            [
                kind == 0 and payer == 0
                for kind, payer, _, _ in joint_settlement.EVENTS
            ],
            tf.float32,
        )
        ready = tf.constant(
            [
                [
                    kind == 2 and bool(mask & (1 << seat))
                    for kind, _, _, mask in joint_settlement.EVENTS
                ]
                for seat in range(1, 4)
            ],
            tf.float32,
        )
        losses_per_row = []
        for offset, width, perturbed in (
            (self.belief_offset, 12, perturbed_shanten),
            (self.belief_offset + 12, 408, perturbed_waits),
        ):
            probe = tf.concat(
                [
                    tf.repeat(state[:, :offset], 6, axis=0),
                    tf.reshape(perturbed, [-1, width]),
                    tf.repeat(state[:, offset + width :], 6, axis=0),
                ],
                axis=1,
            )
            logits = tf.reshape(
                self.event_logits(probe),
                [-1, 3, 2, len(joint_settlement.EVENTS)],
            )
            probability = tf.nn.softmax(
                tf.where(
                    tf.reduce_any(joint_settlement.event_legal(inputs), -1)[
                        :, None, None
                    ],
                    logits,
                    -1e9,
                )
            )
            if width == 12:
                draw = tf.reduce_sum(
                    probability
                    * tf.constant(
                        [
                            kind == 2
                            for kind, _, _, _ in joint_settlement.EVENTS
                        ],
                        tf.float32,
                    ),
                    -1,
                )
                predictions = tf.stack(
                    [
                        tf.einsum("nose,oe->nos", probability, wins),
                        tf.math.divide_no_nan(
                            tf.einsum("nose,oe->nos", probability, ready), draw
                        ),
                    ],
                    axis=-1,
                )
            else:
                predictions = tf.einsum("nose,oe->nos", probability, ron)[
                    ..., None
                ]
            target = tf.where(
                (predictions[:, :, 0] > predictions[:, :, 1])[:, :, None],
                tf.reduce_mean(predictions, axis=2, keepdims=True),
                predictions,
            )
            predictions = tf.clip_by_value(predictions, 1e-7, 1 - 1e-7)
            target = tf.clip_by_value(target, 1e-7, 1 - 1e-7)
            losses_per_row.append(
                tf.reduce_mean(
                    tf.maximum(
                        0.0,
                        losses.constraint_divergence(
                            tf.math.log(predictions)
                            - tf.math.log1p(-predictions),
                            tf.math.log(target) - tf.math.log1p(-target),
                            binary=True,
                        ),
                    ),
                    axis=(1, 2),
                )
            )
        return tf.stack(losses_per_row, axis=1)

    def shanten_direction_loss(self, inputs):
        """Penalize lower J after improving only our explicit shanten input."""
        inputs = {
            name: tf.stop_gradient(value) for name, value in inputs.items()
        }
        state = inputs["state"]
        # HandContext stores asinh(shanten + 1). Non-discard slots use zero.
        distance = tf.math.sinh(state[:, self.shanten_offset])
        payoff = tf.stop_gradient(joint_settlement.position_payoffs(inputs))
        values = []
        for improved in (False, True):
            probe = {
                **inputs,
                "state": tf.concat(
                    [
                        state[:, : self.shanten_offset],
                        tf.math.asinh(
                            tf.maximum(distance - 1, 1)
                            if improved
                            else distance
                        )[:, None],
                        state[:, self.shanten_offset + 1 :],
                    ],
                    axis=1,
                ),
            }
            probability, _ = joint_settlement.probabilities(
                {**probe, **PaymentCritic.distributions(self, probe)}
            )
            values.append(joint_settlement.expectation(probability, payoff))
        # Final-wall/abort support fixes readiness from the actual hand;
        # changing shanten alone there would contradict that public support.
        eligible = (distance > 1.5) & (inputs["ending_draw_kind"] == 0)
        return tf.where(
            eligible, tf.square(tf.nn.relu(values[0] - values[1])), 0.0
        )

    def chunk_statistics(self, inputs, labels, weights=None):
        # Inline the differentiable body into the chunk compilation. Nested
        # compiled gradients can specialize on forward tensor values.
        if weights is None:
            outputs = PaymentCritic.distributions(self, inputs)
            direction = self.direction_losses(inputs)
            shanten_direction = self.shanten_direction_loss(inputs)
        else:
            with tf.keras.StatelessScope(
                state_mapping=zip(self.trainable_variables, weights)
            ):
                outputs = PaymentCritic.distributions(self, inputs)
                direction = self.direction_losses(inputs)
                shanten_direction = self.shanten_direction_loss(inputs)

        def mean(values, mask):
            return losses.masked_totals(
                values,
                mask,
                tf.reshape(
                    labels["valid"], [-1] + [1] * (values.shape.rank - 1)
                ),
            )

        metrics = joint_settlement.observed_terms(
            labels, {**inputs, **outputs}, mean=mean, coverage_mean=mean
        )
        metrics["critic_constraint_loss"] = mean(
            outputs["critic_constraint_loss"]
            + tf.reduce_sum(direction, -1)
            + shanten_direction,
            tf.ones_like(outputs["critic_constraint_loss"]),
        )
        metrics["tenpai_direction_loss"] = mean(
            direction[:, 0], tf.ones_like(direction[:, 0])
        )
        metrics["wait_direction_loss"] = mean(
            direction[:, 1], tf.ones_like(direction[:, 1])
        )
        metrics["shanten_direction_loss"] = mean(
            shanten_direction, tf.ones_like(shanten_direction)
        )
        return tf.stack([metrics[name] for name in CRITIC_METRICS])

    def chunk_gradients(self, inputs, labels, weights, upstream, gradients):
        """Keep recomputation, differentiation and accumulation together."""
        with tf.GradientTape() as tape:
            tape.watch([inputs["state"], *weights])
            statistics = PaymentCritic.chunk_statistics(
                self, inputs, labels, weights
            )
        values = tape.gradient(
            statistics,
            [inputs["state"], *weights],
            output_gradients=upstream,
        )
        return (
            values[0],
            [total + value for total, value in zip(gradients, values[1:])],
        )

    def supervised_statistics(self, inputs, labels, valid=None):
        labels = {
            **labels,
            "valid": (
                tf.ones(tf.shape(inputs["state"])[0])
                if valid is None
                else tf.cast(valid, tf.float32)
            ),
        }

        rows = tf.shape(inputs["state"])[0]
        indices = tf.range(
            (rows + STATISTICS_CHUNK - 1) // STATISTICS_CHUNK * STATISTICS_CHUNK
        )
        inputs = {
            name: tf.gather(value, tf.minimum(indices, rows - 1))
            for name, value in inputs.items()
        }
        labels = {
            **{
                name: tf.gather(value, tf.minimum(indices, rows - 1))
                for name, value in labels.items()
            },
            "valid": tf.gather(labels["valid"], tf.minimum(indices, rows - 1))
            * tf.cast(indices < rows, tf.float32),
        }

        # Retain only loss sufficient statistics. Recompute one chunk at a
        # time in the backward pass instead of retaining every settlement.
        # Explicit weight tensors preserve the local replica's dependency;
        # implicit custom-gradient variable capture can select replica zero.
        @tf.custom_gradient
        def summarize(state, *weights):
            statistics = tf.zeros((len(CRITIC_METRICS), 2))
            for begin in tf.range(0, tf.shape(state)[0], STATISTICS_CHUNK):
                tf.autograph.experimental.set_loop_options(
                    parallel_iterations=1
                )
                statistics += self.chunk_statistics(
                    {
                        **{
                            name: value[begin : begin + STATISTICS_CHUNK]
                            for name, value in inputs.items()
                        },
                        "state": state[begin : begin + STATISTICS_CHUNK],
                    },
                    {
                        name: value[begin : begin + STATISTICS_CHUNK]
                        for name, value in labels.items()
                    },
                    weights,
                )

            @tf.function
            def backward(upstream):
                state_gradients = tf.TensorArray(
                    tf.float32, size=0, dynamic_size=True
                )
                gradients = [tf.zeros_like(value) for value in weights]
                for begin in tf.range(0, tf.shape(state)[0], STATISTICS_CHUNK):
                    tf.autograph.experimental.set_loop_options(
                        parallel_iterations=1
                    )
                    values = self.chunk_gradients(
                        inputs={
                            **{
                                name: value[begin : begin + STATISTICS_CHUNK]
                                for name, value in inputs.items()
                            },
                            "state": state[begin : begin + STATISTICS_CHUNK],
                        },
                        labels={
                            name: value[begin : begin + STATISTICS_CHUNK]
                            for name, value in labels.items()
                        },
                        weights=weights,
                        upstream=upstream,
                        gradients=gradients,
                    )
                    state_gradients = state_gradients.write(
                        begin // STATISTICS_CHUNK, values[0]
                    )
                    gradients = values[1]
                return (
                    state_gradients.concat(),
                    *gradients,
                )

            return statistics, backward

        return summarize(
            inputs["state"],
            *[value.value.read_value() for value in self.trainable_variables],
        )

    def call(self, inputs):
        return self.joint_distributions(inputs)


CRITIC_METRICS = (
    "payment_nll",
    "settlement_multi_ron_fraction",
    "settlement_supported_fraction",
    "critic_constraint_loss",
    "tenpai_direction_loss",
    "wait_direction_loss",
    "shanten_direction_loss",
)


def settlement_objective(metrics):
    """Observed settlement likelihood and structural constraints."""
    return metrics["payment_nll"] + metrics["critic_constraint_loss"]


def chosen_actions(features, labels):
    return tf.where(
        features["decision_phase"] == 4,
        260,
        tf.where(
            labels["discard_policy_logits"] >= 0,
            tf.cast(labels["discard_policy_logits"], tf.int32),
            tf.where(
                labels["riichi_policy_logits"] >= 0,
                37 + tf.cast(labels["riichi_policy_logits"], tf.int32),
                tf.where(
                    labels["response_policy_logits"] >= 0,
                    tf.where(
                        labels["response_policy_logits"] == 149,
                        261 + tf.cast(features["trigger_tile_id"], tf.int32),
                        111
                        + tf.cast(labels["response_policy_logits"], tf.int32),
                    ),
                    260 + tf.cast(labels["kan_action_logits"], tf.int32),
                ),
            ),
        ),
    )


def active_candidates(features, selected):
    phase = tf.gather(features["decision_phase"], selected[:, 0])
    action = selected[:, 1]
    return (
        ((phase == 0) & (action < 37))
        | ((phase == 3) & (action >= 37) & (action < 111))
        | (
            (phase == 1)
            & (
                ((action >= 111) & (action < 260))
                | (
                    action
                    == 261
                    + tf.cast(
                        tf.gather(features["trigger_tile_id"], selected[:, 0]),
                        tf.int32,
                    )
                )
            )
        )
        | ((phase == 2) & (action >= 260))
        | ((phase == 4) & (action == 260))
    )


def candidate_probabilities(features, outputs):
    probability = {
        name: tf.nn.softmax(outputs[name])
        * tf.cast(features["decision_phase"][:, None] == phase, tf.float32)
        for name, phase in (
            ("discard_policy_logits", 0),
            ("riichi_policy_logits", 3),
            ("response_policy_logits", 1),
            ("kan_action_logits", 2),
        )
    }
    # Response daiminkan shares a candidate with the trigger tile's kan.
    probability = tf.tensor_scatter_nd_add(
        tf.concat(
            [
                probability["discard_policy_logits"],
                probability["riichi_policy_logits"],
                probability["response_policy_logits"][:, :149],
                probability["kan_action_logits"],
            ],
            axis=1,
        ),
        tf.stack(
            [
                tf.range(tf.shape(features["decision_phase"])[0]),
                261 + tf.cast(features["trigger_tile_id"], tf.int32),
            ],
            axis=1,
        ),
        probability["response_policy_logits"][:, 149],
    )
    # Win/pass exposes a current-hand forecast, not a fabricated
    # action-conditioned forecast.
    return tf.where(
        (features["decision_phase"] == 4)[:, None],
        tf.one_hot(260, 295),
        probability,
    )
