"""Shared settlement heads preserve roles and train belief direction."""

import numpy
import tensorflow as tf

from integration import settlement_reference
from log_dataset import _native_collection, scoring
from model import critic, encoder, joint_settlement


def test_unobserved_players_differ_only_through_public_wind_effects():
    features = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    features["meld_tiles"][...] = 37
    features["candidate_meld_tiles"][...] = 34
    features["last_tedashi_tile_id"][...] = 37
    features["seat_wind"][0] = [0, 1, 2, 3]
    context = critic.PlayerContext(dtype="float32")
    rivers = tf.zeros((1, 3, 256))
    selected = tf.constant([[0, 0]])
    original = context([features, rivers, selected]).numpy()
    numpy.testing.assert_array_equal(original[:, 1], original[:, 2])
    numpy.testing.assert_array_equal(original[:, 2], original[:, 3])
    features["known_unavailable_counts"][0, 28] = 2
    features["dora_multiplicity"][0, 29] = 1
    changed = context([features, rivers, selected]).numpy()
    numpy.testing.assert_array_equal(changed[:, 3], original[:, 3])
    assert changed[0, 1, 261] == 0.5
    numpy.testing.assert_allclose(changed[0, 2, 262], 0.2)
    features["meld_tiles"][0, 1, 0, :3] = 4
    features["meld_tiles"][0, 1, 0, 0] = 34
    changed = context([features, rivers, selected]).numpy()
    # Tile counts and red count precede the three action-context fields.
    meld_counts = changed[0, 1, -38:-4]
    assert meld_counts[4] == 0.75
    numpy.testing.assert_allclose(changed[0, 1, -4], 1 / 3)

    # A pass continues after the discarder; a call after us. A kan takes
    # one live-wall tile for rinshan, excluded from ordinary draw counts.
    features = {name: tf.constant(value) for name, value in features.items()}
    features.update(
        live_wall_count=tf.constant([5], tf.int8),
        decision_phase=tf.constant([1], tf.int8),
        trigger_source=tf.constant([2], tf.int8),
    )
    selected = tf.constant([[0, 111], [0, 112], [0, 261]])
    changed = context([features, rivers, selected]).numpy()
    numpy.testing.assert_allclose(
        changed[..., -3] * 18,
        [[1, 1, 1, 2], [1, 2, 1, 1], [1, 1, 1, 1]],
    )
    numpy.testing.assert_array_equal(changed[..., -2], [[0, 0, 0, 1]] * 3)
    numpy.testing.assert_array_equal(changed[..., -1], [[0, 0, 1, 0]] * 3)
    numpy.testing.assert_array_equal(
        encoder.last_draw_context(features, selected)[:, 6:],
        [[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
    )


def test_shared_roles_preserve_joint_events_and_equivalent_players():
    tf.keras.utils.set_random_seed(941)
    model = critic.PaymentCritic(
        belief_offset=0, shanten_offset=critic.BELIEF_WIDTH + 4, player_width=8
    )
    public = settlement_reference.inputs()
    public.update(
        has_meld=tf.zeros((1, 4), tf.bool),
        opened=tf.zeros((1, 4), tf.bool),
        riichi_required=tf.zeros((1, 4), tf.bool),
        ippatsu_allowed=tf.ones((1, 4), tf.bool),
        public_fu=tf.zeros((1, 4), tf.int32),
        public_bonus=tf.zeros((1, 4), tf.int32),
        yakuman_allowed=tf.ones((1, 4), tf.bool),
    )
    common = tf.concat(
        [
            tf.fill((1, 12), 0.25),
            tf.fill((1, 408), 0.5),
            tf.fill((1, 555), 0.2),
            tf.random.normal((1, 92)),
            tf.zeros((1, critic.PREDICTED_TAIL_WIDTH)),
        ],
        -1,
    )
    players = tf.random.normal((1, 4, 8))
    permutation = [0, 2, 1, 3]

    def permute_mask(mask):
        return (
            sum(
                1 << permutation[seat]
                for seat in range(4)
                if mask & (1 << seat)
            )
            if mask < 16
            else mask
        )

    event_order = [
        joint_settlement.EVENTS.index(
            (
                kind,
                permutation[source] if source >= 0 else source,
                permute_mask(winners),
                permute_mask(ready),
            )
        )
        for kind, source, winners, ready in joint_settlement.EVENTS
    ]
    payment_order = [
        scoring.CELLS.index((kind, permutation[source], permutation[winner]))
        for kind, source, winner in scoring.CELLS[:24]
    ]
    deposit_order = [permute_mask(mask) for mask in range(16)]
    for equal in (False, True):
        if equal:
            players = tf.concat(
                [players[:, :1], tf.repeat(players[:, 1:2], 3, axis=1)], 1
            )
        public["state"] = tf.concat([common, tf.reshape(players, (1, 32))], -1)
        outputs = model.distributions(public)
        public["state"] = tf.concat(
            [
                common,
                tf.reshape(tf.gather(players, permutation, axis=1), (1, 32)),
            ],
            -1,
        )
        swapped = model.distributions(public)
        numpy.testing.assert_allclose(
            outputs["event_logits"],
            tf.gather(swapped["event_logits"], event_order, axis=1),
            atol=2e-6,
            rtol=2e-6,
        )
        numpy.testing.assert_allclose(
            outputs["deposit_logits"],
            tf.gather(
                tf.gather(swapped["deposit_logits"], event_order, axis=1),
                deposit_order,
                axis=2,
            ),
            atol=2e-6,
            rtol=2e-6,
        )
        numpy.testing.assert_allclose(
            outputs["score_logits"],
            tf.gather(swapped["score_logits"], payment_order, axis=1),
            atol=2e-6,
            rtol=2e-6,
        )
        if equal:
            numpy.testing.assert_array_equal(
                public["state"][:, -32:], tf.reshape(players, (1, 32))
            )


def test_belief_direction_uses_shared_event_path_without_moving_beliefs():
    tf.keras.utils.set_random_seed(942)
    public = {
        name: tf.repeat(value, 2, axis=0)
        for name, value in settlement_reference.inputs().items()
    }
    public["action_tile"] = tf.constant([31, 37])
    public["state"] = tf.Variable(
        tf.concat(
            [
                tf.fill((2, 12), 0.25),
                tf.fill((2, 408), 0.5),
                tf.fill((2, 555), 0.2),
                tf.ones((2, 92)),
                tf.zeros((2, critic.PREDICTED_TAIL_WIDTH)),
                tf.random.normal((2, 32)),
            ],
            -1,
        )
    )
    model = critic.PaymentCritic(
        belief_offset=0, shanten_offset=critic.BELIEF_WIDTH + 4, player_width=8
    )
    with tf.GradientTape(persistent=True) as tape:
        penalties = model.direction_losses(public)
        total = tf.reduce_sum(penalties)
    assert tape.gradient(total, public["state"]) is None
    assert float(penalties[1, 1]) == 0
    assert float(total) > 0
    for variables in (
        model.state.trainable_variables,
        [
            variable
            for layer in model.refinement
            for variable in layer.layers[-1].trainable_variables
        ],
        model.belief_context.projection.trainable_variables,
        model.self_player.trainable_variables,
        model.opponent_players.trainable_variables,
        model.event.trainable_variables,
    ):
        gradients = tape.gradient(total, variables)
        assert all(numpy.isfinite(value).all() for value in gradients)
        assert float(tf.linalg.global_norm(gradients)) > 0


def test_own_shanten_direction_penalizes_only_falling_ev(monkeypatch):
    public = settlement_reference.inputs()
    # Compare one-shanten with tenpai, with identical value and danger inputs.
    public["state"] = tf.Variable([[numpy.arcsinh(2.0)]], dtype=tf.float32)
    model = critic.PaymentCritic(
        belief_offset=0, shanten_offset=0, player_width=8
    )
    slope = tf.Variable(1.0)

    def distributions(self, inputs):
        # Deliberately predict more self wins as distance increases.
        return {
            **public,
            "event_logits": (
                slope
                * tf.math.sinh(inputs["state"][:, :1])
                * tf.constant(
                    [
                        float(kind == 1 and source == 0)
                        for kind, source, _, _ in joint_settlement.EVENTS
                    ]
                )[None]
            ),
        }

    monkeypatch.setattr(critic.PaymentCritic, "distributions", distributions)
    with tf.GradientTape(persistent=True) as tape:
        penalty = tf.reduce_sum(model.shanten_direction_loss(public))
    assert float(penalty) > 0
    assert float(tape.gradient(penalty, slope)) > 0
    assert tape.gradient(penalty, public["state"]) is None
    slope.assign(-1.0)
    numpy.testing.assert_array_equal(model.shanten_direction_loss(public), 0)
    slope.assign(1.0)
    for distance, ending in ((1, 0), (0, 0), (2, 1), (2, 2)):
        public["state"].assign([[numpy.arcsinh(distance)]])
        public["ending_draw_kind"] = tf.constant([ending])
        numpy.testing.assert_array_equal(
            model.shanten_direction_loss(public), 0
        )
