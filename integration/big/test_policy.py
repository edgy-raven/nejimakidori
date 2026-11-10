"""Policy priors, safety masks and optional-win learning contracts."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import _native_collection, replay, training_data
from model import actions
from model import actor as actor_layers
from model import critic, network


def test_efficiency_ranking_decay_masking_and_override():
    inputs = {
        name: tf.zeros([4, *spec["shape"]], spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs.update(
        {
            "discard_action_all_shanten": tf.constant(
                [[2, 2, 2, 3, 5] + [0] * 32] * 4, tf.int8
            ),
            "discard_action_all_ukeire_count": tf.constant(
                [[20, 30, 30, 100, 100] + [0] * 32] * 4, tf.int16
            ),
            "discard_action_all_upgrade_weighted_gain": tf.constant(
                [[0, 0, 1360, 0, 0] + [0] * 32] * 4, tf.int16
            ),
            "discard_action_all_upgrade_valid": tf.constant(
                [[1] * 37] * 3 + [[0] * 37], tf.int8
            ),
            "opponent_discard_count": tf.constant(
                [[0, 0, 0, 0], [0, 2, 1, 1], [0, 5, 4, 4], [0, 0, 0, 0]],
                tf.int8,
            ),
        }
    )
    legal = tf.constant([[True] * 5 + [False] * 32] * 4)
    logits = tf.Variable(tf.zeros((4, 37)))
    with tf.GradientTape() as tape:
        biased = actor_layers.opening_discard_efficiency(logits, inputs, legal)
        loss = tf.reduce_sum(biased)
    values = biased.numpy()
    assert values[0, 2] == values[0, 1]
    assert values[0, 1] > values[0, 0] > values[0, 3]
    assert values[0, 4] == -1.5
    numpy.testing.assert_allclose(values[1], values[0] * 0.6, atol=1e-7)
    numpy.testing.assert_array_equal(values[2], 0)
    numpy.testing.assert_array_equal(values[:, 5:], 0)
    assert values[3, 1] == values[3, 2]
    numpy.testing.assert_array_equal(tape.gradient(loss, logits), 1)
    # A learned value/safety preference can outweigh even the capped penalty.
    logits.assign(tf.one_hot([4] * 4, 37) * 3)
    assert (
        numpy.argmax(
            actor_layers.opening_discard_efficiency(logits, inputs, legal)[0]
        )
        == 4
    )


def test_dealer_override_and_raw_policy_prior_gradients():
    inputs = {
        name: tf.zeros([1, *spec["shape"]], spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["candidate_hand_counts"] = tf.tensor_scatter_nd_update(
        inputs["candidate_hand_counts"], [[0, 0, 0], [0, 0, 9]], [1, 1]
    )
    inputs["discard_action_legal"] = tf.constant(
        [[1] + [0] * 8 + [1] + [0] * 27], tf.int8
    )
    inputs["riichi_discard_legal_mask"] = inputs["discard_action_legal"]
    inputs["live_wall_count"] = tf.constant([69], tf.int8)
    logits = tf.Variable(tf.one_hot([9], 37) * 2)
    discard = actor_layers.PolicyMask("discard_policy_logits")
    prior = actor_layers.PolicyPrior("discard_policy_logits")
    with tf.GradientTape() as tape:
        loss = prior([logits, inputs])
    gradient = tape.gradient(loss, logits).numpy()
    assert gradient[0, 0] < 0 < gradient[0, 9]
    numpy.testing.assert_allclose(gradient.sum(), 0, atol=1e-7)
    output = discard([logits, inputs])
    numpy.testing.assert_allclose(output[0, 0], output[0, 9])
    numpy.testing.assert_allclose(
        tf.reduce_sum(tf.exp(tf.gather(output[0], [0, 9]))),
        tf.reduce_sum(tf.exp(tf.gather(logits[0], [0, 9]))),
        rtol=1e-6,
    )
    # Non-dealers' first cuts and later dealer cuts keep raw preferences.
    for dealer, counts in ((3, [0, 1, 0, 0]), (0, [1, 1, 1, 1])):
        inputs["dealer_relative_seat"] = tf.constant([dealer], tf.int8)
        inputs["opponent_discard_count"] = tf.constant([counts], tf.int8)
        numpy.testing.assert_array_equal(
            tf.gather(discard([logits, inputs])[0], [0, 9]), [0, 2]
        )
        assert 0 < prior([logits, inputs])[0] < loss[0]
    inputs["opponent_discard_count"] = tf.constant([[2, 5, 5, 5]], tf.int8)
    numpy.testing.assert_allclose(prior([logits, inputs]), 0, atol=1e-7)
    inputs["opponent_discard_count"] = tf.zeros((1, 4), tf.int8)
    inputs["dora_multiplicity"] = tf.one_hot([0], 34, dtype=tf.int8)
    assert discard([logits, inputs])[0, 9] > discard([logits, inputs])[0, 0]
    numpy.testing.assert_allclose(prior([logits, inputs]), 0, atol=1e-7)
    # Both branches learn symmetry; unavailable riichi actions stay masked.
    inputs["dora_multiplicity"] = tf.zeros((1, 34), tf.int8)
    riichi_logits = tf.Variable(tf.concat([logits, logits + 0.7], axis=-1))
    with tf.GradientTape() as tape:
        loss = actor_layers.PolicyPrior("riichi_policy_logits")(
            [riichi_logits, inputs]
        )
    gradient = tape.gradient(loss, riichi_logits).numpy()
    assert gradient[0, 0] < 0 < gradient[0, 9]
    assert gradient[0, 37] < 0 < gradient[0, 46]
    inputs["riichi_discard_legal_mask"] = tf.one_hot([0], 37, dtype=tf.int8)
    output = actor_layers.PolicyMask("riichi_policy_logits")(
        [riichi_logits, inputs]
    )
    assert output[0, 46] == -1e4
    traced = tf.function(lambda x: prior([x, inputs]))(logits)
    numpy.testing.assert_allclose(traced, prior([logits, inputs]), atol=1e-6)


def test_fourth_honor_requires_an_independent_kokushi_blocker():
    inputs = {
        name: numpy.zeros((4, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["known_unavailable_counts"][:, 29] = 4
    inputs["known_unavailable_counts"][1, 0] = 4
    inputs["known_unavailable_counts"][2, 28] = 4
    inputs["known_unavailable_counts"][3, 4] = 4  # Middle tiles do not block.
    inputs["meld_valid"][0, 2, 0] = 1
    inputs["genbutsu_to_seat"][:, 1, 16] = 1
    safe = actions.guaranteed_safe_tiles(inputs).numpy()
    numpy.testing.assert_array_equal(
        safe[:, :, 29],
        [[False, True, False], [True] * 3, [True] * 3, [False] * 3],
    )
    assert safe[:, 0, 16].all()
    # Four accounted-for 1m still permits a 23m sequence wait.
    assert not safe[1, :, 0].any()
    assert not safe[3, :, 4].any()
    inputs["known_unavailable_counts"][1, 29] = 3
    assert not actions.guaranteed_safe_tiles(inputs)[1, :, 29].numpy().any()


def test_late_noten_ramp_equal_safe_override_and_exceptions():
    inputs = {
        name: numpy.zeros((7, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    # West complaint: safe 1s and fourth West; unsafe 8p. No tenpai choice.
    inputs["known_unavailable_counts"][:, [18, 29]] = 4
    inputs["genbutsu_to_seat"][:, 1:, 18] = 1
    inputs["genbutsu_to_seat"][:, 0, 4] = 1  # Nagashi already impossible.
    inputs["discard_action_legal"][:, [16, 18, 29]] = 1
    inputs["discard_action_all_shanten"][:] = 2  # Stored shanten + 1.
    inputs["discard_action_all_shanten"][:, 29] = 3
    inputs["opponent_discard_count"][:] = 16
    inputs["live_wall_count"][:] = [9, 8, 4, 0, 0, 0, 0]
    inputs["discard_action_all_shanten"][4, 16] = 1  # Can secure tenpai.
    inputs["genbutsu_to_seat"][5, 1:] = 0
    inputs["known_unavailable_counts"][5] = 0  # No proven-safe option.
    inputs["genbutsu_to_seat"][6, 1:, 16] = 1  # Safe but forfeits nagashi.
    inputs["genbutsu_to_seat"][6, 0] = 0
    inputs["genbutsu_to_seat"][6, 0, 27] = 1  # Possible nagashi.
    inputs = {name: tf.constant(value) for name, value in inputs.items()}
    logits = tf.Variable(numpy.tile(numpy.arange(37), (7, 1)), dtype=tf.float32)
    policy = actor_layers.PolicyMask("discard_policy_logits")
    with tf.GradientTape() as tape:
        result = policy([logits, inputs])
        probabilities = tf.nn.softmax(result)
        loss = -tf.reduce_sum(tf.nn.log_softmax(result)[:, 18])
    assert numpy.isfinite(tape.gradient(loss, logits)).all()
    original = tf.nn.softmax(
        tf.where(inputs["discard_action_legal"] > 0, logits, -1e4)
    ).numpy()
    target = numpy.zeros(37)
    target[[18, 29]] = 0.5
    numpy.testing.assert_allclose(probabilities[:2], original[:2])
    numpy.testing.assert_allclose(probabilities[2], original[2])
    with tf.GradientTape() as tape:
        prior_loss = actor_layers.PolicyPrior("discard_policy_logits")(
            [logits, inputs]
        )
    gradient = tape.gradient(prior_loss, logits).numpy()
    numpy.testing.assert_allclose(
        prior_loss.numpy()[[0, 1, 4, 5]], 0, atol=1e-6
    )
    assert 0 < prior_loss[2] < prior_loss[3]
    assert gradient[3, 16] > 0
    assert gradient[3, 18] < 0 < gradient[3, 29]
    assert numpy.isfinite(gradient).all()
    numpy.testing.assert_array_equal(probabilities[3], target)
    numpy.testing.assert_allclose(probabilities[4:6], original[4:6])
    numpy.testing.assert_array_equal(probabilities[6], target)
    traced = tf.function(lambda x: policy([x, inputs]))(logits)
    numpy.testing.assert_allclose(traced, result, atol=1e-5)
    # Dama uses the same prior; a nonexistent riichi branch stays illegal.
    riichi = actor_layers.PolicyMask("riichi_policy_logits")(
        [tf.concat([logits, logits], axis=1), inputs]
    )
    numpy.testing.assert_array_equal(tf.nn.softmax(riichi)[3, :37], target)
    assert not tf.nn.softmax(riichi)[3, 37:].numpy().any()


def test_recorded_win_head_learns_both_choices_and_uses_state_baseline():
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/payment_dealer_90fu.json"
        ).read_text()
    )
    targets = training_data.game_targets(events + [{"type": "end_game"}])
    wins = [
        row
        for row in replay.Replay("win-policy", events).observations()
        if row.win_action >= 0
    ]
    assert wins and all(row.win_action == 1 for row in wins)
    inputs = training_data.parse_batch(
        tf.constant(
            [training_data.serialize(wins[0], targets[wins[0].kyoku_id])]
        )
    )
    model = network.JointPolicyPayment()
    head = model.actor.get_layer("win_decision_logits")
    for target in (0, 1):
        for variable in head.trainable_variables:
            variable.assign(tf.zeros_like(variable))
        labels = {
            name: tf.constant([target if name == "win_decision_logits" else -1])
            for name in actions.TRAIN_POLICY_HEADS
        }
        with tf.GradientTape(persistent=True) as tape:
            outputs = model.actor(inputs[0])
            imitation = actor_layers.policy_loss(labels, outputs)
            current = tf.nn.log_softmax(outputs["win_decision_logits"])
            ppo = actor_layers.ppo_loss(
                current=current,
                behavior=tf.stop_gradient(current),
                chosen=tf.constant([target]),
                advantages=tf.ones(1),
                clip=0.2,
                dual_clip=3.0,
            )
        numpy.testing.assert_allclose(
            tape.gradient(imitation, head.bias),
            tape.gradient(ppo, head.bias),
        )
        assert (
            float(
                tf.linalg.global_norm(
                    tape.gradient(imitation, head.trainable_variables)
                )
            )
            > 0
        )
        tf.keras.optimizers.SGD(0.1).apply_gradients(
            zip(
                tape.gradient(imitation, head.trainable_variables),
                head.trainable_variables,
            )
        )
        assert (
            int(
                tf.argmax(
                    model.actor(inputs[0])["win_decision_logits"], axis=-1
                )[0]
            )
            == target
        )
    numpy.testing.assert_array_equal(
        critic.chosen_actions(inputs[0], inputs[1]), [260]
    )
    unavailable = model.actor({**inputs[0], "win_kind": tf.zeros(1, tf.int8)})
    assert unavailable["win_decision_logits"][0, 1] == -1e4
