"""Policy masks, opening symmetry and optional-win learning contracts."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import (
    _native_collection,
    call_states,
    replay,
    scoring,
    training_data,
)
from model import actions
from model import actor as actor_layers
from model import critic, network
from training import restoration


def test_dealer_opening_symmetry_and_legal_branch_masks():
    inputs = {
        name: tf.zeros([1, *spec["shape"]], spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["meld_tiles"] = tf.fill((1, 4, 4, 4), tf.constant(37, tf.int8))
    inputs["candidate_hand_counts"] = tf.tensor_scatter_nd_update(
        inputs["candidate_hand_counts"], [[0, 0, 0], [0, 0, 9]], [1, 1]
    )
    inputs["discard_action_legal"] = tf.constant(
        [[1] + [0] * 8 + [1] + [0] * 27], tf.int8
    )
    inputs["riichi_discard_legal_mask"] = inputs["discard_action_legal"]
    inputs["live_wall_count"] = tf.constant([69], tf.int8)
    logits = tf.one_hot([9], 37) * 2
    discard = actor_layers.PolicyMask("discard_policy_logits")
    output = discard([logits, inputs])
    numpy.testing.assert_allclose(output[0, 0], output[0, 9])
    numpy.testing.assert_allclose(
        tf.reduce_sum(tf.exp(tf.gather(output[0], [0, 9]))),
        tf.reduce_sum(tf.exp(tf.gather(logits[0], [0, 9]))),
        rtol=1e-6,
    )
    for dealer, counts in ((3, [0, 1, 0, 0]), (0, [1, 1, 1, 1])):
        inputs["dealer_relative_seat"] = tf.constant([dealer], tf.int8)
        inputs["opponent_discard_count"] = tf.constant([counts], tf.int8)
        numpy.testing.assert_array_equal(
            tf.gather(discard([logits, inputs])[0], [0, 9]), [0, 2]
        )
    inputs["opponent_discard_count"] = tf.zeros((1, 4), tf.int8)
    inputs["dora_multiplicity"] = tf.one_hot([0], 34, dtype=tf.int8)
    assert discard([logits, inputs])[0, 9] > discard([logits, inputs])[0, 0]
    inputs["dora_multiplicity"] = tf.zeros((1, 34), tf.int8)
    inputs["riichi_discard_legal_mask"] = tf.one_hot([0], 37, dtype=tf.int8)
    output = actor_layers.PolicyMask("riichi_policy_logits")(
        [tf.concat([logits, logits + 0.7], axis=-1), inputs]
    )
    assert output[0, 46] == -1e4
    traced = tf.function(lambda x: discard([x, inputs]))(logits)
    numpy.testing.assert_allclose(traced, discard([logits, inputs]), atol=1e-6)


def test_fourth_honor_requires_an_independent_kokushi_blocker():
    inputs = {
        name: numpy.zeros((4, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["known_unavailable_counts"][:, 29] = 4
    inputs["known_unavailable_counts"][1, 0] = 4
    inputs["known_unavailable_counts"][2, 28] = 4
    inputs["known_unavailable_counts"][3, 4] = 4  # Middle tiles do not block.
    inputs["meld_tiles"][...] = 37
    inputs["meld_tiles"][0, 2, 0, :3] = 8
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


def test_final_wall_cut_support_matches_policy_and_teacher_targets():
    inputs = {
        name: numpy.zeros((7, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["meld_tiles"][...] = 37
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
    numpy.testing.assert_array_equal(probabilities[3], target)
    numpy.testing.assert_allclose(probabilities[4:6], original[4:6])
    numpy.testing.assert_array_equal(probabilities[6], target)
    # Suppressed legal labels cannot move the policy's common logit offset.
    with tf.GradientTape() as tape:
        unsupported_loss = -tf.nn.log_softmax(policy([logits, inputs]))[3, 16]
    numpy.testing.assert_allclose(
        tape.gradient(unsupported_loss, logits), 0, atol=1e-6
    )
    # Ordinary/dama use the parent support. A joint chi uses its child's cut.
    features = call_states.combine(
        inputs,
        {name: value[3:4] for name, value in inputs.items()},
        tf.constant([2]),
        tf.constant([1]),
    )
    supported = restoration.policy_target_supported(
        features,
        tf.constant([0, 1, 2, 0, 3, 4, 0]),
        tf.constant([16, 16, 1 + 37 + 16, 16, 16, 1, 18]),
    )
    numpy.testing.assert_array_equal(
        supported, [True, True, False, False, True, True, True]
    )
    traced = tf.function(lambda x: policy([x, inputs]))(logits)
    numpy.testing.assert_allclose(traced, result, atol=1e-5)
    # Dama uses the same safety mask; a nonexistent riichi branch stays illegal.
    riichi = actor_layers.PolicyMask("riichi_policy_logits")(
        [tf.concat([logits, logits], axis=1), inputs]
    )
    numpy.testing.assert_array_equal(tf.nn.softmax(riichi)[3, :37], target)
    assert not tf.nn.softmax(riichi)[3, 37:].numpy().any()


def test_recorded_win_head_learns_both_choices():
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/payment_dealer_90fu.json"
        ).read_text()
    )
    targets = scoring.round_labels(events + [{"type": "end_game"}])
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
    head = model.actor.policy.get_layer("win_decision_logits")
    for target in (0, 1):
        for variable in head.trainable_variables:
            variable.assign(tf.zeros_like(variable))
        labels = {
            name: tf.constant([target if name == "win_decision_logits" else -1])
            for name in actions.TRAIN_POLICY_HEADS
        }
        with tf.GradientTape(persistent=True) as tape:
            outputs = model.actor(inputs[0])
            active, chosen, legal = restoration.policy_choices(
                inputs[0], labels
            )
            imitation = -tf.reduce_mean(
                tf.gather(
                    actor_layers.distribution(outputs, active, legal),
                    chosen,
                    batch_dims=1,
                )
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
