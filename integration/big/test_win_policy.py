"""Recorded wins and on-policy returns train the same optional-win head."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import replay, scoring, training_data
from model import actions, actor, critic, network


def test_recorded_win_head_learns_both_choices_and_uses_state_baseline():
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/payment_dealer_90fu.json"
        ).read_text()
    )
    targets = scoring.round_labels(events)
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
    model(inputs)
    model.compile_training()
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
            imitation = actor.policy_loss(labels, outputs)
            current = tf.nn.log_softmax(outputs["win_decision_logits"])
            ppo = actor.ppo_loss(
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
    outputs = model(inputs)
    numpy.testing.assert_allclose(
        critic.advantages(model, inputs, outputs, 3.0), [0.0], atol=1e-6
    )
    unavailable = model.actor({**inputs[0], "win_kind": tf.zeros(1, tf.int8)})
    assert unavailable["win_decision_logits"][0, 1] == -1e4
