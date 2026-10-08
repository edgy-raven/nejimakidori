"""Phasic training preserves behavior probabilities and snapshot anchors."""

import types

import numpy
import tensorflow as tf
from model import feature_vector, phasic


def test_clipped_ppo_and_forward_kl_have_the_expected_gradients():
    behavior = tf.math.log([[0.5, 0.5], [0.5, 0.5]])
    logits = tf.Variable(tf.math.log([[0.8, 0.2], [0.2, 0.8]]))
    with tf.GradientTape() as tape:
        loss = phasic.ppo_loss(
            current=tf.nn.log_softmax(logits),
            behavior=behavior,
            chosen=tf.constant([0, 0]),
            advantages=tf.constant([1.0, -1.0]),
            clip=0.2,
        )
    numpy.testing.assert_allclose(loss, -0.2, atol=1e-6)
    numpy.testing.assert_allclose(tape.gradient(loss, logits), 0, atol=1e-6)
    reference = tf.Variable(behavior)
    with tf.GradientTape(persistent=True) as tape:
        kl = phasic.forward_kl(reference, tf.nn.log_softmax(logits))
    assert tape.gradient(kl, reference) is None
    assert float(kl) > 0
    assert numpy.linalg.norm(tape.gradient(kl, logits)) > 0
    numpy.testing.assert_allclose(phasic.forward_kl(behavior, behavior), 0)


def test_native_self_play_records_legal_sampled_actions():
    class Actor:
        def __call__(self, inputs, training=False):
            count = tf.shape(inputs["decision_phase"])[0]
            return {
                **{
                    name: tf.zeros(
                        (count, feature_vector.ACTOR_OUTPUTS[name][0])
                    )
                    for name in feature_vector.TRAIN_POLICY_HEADS
                },
                "board": tf.zeros((count, 1)),
            }

    tf.keras.utils.set_random_seed(12)
    batch = phasic.collect(Actor(), lambda board: board, 1, 12)
    assert len(batch["chosen"]) > 100
    probability = numpy.exp(batch["behavior"])
    numpy.testing.assert_allclose(probability.sum(-1), 1, atol=1e-6)
    assert numpy.all(probability[~batch["legal"]] == 0)
    assert numpy.all(
        batch["legal"][numpy.arange(len(batch["chosen"])), batch["chosen"]]
    )
    assert len(numpy.unique(batch["returns"])) > 1
    numpy.testing.assert_allclose(batch["advantages"].mean(), 0, atol=1e-6)
    numpy.testing.assert_allclose(batch["advantages"].std(), 1, atol=1e-6)
    current = phasic.distribution(
        Actor()(batch["features"]), batch["head"], batch["legal"]
    )
    numpy.testing.assert_allclose(current, batch["behavior"], atol=1e-6)


def test_auxiliary_phase_updates_actor_but_keeps_snapshot_fixed(
    monkeypatch, tmp_path
):
    class Network(tf.keras.Model):
        def __init__(self):
            super().__init__()
            inputs = tf.keras.Input((1,), name="x")
            board = tf.keras.layers.Dense(2)(inputs)
            logits = tf.keras.layers.Dense(150)(board)
            self.actor = tf.keras.Model(
                {"x": inputs},
                {
                    "board": board,
                    **{
                        name: logits
                        for name in feature_vector.TRAIN_POLICY_HEADS
                    },
                },
            )

        def call(self, inputs):
            return self.actor(inputs[0])

    monkeypatch.setattr(
        phasic.critic_utility, "advantages", lambda *args: tf.zeros(2)
    )
    monkeypatch.setattr(
        phasic.critic_full,
        "objective_metrics",
        lambda inputs, outputs, utility: (
            -tf.reduce_mean(outputs["discard_policy_logits"][:, 0]),
            {},
        ),
    )
    network = Network()
    inputs = ({"x": tf.ones((2, 1))}, {}, {})
    network(inputs)
    value = tf.keras.layers.Dense(1)
    value(network.actor(inputs[0])["board"])
    trainer = phasic.Trainer(
        network,
        value,
        types.SimpleNamespace(
            learning_rate=0.01,
            clip=0.2,
            entropy=0.001,
            awr_temperature=3.0,
            awr_mix=1.0,
            auxiliary_kl=1.0,
            human_kl=0.01,
        ),
    )
    batch = dict(
        features=inputs[0],
        head=tf.constant([0, 0]),
        legal=tf.ones((2, 150), tf.bool),
        chosen=tf.constant([0, 1]),
        returns=tf.constant([1.0, -1.0]),
        advantages=tf.constant([1.0, -1.0]),
    )
    batch["behavior"] = trainer.predict(batch)
    batch["human"] = trainer.predict(batch)
    trainer.policy_update(batch)
    batch["reference"] = trainer.predict(batch)
    snapshot = batch["reference"].numpy().copy()
    before = [v.numpy().copy() for v in network.trainable_variables]
    trainer.auxiliary_update(inputs, batch)
    assert any(
        numpy.any(old != variable.numpy())
        for old, variable in zip(before, network.trainable_variables)
    )
    numpy.testing.assert_array_equal(batch["reference"], snapshot)
    assert (
        float(phasic.forward_kl(batch["reference"], trainer.predict(batch))) > 0
    )

    checkpoint = tf.train.Checkpoint(
        network=network,
        value=value,
        policy_optimizer=trainer.policy_optimizer,
        auxiliary_optimizer=trainer.auxiliary_optimizer,
    )
    saved = checkpoint.save(str(tmp_path / "cycle"))
    expected = trainer.predict(batch).numpy()
    trainer.policy_update(batch)
    trainer.auxiliary_update(inputs, batch)
    checkpoint.restore(saved).assert_consumed()
    numpy.testing.assert_allclose(trainer.predict(batch), expected)
    assert int(trainer.policy_optimizer.iterations) == 1
    assert int(trainer.auxiliary_optimizer.iterations) == 1


def test_real_joint_network_supports_both_training_phases():
    import json
    import pathlib

    from log_dataset import afk, critic_data, payment_bonus, replay
    from model import critic

    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    targets = payment_bonus.round_labels(events)
    observations = list(
        replay.Replay(
            "phasic-probe",
            afk.retained_hands(events)[0],
            hanchan=replay.hanchan_targets(events),
        ).observations()
    )[:2]
    inputs = critic_data.parse_batch(
        tf.constant(
            [
                critic_data.serialize(row, targets[row.kyoku_id])
                for row in observations
            ]
        )
    )
    network = critic.JointPolicyPayment()
    outputs = network(inputs)
    value = tf.keras.layers.Dense(1, dtype="float32")
    value(outputs["board"])
    network.compile_training()
    trainer = phasic.Trainer(
        network,
        value,
        types.SimpleNamespace(
            learning_rate=1e-5,
            clip=0.2,
            entropy=0.001,
            awr_temperature=3.0,
            awr_mix=1.0,
            auxiliary_kl=1.0,
            human_kl=0.01,
        ),
    )
    heads = tf.stack(
        [inputs[1][name] >= 0 for name in feature_vector.TRAIN_POLICY_HEADS],
        axis=1,
    )
    head = tf.argmax(tf.cast(heads, tf.int32), axis=1, output_type=tf.int32)
    chosen = tf.reduce_max(
        tf.stack(
            [
                tf.cast(inputs[1][name], tf.int32)
                for name in feature_vector.TRAIN_POLICY_HEADS
            ],
            axis=1,
        ),
        axis=1,
    )
    legal = tf.gather(
        tf.stack(
            [
                tf.pad(
                    outputs[name] > -1e3,
                    [[0, 0], [0, 150 - outputs[name].shape[-1]]],
                )
                for name in feature_vector.TRAIN_POLICY_HEADS
            ],
            axis=1,
        ),
        head,
        batch_dims=1,
    )
    batch = dict(
        features=inputs[0],
        head=head,
        legal=legal,
        chosen=chosen,
        returns=tf.constant([1.0, -1.0]),
        advantages=tf.constant([1.0, -1.0]),
    )
    batch["behavior"] = trainer.predict(batch)
    batch["human"] = trainer.predict(batch)
    before = network.actor.get_layer("final_pass").kernel.numpy().copy()
    policy = trainer.policy_update(batch)
    assert numpy.any(
        before != network.actor.get_layer("final_pass").kernel.numpy()
    )
    batch["reference"] = trainer.predict(batch)
    auxiliary = trainer.auxiliary_update(inputs, batch)
    assert all(
        numpy.isfinite(float(v)) for v in {**policy, **auxiliary}.values()
    )
