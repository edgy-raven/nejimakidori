"""Expert records restore policy and belief structures without RL targets."""

import tensorflow as tf

from training import restoration


def test_recorded_structures_restore_policy_and_beliefs():
    import json
    import pathlib

    import numpy

    from log_dataset import replay, scoring, training_data
    from model import actor as actor_layers
    from model import beliefs
    from model import network as joint_network

    events = json.loads(
        pathlib.Path(__file__)
        .parents[1]
        .joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    targets = scoring.round_labels(events + [{"type": "end_game"}])
    observations = list(replay.Replay("restoration", events).observations())[:2]
    raw = tf.constant(
        [
            training_data.serialize(row, targets[row.kyoku_id])
            for row in observations
        ]
    )
    features, labels, _ = training_data.parse_batch(raw)
    restored, head, chosen, legal = restoration.decode_expert(raw)
    tf.nest.map_structure(numpy.testing.assert_array_equal, restored, features)
    numpy.testing.assert_array_equal(
        tf.gather(legal, chosen, batch_dims=1), True
    )
    network = joint_network.JointPolicyPayment()
    with tf.GradientTape(persistent=True) as tape:
        outputs = network.actor(features, training=False)
        policy = actor_layers.offline_loss_terms((features, labels), outputs)
        belief, _ = beliefs.belief_loss_terms(features, labels, outputs)
        policy_loss = tf.add_n(list(policy.values()))
        belief_loss = tf.add_n(list(belief.values()))
    # Restoration is ordinary supervised learning of the recorded choice;
    # critic values and magnetic references are not needed.
    numpy.testing.assert_allclose(
        policy["policy_loss"],
        -tf.reduce_mean(
            tf.gather(
                actor_layers.distribution(outputs, head, legal),
                chosen,
                batch_dims=1,
            )
        ),
    )
    assert numpy.isfinite(float(policy_loss))
    assert numpy.isfinite(float(belief_loss))
    for loss in (policy_loss, belief_loss):
        gradients = tape.gradient(loss, network.actor.trainable_variables)
        assert (
            tf.linalg.global_norm(
                [gradient for gradient in gradients if gradient is not None]
            )
            > 0
        )
    assert not network.critic.built
