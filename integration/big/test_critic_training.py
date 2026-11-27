"""Critic and beliefs learn shared features; actor objectives stay private."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import call_states, replay, scoring, training_data
from model import beliefs, critic, joint_settlement, losses
from model import network as joint_network
from training import auxiliary, trainer


def test_payment_fit_updates_encoder_and_critic_with_likelihood_constraints():
    events = json.loads(
        pathlib.Path(__file__)
        .parents[1]
        .joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    outcomes = scoring.round_labels(events + [{"type": "end_game"}])
    observations = list(replay.Replay("q-fit", events).observations())[:2]
    parsed = training_data.parse_batch(
        tf.constant(
            [
                training_data.serialize(row, outcomes[row.kyoku_id])
                for row in observations
            ]
        )
    )
    network = joint_network.JointPolicyPayment()
    encoded = network.actor(parsed[0])
    state = network.candidate_context(parsed[0], encoded)
    numpy.testing.assert_allclose(
        tf.math.sinh(state[:, network.critic.shanten_offset]),
        encoded["structure"][:, 0],
        atol=1e-5,
    )
    with tf.GradientTape() as tape:
        statistics = network(parsed)["critic_statistics"]
        payment_nll = losses.packed_means(statistics)[
            critic.CRITIC_METRICS.index("payment_nll")
        ]
    payment_gradients = tape.gradient(
        payment_nll,
        [
            network.critic.event.kernel,
            network.actor.policy.get_layer("board_block_3_contract").kernel,
            network.actor.policy.get_layer(
                "decision_candidates"
            ).projection.kernel,
            network.actor.policy.get_layer("river_encoder").projection.kernel,
            network.actor.policy.get_layer("hand_bank")
            .projection.layers[0]
            .kernel,
            network.actor.policy.get_layer(
                "neural_features"
            ).tile_projection.kernel,
            *network.beliefs.trainable_variables,
            *network.actor.policy.get_layer(
                "candidate_ron"
            ).trainable_variables,
            *network.actor.policy.get_layer(
                "route_affinity"
            ).trainable_variables,
        ],
    )
    assert all(tf.linalg.norm(value) > 0 for value in payment_gradients[:6])
    assert all(value is None for value in payment_gradients[6:])
    network.compile_training()
    fitting = trainer.CriticTrainer(network, parsed[0], 1e-5)
    batch = {
        "features": call_states.pack_numpy(
            {name: value.numpy() for name, value in parsed[0].items()},
            fitting.feature_packing,
        ),
        "chosen": critic.chosen_actions(parsed[0], parsed[1]).numpy(),
        "labels": {name: parsed[1][name] for name in auxiliary.LABELS},
        "payment_points": tf.gather(
            joint_settlement.payment_values(parsed[2]),
            parsed[2]["payment"],
            axis=1,
            batch_dims=1,
        ).numpy(),
    }
    policy_before = (
        network.actor.policy.get_layer("final_pass").kernel.numpy().copy()
    )
    critic_before = network.critic.event.kernel.numpy().copy()
    encoder_before = (
        network.actor.policy.get_layer("board_block_3_contract")
        .kernel.numpy()
        .copy()
    )
    metrics = fitting.critic_update(batch, tf.constant([True, True]))
    numpy.testing.assert_allclose(
        metrics["loss"],
        metrics["payment_nll"]
        + metrics["critic_constraint_loss"]
        + sum(
            metrics[name]
            for name in (
                "opponent_shanten_loss",
                "opponent_ukeire_loss",
                "opponent_hand_count_loss",
                "completion_yaku_loss",
                "completion_score_loss",
                "route_affinity_loss",
                "candidate_ron_loss",
            )
        ),
        rtol=1e-6,
    )
    assert all(numpy.isfinite(float(value)) for value in metrics.values())
    assert numpy.any(network.critic.event.kernel.numpy() != critic_before)
    assert all(
        numpy.any(layer.layers[-1].kernel.numpy())
        for layer in network.critic.refinement
    )
    assert numpy.any(network.critic.belief_context.projection.kernel.numpy())
    assert numpy.any(network.beliefs.refinement.layers[-1].kernel.numpy())
    assert numpy.any(network.beliefs.tile_attention.correction.kernel.numpy())
    for name in beliefs.COMPLETION_HEADS:
        assert numpy.any(network.beliefs.heads[name].bias.numpy())
    assert numpy.any(
        network.actor.policy.get_layer("board_block_3_contract").kernel.numpy()
        != encoder_before
    )
    numpy.testing.assert_array_equal(
        network.actor.policy.get_layer("final_pass").kernel, policy_before
    )

    # After the first update, zero-initialized readouts also transmit their
    # direct supervision. Every named belief must still train shared features.
    with tf.GradientTape(persistent=True) as tape:
        _, supervised, _ = auxiliary.reduce_terms(
            features=parsed[0],
            labels=batch["labels"],
            outputs=network.actor(parsed[0]),
            valid=tf.constant([True, True]),
            terms={},
        )
    assert len(supervised) == 7
    for name, loss in supervised.items():
        gradients = tape.gradient(
            loss,
            [
                network.actor.policy.get_layer("board_block_3_contract").kernel,
                network.actor.policy.get_layer(
                    "neural_features"
                ).tile_projection.kernel,
            ],
        )
        for gradient in gradients:
            assert gradient is not None, name
            assert numpy.isfinite(gradient).all(), name
            assert tf.linalg.norm(gradient) > 0, name
    del tape

    encoded = network.actor(parsed[0], training=False)
    state = network.candidate_context(parsed[0], encoded)
    observed = network.chosen_inputs(parsed[0], encoded, state, batch["chosen"])
    selected = tf.stack([tf.range(2), batch["chosen"]], axis=1)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(observed["state"])
        values = trainer.action_values(
            network=network,
            features=parsed[0],
            selected=selected,
            state=observed["state"],
        )
        total = tf.reduce_sum(values)
    assert values.shape == (2, 295)
    numpy.testing.assert_allclose(
        tf.gather_nd(values, selected),
        tf.reduce_sum(
            network.critic.joint_distributions(observed)["joint_probability"]
            * joint_settlement.position_payoffs(observed),
            axis=(1, 2),
        ),
        atol=1e-6,
        rtol=1e-5,
    )
    numpy.testing.assert_array_equal(
        tf.tensor_scatter_nd_update(values, selected, tf.zeros(2)), 0
    )
    assert tape.gradient(total, observed["state"]) is None
    assert all(
        gradient is None
        for gradient in tape.gradient(total, network.critic.trainable_variables)
    )
