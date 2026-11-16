"""Q fitting updates shared features and critic without changing actor heads."""

import json
import pathlib

import numpy
import tensorflow as tf
from log_dataset import replay, training_data
from model import critic, joint_settlement
from model import network as joint_network
from training import trainer


def test_q_only_fit_updates_encoder_and_critic_with_likelihood_constraints():
    events = json.loads(
        pathlib.Path(__file__)
        .parents[1]
        .joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    outcomes = training_data.game_targets(events + [{"type": "end_game"}])
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
    network(parsed)
    network.compile_training()
    fitting = trainer.CriticTrainer(network, parsed[0], 1e-5)
    batch = {
        "features": fitting.feature_packing.pack_numpy(
            {name: value.numpy() for name, value in parsed[0].items()}
        ),
        "chosen": critic.chosen_actions(parsed[0], parsed[1]).numpy(),
        "payment_points": tf.gather(
            joint_settlement.payment_values(parsed[2]),
            parsed[2]["payment"],
            axis=1,
            batch_dims=1,
        ).numpy(),
    }
    policy_before = network.actor.get_layer("final_pass").kernel.numpy().copy()
    critic_before = network.critic.event.kernel.numpy().copy()
    encoder_before = (
        network.actor.get_layer("board_block_3_contract").kernel.numpy().copy()
    )
    metrics = trainer.DistributedCriticTrainer(
        fitting, tf.distribute.get_strategy()
    ).step(batch, tf.constant(2))
    numpy.testing.assert_allclose(
        metrics["loss"],
        metrics["payment_nll"] + metrics["critic_constraint_loss"],
        rtol=1e-6,
    )
    assert all(numpy.isfinite(float(value)) for value in metrics.values())
    assert numpy.any(network.critic.event.kernel.numpy() != critic_before)
    assert numpy.any(
        network.actor.get_layer("board_block_3_contract").kernel.numpy()
        != encoder_before
    )
    numpy.testing.assert_array_equal(
        network.actor.get_layer("final_pass").kernel, policy_before
    )
