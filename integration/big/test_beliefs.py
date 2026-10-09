"""Belief training owns its encoder, labels, and recoverable optimizer."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import afk, critic_data, payment_bonus, replay
from model import belief_training, beliefs, model


def test_belief_updates_and_optimizer_survive_a_new_training_instance(tmp_path):
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model")
        .joinpath("payment_dealer_90fu.json")
        .read_text()
    )
    targets = payment_bonus.round_labels(events)
    rows = list(
        replay.Replay(
            "belief-training", afk.retained_hands(events)[0]
        ).observations()
    )[:4]
    inputs = critic_data.parse_batch(
        tf.constant(
            [critic_data.serialize(row, targets[row.kyoku_id]) for row in rows]
        )
    )
    network = beliefs.OpponentBeliefs()
    before = network(inputs[0])
    checkpoint = belief_training.optimizer_checkpoint(network, 1e-4)
    metrics = belief_training.update(checkpoint, inputs)
    assert all(numpy.isfinite(value) for value in metrics.values())
    assert int(checkpoint.optimizer.iterations) == 1
    for name in beliefs.HEADS:
        assert numpy.any(
            network(inputs[0])[name].numpy() != before[name].numpy()
        )
    path = checkpoint.save(str(tmp_path / "beliefs"))
    restored = beliefs.OpponentBeliefs()
    restored(inputs[0])
    recovery = belief_training.optimizer_checkpoint(restored, 1e-4)
    recovery.restore(path).assert_consumed()
    for original, actual in zip(
        checkpoint.optimizer.variables, recovery.optimizer.variables
    ):
        numpy.testing.assert_array_equal(original, actual)
    for original, actual in zip(network.weights, restored.weights):
        numpy.testing.assert_array_equal(original, actual)
    belief_training.update(checkpoint, inputs)
    belief_training.update(recovery, inputs)
    for original, actual in zip(network.weights, restored.weights):
        numpy.testing.assert_allclose(original, actual, atol=1e-7, rtol=1e-6)
    restored.save_weights(tmp_path / "snapshot.weights.h5")
    frozen = beliefs.OpponentBeliefs()
    frozen.trainable = False
    frozen(inputs[0])
    frozen.load_weights(tmp_path / "snapshot.weights.h5")
    assert not frozen.trainable_variables
    for name in beliefs.HEADS:
        numpy.testing.assert_array_equal(
            frozen(inputs[0])[name], restored(inputs[0])[name]
        )


def test_count_projection_respects_melds_reds_and_shared_ankou_blockers():
    features = {
        "meld_tile_ids": numpy.zeros((1, 4, 4, 4), numpy.int8),
        "meld_tile_red": numpy.zeros((1, 4, 4, 4), numpy.int8),
        "meld_tile_valid": numpy.zeros((1, 4, 4, 4), numpy.int8),
        "meld_valid": numpy.zeros((1, 4, 4), numpy.int8),
        "known_unavailable_counts": numpy.zeros((1, 34), numpy.int8),
        "candidate_hand_red": numpy.zeros((1, 96, 3), numpy.int8),
        "public_visible_red_counts": numpy.array([[1, 0, 0]], numpy.int8),
        "seat_wind": numpy.array([[0, 1, 2, 3]], numpy.int8),
        "prevailing_wind": numpy.array([0], numpy.int8),
        "dora_multiplicity": numpy.eye(34, dtype=numpy.int8)[[0]],
    }
    # Our 111m and ten other tiles; opponent 1 has melded 550m.
    features["known_unavailable_counts"][0, [0, 4]] = 3
    features["known_unavailable_counts"][0, 18:28] = 1
    features["meld_tile_ids"][0, 1, 0, :3] = 4
    features["meld_tile_red"][0, 1, 0, 2] = 1
    features["meld_tile_valid"][0, 1, 0, :3] = 1
    features["meld_valid"][0, 1, 0] = 1
    features = {key: tf.constant(value) for key, value in features.items()}
    counts = numpy.zeros((1, 3, 37), numpy.int32)
    counts[0, :, [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]] = 1
    counts[0, 1:, [15, 16, 17]] = 1
    counts[0, 0, [4, 34]] = [2, 1]
    logits = tf.Variable(numpy.zeros((1, 3, 37, 5), numpy.float32))
    lower, available, size = model.held_count_context(features)
    # Projection must resolve the shared last copy while preserving the
    # exposed red meld and each concealed hand size. Gradients stay usable.
    with tf.GradientTape() as tape:
        projected = beliefs.CountProjection()(
            [model.HeldCountMask()([logits, features]), features]
        )
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=counts, logits=projected
            )
        )
    gradient = tape.gradient(loss, logits).numpy()
    assert numpy.isfinite(gradient).all()
    assert numpy.any(gradient != 0)
    probability = tf.nn.softmax(projected, -1).numpy()
    concealed = (probability * numpy.arange(5)).sum(-1) - lower.numpy()
    numpy.testing.assert_allclose(concealed.sum(-1), size.numpy(), atol=1e-5)
    assert numpy.all(concealed.sum(1) <= available.numpy() + 1e-5)
    numpy.testing.assert_array_equal(probability[0, :, 0, 2:], 0)
    assert probability[0, :, 0, 1].sum() <= 1 + 1e-5
    numpy.testing.assert_array_equal(probability[0, 0, 34], [0, 1, 0, 0, 0])
    numpy.testing.assert_allclose(
        tf.nn.softmax(beliefs.CountProjection()([projected, features]), -1),
        probability,
        atol=1e-5,
    )

    # Saturated count heads previously exhausted the solver's 80 steps and
    # returned physically inconsistent means. Keep their gradients finite.
    logits.assign(tf.random.stateless_normal(logits.shape, [941, 2]) * 32)
    with tf.GradientTape() as tape:
        projected = beliefs.CountProjection()(
            [model.HeldCountMask()([logits, features]), features]
        )
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=counts, logits=projected
            )
        )
    gradient = tape.gradient(loss, logits).numpy()
    assert numpy.isfinite(gradient).all()
    assert numpy.any(gradient != 0)
    concealed = tf.reduce_sum(
        tf.nn.softmax(projected)
        * (tf.range(5, dtype=tf.float32) - lower[..., None]),
        -1,
    ).numpy()
    numpy.testing.assert_allclose(concealed.sum(-1), size, atol=1e-4)
    assert numpy.all(concealed.sum(1) <= available.numpy() + 1e-4)
