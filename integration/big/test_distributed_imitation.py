"""Masked imitation has the same gradient across replica layouts."""

import os
import pathlib
import subprocess
import sys


def test_imitation_keeps_its_weight_with_uneven_replica_labels():
    # Labeled rows can all land on the last replica. Their policy
    # gradient must match a single-device mean over the same valid labels.
    subprocess.run(
        args=[
            sys.executable,
            "-c",
            """
import tensorflow as tf
cpu = tf.config.list_physical_devices("CPU")[0]
tf.config.set_logical_device_configuration(cpu, [
    tf.config.LogicalDeviceConfiguration(),
    tf.config.LogicalDeviceConfiguration(),
])
from model import actions
from model import losses
from model import actor as actor_layers
for replicas in (1, 2):
    strategy = tf.distribute.MirroredStrategy(
        devices=[f"/cpu:{index}" for index in range(replicas)])
    for packed in (False, True):
        with strategy.scope():
            value = tf.Variable(0.0)

        def update():
            active = (
                tf.distribute.get_replica_context().replica_id_in_sync_group
                == replicas - 1)
            labels = {
                name: tf.constant([-1])
                for name in actions.TRAIN_POLICY_HEADS
            }
            labels["discard_policy_logits"] = tf.reshape(
                tf.where(active, 0, -1), (1,))
            with tf.GradientTape(persistent=True) as tape:
                outputs = {
                    name: tf.reshape(tf.stack([value, 0.0]), (1, 2))
                    for name in actions.TRAIN_POLICY_HEADS
                }
                def terms(mean):
                    return (
                        actor_layers.policy_loss(labels, outputs, mean=mean),
                        actor_layers.awr_policy_terms(
                            actor_layers.policy_nll(labels, outputs)[0],
                            tf.ones((1,)), tf.reshape(active, (1,)),
                            mean=mean)["loss"],
                    )
                if packed:
                    means = losses.MaskedMeans()
                    terms(means.collect)
                    means.reduce()
                    imitation, loss = terms(means.read)
                else:
                    imitation, loss = terms(losses.mean_valid)
                scaled_imitation = imitation / replicas
                scaled = loss / replicas
            return (loss, tape.gradient(scaled, value),
                    tape.gradient(scaled_imitation, value))

        loss_values, gradients, imitation_gradients = strategy.run(update)
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_values, None)
        gradient = strategy.reduce(tf.distribute.ReduceOp.SUM, gradients, None)
        tf.debugging.assert_near(loss, tf.math.log(2.0))
        tf.debugging.assert_near(gradient, -0.5)
        tf.debugging.assert_near(gradient, strategy.reduce(
            tf.distribute.ReduceOp.SUM, imitation_gradients, None))
""",
        ],
        cwd=pathlib.Path(__file__).parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
        timeout=60,
    )


def test_streamed_payment_backward_matches_one_and_two_replicas():
    subprocess.run(
        args=[
            sys.executable,
            "-c",
            """
import numpy
import tensorflow as tf

cpu = tf.config.list_physical_devices("CPU")[0]
tf.config.set_logical_device_configuration(
    cpu,
    [
        tf.config.LogicalDeviceConfiguration(),
        tf.config.LogicalDeviceConfiguration(),
    ],
)
from model import critic
from model import losses
from model import payment_scoring
from log_dataset import scoring
from integration.small.test_joint_settlement import inputs

public = inputs()
public.update(
    {
        "state": tf.ones((1, 64)),
        "immediate_wait": tf.ones((1, 3)) * 0.3,
        "immediate_safe": tf.zeros((1, 3), tf.bool),
        "immediate_allowed": tf.ones(1, tf.bool),
        "immediate_ankan": tf.zeros(1, tf.bool),
        "has_meld": tf.zeros((1, 4), tf.bool),
        "opened": tf.zeros((1, 4), tf.bool),
        "riichi_required": tf.zeros((1, 4), tf.bool),
        "ippatsu_allowed": tf.ones((1, 4), tf.bool),
        "immediate_ippatsu": tf.ones((1, 4), tf.bool),
        "public_fu": tf.zeros((1, 4), tf.int32),
        "public_bonus": tf.zeros((1, 4), tf.int32),
        "yakuman_allowed": tf.ones((1, 4), tf.bool),
        "immediate_draw_kind": tf.zeros(1, tf.int32),
        "immediate_tenpai": tf.fill((1, 4), -1),
    }
)
for name in [
    "immediate_probability",
    "immediate_score_probability",
    "score_probability",
    "event_logits",
    "deposit_logits",
]:
    del public[name]
labels = {
    "payment": tf.ones((1, 40), tf.int32),
    "round_placement": tf.constant([[0, 1, 2, 3]]),
    "pot": public["pot"],
    "points": tf.zeros(1),
    "bonus": tf.fill((1, 24), 2),
    "score": tf.fill(
        (1, 24, len(scoring.BONUS_HAN)),
        payment_scoring.SCORES.index((0, 1)),
    ),
}
public = {name: tf.repeat(value, 2, axis=0) for name, value in public.items()}
public["state"] *= tf.constant([[1.0], [2.0]])
public["immediate_wait"] *= tf.constant([[1.0], [2.0]])
labels = {name: tf.repeat(value, 2, axis=0) for name, value in labels.items()}
tf.keras.utils.set_random_seed(941)
for replicas in (1, 2):
    strategy = tf.distribute.MirroredStrategy(
        devices=[f"/cpu:{i}" for i in range(replicas)]
    )
    with strategy.scope():
        model = critic.PaymentCritic()
        model(public)
        if replicas == 1:
            weights = model.get_weights()
        else:
            model.set_weights(weights)
        model.distributions = tf.function(model.distributions, jit_compile=True)
        model.joint_distributions = tf.function(
            model.joint_distributions, jit_compile=True
        )
        model.chunk_statistics = tf.function(
            model.chunk_statistics, jit_compile=True
        )
        model.chunk_gradients = tf.function(
            model.chunk_gradients, jit_compile=True
        )
        model.supervised_statistics = tf.function(model.supervised_statistics)

    def update():
        begin = (
            tf.distribute.get_replica_context().replica_id_in_sync_group
            * (2 // replicas)
        )
        with tf.GradientTape() as tape:
            stats = model.supervised_statistics(
                {name: value[begin:begin + 2 // replicas]
                 for name, value in public.items()},
                {name: value[begin:begin + 2 // replicas]
                 for name, value in labels.items()},
            )
            means = losses.packed_means(stats)
            loss = (
                0.1 * means[critic.CRITIC_METRICS.index("payment_nll")]
                + 0.01 * means[critic.CRITIC_METRICS.index("score_nll")]
                + 0.01 * means[critic.CRITIC_METRICS.index("bonus_han_nll")]
            ) / replicas
        return loss, tape.gradient(loss, model.trainable_variables)

    @tf.function
    def distributed():
        loss, gradient = strategy.run(update)
        return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, None), [
            strategy.reduce(tf.distribute.ReduceOp.SUM, g, None)
            for g in gradient
        ]

    loss, gradients = distributed()
    if replicas == 1:
        # Recomputed training gradients must equal the direct objective;
        # detaching reported diagnostics cannot change an optimizer update.
        with tf.GradientTape() as tape:
            outputs = model.joint_distributions(public)
            from integration.small.test_joint_settlement import expanded_terms
            terms = expanded_terms(labels, outputs)
            direct = (
                0.1 * terms["payment_nll"]
                + 0.01 * critic.score_cross_entropy(
                    labels["score"],
                    outputs["score_logits"]
                )
                + 0.01 * losses.mean_valid(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels["bonus"], logits=outputs["bonus_logits"]
                    ), labels["bonus"] >= 0
                )
            )
        numpy.testing.assert_allclose(loss, direct, atol=1e-5, rtol=1e-4)
        for actual, expected in zip(
            gradients, tape.gradient(direct, model.trainable_variables)
        ):
            numpy.testing.assert_allclose(
                actual, expected, atol=1e-5, rtol=1e-4
            )
        reference = [g.numpy() for g in gradients]
    else:
        for first, second in zip(reference, gradients):
            numpy.testing.assert_allclose(first, second, atol=1e-5, rtol=1e-4)
""",
        ],
        cwd=pathlib.Path(__file__).parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
        timeout=180,
    )


def test_training_adamw_and_masked_losses_match_reference():
    subprocess.run(
        args=[
            sys.executable,
            "-c",
            """
import numpy
import tensorflow as tf

cpu = tf.config.list_physical_devices("CPU")[0]
tf.config.set_logical_device_configuration(
    cpu,
    [
        tf.config.LogicalDeviceConfiguration(),
        tf.config.LogicalDeviceConfiguration(),
    ],
)
from log_dataset import _native_collection
from model import actions
from model import training
from model import losses
import types
strategies = {
    replicas: tf.distribute.MirroredStrategy(
        devices=[f"/cpu:{i}" for i in range(replicas)]
    )
    for replicas in (1, 2)
}

class Payment(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.value = tf.keras.layers.Dense(1)

    def supervised_statistics(self, inputs, labels, valid):
        prediction = self.value(inputs["state"])[:, 0]
        return tf.stack([
            losses.masked_totals(tf.square(prediction - 0.02), valid),
            tf.zeros(2),
            tf.zeros(2),
            losses.masked_totals(tf.zeros_like(prediction), valid),
            losses.masked_totals(tf.ones_like(prediction), valid),
        ])


class Network(tf.keras.Model):
    def __init__(self):
        super().__init__()
        x = tf.keras.Input((1,), name="x")
        board = tf.keras.layers.Dense(2)(x)
        logits = tf.keras.layers.Dense(150)(board)
        self.critic = Payment()
        self.actor = tf.keras.Model(
            {"x": x},
            {
                "board": board,
                **{
                    name + "_prior_loss": logits[:, 0] ** 2
                    for name in actions.TRAIN_POLICY_HEADS
                },
                **{
                    name: logits
                    for name in actions.TRAIN_POLICY_HEADS
                },
            },
        )

    def compile_training(self):
        pass

    def candidate_context(self, features, outputs):
        return outputs, outputs["board"]

    def chosen_inputs(self, features, outputs, state, chosen):
        return {"state": state, "pot": tf.zeros(tf.shape(state)[0])}

    def call(self, inputs):
        outputs = self.actor(inputs[0])
        self.critic.value(outputs["board"])
        return outputs


def belief_terms(features, labels, outputs, mean=losses.mean_valid):
    return {
        "belief": mean(
            tf.square(outputs["board"][:, 0] - labels["y"]),
            labels["y"] >= 0,
        )
    }, {}


training.critic.chosen_actions = lambda features, labels: (
    labels["discard_policy_logits"])
training.beliefs.belief_loss_terms = belief_terms
training.critic.advantages = lambda *args: tf.zeros(1)
training.objective_metrics = lambda inputs, outputs, advantages: (
    belief_terms(inputs[0], inputs[1], outputs)[0]["belief"],
    {
        name.removesuffix("_logits") + "_" + suffix: tf.reduce_sum(
            tf.cast(inputs[1]["y"] == 0.01 if suffix == "correct"
                    else inputs[1]["y"] >= 0, tf.float32))
        for name in actions.TRAIN_POLICY_HEADS
        for suffix in ("count", "correct", "top3_correct")
    } | {
        "policy_entropy": tf.constant(0.0),
        "policy_prior_loss": tf.constant(0.0),
    },
)
first = (
    {"x": tf.constant([[0.01], [0.09], [0.03], [0.05]])},
    {"y": tf.constant([-1.0, 0.02, 0.01, -1.0])},
    {},
)
for replicas in (1, 2):
    tf.keras.utils.set_random_seed(941)
    strategy = strategies[replicas]
    with strategy.scope():
        network = Network()
        network(first)
        trainer = training.Trainer(
            network=network,
            args=types.SimpleNamespace(
                ppo_learning_rate=0.01,
                supervised_learning_rate=2e-4,
                awr_only_exposures=0,
                final_awr_exposures=0,
                final_awr_kl=0.1,
                prior_weight=1.0,
                foundation_exposures=400,
                awr_exposures=600,
                block_exposures=100,
                batch_size=4,
                clip=0.2,
                dual_clip=3.0,
                entropy=0.001,
                awr_entropy=0.01,
                awr_temperature=3.0,
            ),
            first=first,
        )
    trainer = training.DistributedTrainer(trainer, strategy)
    batch = dict(
        features=trainer.feature_packing.pack(first[0]),
        head=tf.zeros(4, tf.int32),
        legal=tf.ones((4, 150), tf.bool),
        chosen=tf.constant([0, 1, 2, 3]),
        advantages=tf.constant([0.02, -0.01, 0.03, 0.04]),
        payment_points=tf.zeros((4, 40), tf.int32),
        critic_valid=tf.constant([False, False, True, True]),
    )
    batch["behavior"] = trainer.predict(
        {**batch, "unused_replay_metadata": object()})
    inputs = trainer.packing.pack(first)
    before = [v.numpy().copy() for v in network.weights]
    baseline = trainer.auxiliary_update(inputs, False, train=False)
    assert float(baseline["imitation_accuracy"]) == 0.5
    assert int(trainer.auxiliary_optimizer.iterations.numpy()) == 0
    assert int(trainer.policy_optimizer.iterations.numpy()) == 0
    for left, right in zip(before, network.weights):
        numpy.testing.assert_array_equal(left, right.numpy())
    metrics = []
    for count in (3, 1):
        metrics.extend(
            trainer.policy_update(
                batch, inputs, tf.constant(count)
            ).values()
        )
    for awr in (False, True):
        auxiliary = trainer.auxiliary_update(inputs, awr)
        assert float(auxiliary["imitation_accuracy"]) == 0.5
        assert float(auxiliary["imitation_top3_accuracy"]) == 1.0
        metrics.extend(auxiliary.values())
    result = [numpy.asarray(v) for v in [*network.weights, *metrics]]
    if replicas == 1:
        expected = result
    else:
        for left, right in zip(expected, result):
            numpy.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
from model import self_play
from model import losses
class Actor(tf.Module):
    def __init__(self):
        super().__init__()
        self.scale = tf.Variable(1.0)
    def __call__(self, features, training=False):
        return {
            name: tf.cast(features["decision_phase"][:, None], tf.float32)
            * self.scale * tf.ones((1, actions.ACTOR_OUTPUTS[name][0]))
            for name in actions.TRAIN_POLICY_HEADS
        }
for replicas in (1, 2):
    with strategies[replicas].scope():
        actor = Actor()
        predict = self_play.predictor(actor)
    for scale in (1.0, 2.0):
        actor.scale.assign(scale)
        for count in (1, 3, 8):
            features = {
                name: numpy.zeros((count, *spec["shape"]), spec["dtype"])
                for name, spec in _native_collection.INPUTS.items()
            }
            features["decision_phase"][:] = numpy.arange(count)
            for values in predict(features).values():
                numpy.testing.assert_array_equal(
                    values[:, 0], numpy.arange(count) * scale
                )
                assert len(values) == count
print("Distributed optimizer, training loss and inference parity passed.")

""",
        ],
        cwd=pathlib.Path(__file__).parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
        timeout=120,
    )
