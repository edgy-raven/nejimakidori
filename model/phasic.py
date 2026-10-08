"""Fresh self-play PPO followed by AWR/auxiliary training with policy KL."""

import argparse
import json
import pathlib

import numpy
import tensorflow as tf
from log_dataset import critic_data, replay
from model import (
    actions,
    critic,
    critic_full,
    critic_utility,
    feature_vector,
    features,
    outcomes,
)

import riichienv


def distribution(outputs, heads, legal):
    """Normalize over the exact native candidates used by the collector."""
    logits = tf.stack(
        [
            tf.pad(outputs[name], [[0, 0], [0, 150 - outputs[name].shape[-1]]])
            for name in feature_vector.TRAIN_POLICY_HEADS
        ],
        axis=1,
    )
    logits = tf.gather(logits, heads, batch_dims=1)
    return tf.nn.log_softmax(tf.where(legal, logits, -1e9), axis=-1)


def forward_kl(reference, current):
    reference = tf.stop_gradient(reference)
    return tf.reduce_mean(
        tf.reduce_sum(tf.exp(reference) * (reference - current), axis=-1)
    )


def ppo_loss(*, current, behavior, chosen, advantages, clip):
    log_ratio = tf.gather(current - behavior, chosen, batch_dims=1)
    ratio = tf.exp(log_ratio)
    objective = tf.minimum(
        ratio * advantages,
        tf.clip_by_value(ratio, 1 - clip, 1 + clip) * advantages,
    )
    return -tf.reduce_mean(objective)


def collect(actor, value, games, seed):
    """Complete south games; networks stay fixed until collection finishes."""

    @tf.function(
        input_signature=[
            {
                name: tf.TensorSpec([None, *spec["shape"]], spec["dtype"])
                for name, spec in feature_vector.INPUTS.items()
            }
        ]
    )
    def predict(inputs):
        outputs = actor(inputs, training=False)
        return {
            **{
                name: outputs[name]
                for name in feature_vector.TRAIN_POLICY_HEADS
            },
            "value": value(outputs["board"])[:, 0],
        }

    generator = numpy.random.default_rng(seed)
    envs = [
        riichienv.RiichiEnv(
            game_mode=2,
            seed=seed + game,
            rule=riichienv.GameRule.default_tenhou(),
        )
        for game in range(games)
    ]
    observations = [env.reset(oya=0) for env in envs]
    contexts = [dict(cursor=0, state=None, pending={}) for _ in envs]
    samples = []
    while any(not env.done() for env in envs):
        requests = []
        choices = [{} for _ in envs]
        for game, env in enumerate(envs):
            if env.done():
                continue
            ctx = contexts[game]
            for event in env.mjai_log[ctx["cursor"] :]:
                if event["type"] == "start_kyoku":
                    ctx["state"] = replay.Replay.from_start("self-play", event)
                    ctx["pending"].clear()
                elif ctx["state"] is not None:
                    if event["type"] in ("chi", "pon", "daiminkan"):
                        ctx["pending"] = {
                            seat: plan
                            for seat, plan in ctx["pending"].items()
                            if seat == event["actor"]
                            and plan[1] == event["type"]
                        }
                    elif event["type"] in ("tsumo", "dahai"):
                        ctx["pending"].pop(event["actor"], None)
                    elif event["type"] in ("hora", "ryukyoku", "end_kyoku"):
                        ctx["pending"].clear()
                    ctx["state"].apply(event)
                    if event["type"] == "tsumo" and "tiles_left" in event:
                        ctx["state"].live_wall = event["tiles_left"]
            ctx["cursor"] = len(env.mjai_log)
            for seat, observation in observations[game].items():
                legal = observation.legal_actions()
                selected = actions.forced_policy_action(
                    ctx["state"], seat, legal
                )
                if seat in ctx["pending"] and any(
                    action.action_type == riichienv.ActionType.DISCARD
                    for action in legal
                ):
                    cut, _ = ctx["pending"].pop(seat)
                    if selected is None:
                        selected = actions.select_discard(
                            legal, cut, observation.drawn_tile
                        )
                if selected is not None:
                    choices[game][seat] = selected
                else:
                    _, request = actions.policy_request(
                        ctx["state"], seat, observation
                    )
                    requests.append((game, seat, request))
        while requests:
            packed = {
                name: tf.convert_to_tensor(
                    numpy.stack([r[2]["features"][name] for r in requests]),
                    dtype=spec["dtype"],
                )
                for name, spec in feature_vector.INPUTS.items()
            }
            outputs = {
                name: tensor.numpy() for name, tensor in predict(packed).items()
            }
            deferred = []
            for row, (game, seat, request) in enumerate(requests):
                candidates = actions.policy_candidates(request)
                head = feature_vector.TRAIN_POLICY_HEADS.index(candidates[0][0])
                legal = numpy.zeros(150, bool)
                codes = [candidate[1] for candidate in candidates]
                legal[codes] = True
                scores = outputs[candidates[0][0]][row, codes]
                scores = scores - scores.max()
                log_probability = numpy.full(150, -1e9, numpy.float32)
                log_probability[codes] = scores - numpy.log(
                    numpy.exp(scores).sum()
                )
                chosen = int(
                    generator.choice(150, p=numpy.exp(log_probability))
                )
                _, _, selected, metadata = candidates[codes.index(chosen)]
                samples.append(
                    dict(
                        features=request["features"],
                        head=head,
                        legal=legal,
                        chosen=chosen,
                        behavior=log_probability,
                        value=outputs["value"][row],
                        game=game,
                        seat=seat,
                    )
                )
                if selected is None:
                    deferred.append(
                        (game, seat, actions.continue_request(request))
                    )
                    continue
                choices[game][seat] = selected
                if "discard_action" in metadata and selected.action_type in (
                    riichienv.ActionType.RIICHI,
                    riichienv.ActionType.CHI,
                    riichienv.ActionType.PON,
                ):
                    contexts[game]["pending"][seat] = (
                        metadata["discard_action"],
                        features.action_event(selected, request["drawn_tile"])[
                            "type"
                        ],
                    )
            requests = deferred
        for game, env in enumerate(envs):
            if not env.done():
                observations[game] = env.step(choices[game])
    returns = (
        numpy.asarray(
            [
                outcomes.saint3_jade_south(env.scores(), env.ranks())
                for env in envs
            ],
            numpy.float32,
        )
        / 100.0
    )
    batch = {
        "features": {
            name: numpy.stack([sample["features"][name] for sample in samples])
            for name in feature_vector.INPUTS
        },
        **{
            name: numpy.asarray([sample[name] for sample in samples])
            for name in ("head", "legal", "chosen", "behavior", "value")
        },
        "returns": numpy.asarray(
            [returns[sample["game"], sample["seat"]] for sample in samples]
        ),
    }
    advantage = batch["returns"] - batch.pop("value")
    batch["advantages"] = (advantage - advantage.mean()) / max(
        advantage.std(), 1e-8
    )
    return batch


class Trainer:
    def __init__(self, network, value, args):
        self.network = network
        self.value = value
        self.args = args
        self.policy_optimizer = tf.keras.optimizers.Adam(args.learning_rate)
        self.auxiliary_optimizer = tf.keras.optimizers.Adam(args.learning_rate)
        self.policy_optimizer.build(
            network.actor.trainable_variables + value.trainable_variables
        )
        self.auxiliary_optimizer.build(network.trainable_variables)

    @tf.function(reduce_retracing=True)
    def predict(self, batch):
        return distribution(
            self.network.actor(batch["features"], training=False),
            batch["head"],
            batch["legal"],
        )

    @tf.function(reduce_retracing=True)
    def policy_update(self, batch):
        with tf.GradientTape() as tape:
            outputs = self.network.actor(batch["features"], training=False)
            current = distribution(outputs, batch["head"], batch["legal"])
            policy = ppo_loss(
                current=current,
                behavior=batch["behavior"],
                chosen=batch["chosen"],
                advantages=batch["advantages"],
                clip=self.args.clip,
            )
            value_loss = tf.reduce_mean(
                tf.square(self.value(outputs["board"])[:, 0] - batch["returns"])
            )
            entropy = -tf.reduce_mean(
                tf.reduce_sum(tf.exp(current) * current, axis=-1)
            )
            human_kl = forward_kl(batch["human"], current)
            loss = (
                policy
                + 0.5 * value_loss
                - self.args.entropy * entropy
                + self.args.human_kl * human_kl
            )
        variables = (
            self.network.actor.trainable_variables
            + self.value.trainable_variables
        )
        gradients = tape.gradient(loss, variables)
        # Actor also exposes auxiliary heads absent from this objective.
        pairs = [(g, v) for g, v in zip(gradients, variables) if g is not None]
        gradients, norm = tf.clip_by_global_norm([g for g, _ in pairs], 1.0)
        tf.debugging.assert_all_finite(norm, "Nonfinite PPO gradient")
        self.policy_optimizer.apply_gradients(
            zip(gradients, [v for _, v in pairs])
        )
        return {
            "policy": policy,
            "value": value_loss,
            "entropy": entropy,
            "human_kl": human_kl,
        }

    @tf.function(reduce_retracing=True)
    def auxiliary_update(self, inputs, anchor):
        with tf.GradientTape() as tape:
            outputs = self.network(inputs)
            with tape.stop_recording():
                advantages = critic_utility.advantages(
                    self.network,
                    inputs,
                    outputs,
                    {"temperature": self.args.awr_temperature},
                )
            auxiliary, _ = critic_full.objective_metrics(
                inputs,
                outputs,
                utility={
                    "enabled": tf.constant(True),
                    "mix": self.args.awr_mix,
                    "advantages": advantages,
                },
            )
            current = self.predict(anchor)
            kl = forward_kl(anchor["reference"], current)
            human_kl = forward_kl(anchor["human"], current)
            loss = (
                auxiliary
                + self.args.auxiliary_kl * kl
                + self.args.human_kl * human_kl
            )
        variables = self.network.trainable_variables
        gradients = tape.gradient(loss, variables)
        gradients, norm = tf.clip_by_global_norm(gradients, 1.0)
        tf.debugging.assert_all_finite(norm, "Nonfinite auxiliary gradient")
        self.auxiliary_optimizer.apply_gradients(zip(gradients, variables))
        return {"auxiliary": auxiliary, "snapshot_kl": kl}


def run(args):
    if not (
        min(
            (
                args.cycles,
                args.games,
                args.batch_size,
                args.ppo_epochs,
                args.auxiliary_updates,
            )
        )
        > 0
        and args.auxiliary_kl > 0
        and args.target_kl > 0
        and 0 < args.clip < 1
        and 0 <= args.awr_mix <= 1
        and args.awr_temperature > 0
        and args.learning_rate > 0
        and args.entropy >= 0
        and args.human_kl >= 0
    ):
        raise ValueError("Invalid phasic training settings")
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) > 1:
        raise ValueError(
            "Select one training GPU by UUID with CUDA_VISIBLE_DEVICES"
        )
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(args.seed)
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    plan = vars(args)
    if (root / "plan.json").exists():
        if json.loads((root / "plan.json").read_text()) != plan:
            raise ValueError("Changed settings require a new output directory")
    (root / "plan.json").write_text(json.dumps(plan, indent=2))
    stream = critic_data.dataset(
        root=args.data,
        sampling="mixed",
        partition="train",
        batch_size=args.batch_size,
        repeat=True,
        seed=args.seed,
    )
    first = critic_data.parse_batch(next(iter(stream)))
    network = critic.JointPolicyPayment()
    network(first)
    network.load_weights(args.initial)
    human_actor = critic.JointPolicyPayment().actor
    human_actor.set_weights(network.actor.get_weights())
    human_actor.trainable = False

    @tf.function(reduce_retracing=True)
    def human_predict(batch):
        return distribution(
            human_actor(batch["features"], training=False),
            batch["head"],
            batch["legal"],
        )

    value = tf.keras.layers.Dense(
        1, kernel_initializer="zeros", dtype="float32"
    )
    value(network.actor(first[0])["board"])
    network.compile_training()
    trainer = Trainer(network, value, args)
    cycle = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(
        network=network,
        value=value,
        cycle=cycle,
        policy_optimizer=trainer.policy_optimizer,
        auxiliary_optimizer=trainer.auxiliary_optimizer,
    )
    manager = tf.train.CheckpointManager(checkpoint, str(root / "recovery"), 2)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).assert_consumed()
    writer = tf.summary.create_file_writer(str(root / "tensorboard"))
    for cycle_number in range(int(cycle.numpy()), args.cycles):
        tf.keras.utils.set_random_seed(args.seed + cycle_number)
        (root / "status.json").write_text(
            json.dumps(
                {
                    "stage": "collecting",
                    "cycle": cycle_number + 1,
                    "target_cycles": args.cycles,
                }
            )
        )
        batch = collect(
            network.actor,
            value,
            args.games,
            args.seed + cycle_number * args.games,
        )
        batch["human"] = numpy.concatenate(
            [
                human_predict(part).numpy()
                for part in tf.data.Dataset.from_tensor_slices(batch).batch(
                    args.batch_size
                )
            ]
        )
        dataset = tf.data.Dataset.from_tensor_slices(batch)
        batches = dataset.batch(args.batch_size)
        (root / "status.json").write_text(
            json.dumps(
                {
                    "stage": "ppo",
                    "cycle": cycle_number + 1,
                    "decisions": len(batch["chosen"]),
                }
            )
        )
        for epoch in range(args.ppo_epochs):
            for minibatch in dataset.shuffle(
                len(batch["chosen"]), seed=args.seed + cycle_number + epoch
            ).batch(args.batch_size):
                policy_metrics = trainer.policy_update(minibatch)
            behavior_kl = sum(
                float(forward_kl(part["behavior"], trainer.predict(part)))
                * len(part["chosen"])
                for part in batches
            ) / len(batch["chosen"])
            if behavior_kl >= args.target_kl:
                break
        # Frozen outputs are the post-PPO snapshot, before any auxiliary step.
        batch["reference"] = numpy.concatenate(
            [trainer.predict(part).numpy() for part in batches]
        )
        anchors = iter(
            tf.data.Dataset.from_tensor_slices(batch)
            .shuffle(len(batch["chosen"]), seed=args.seed + cycle_number)
            .repeat()
            .batch(args.batch_size)
        )
        stream = iter(
            critic_data.dataset(
                root=args.data,
                sampling="mixed",
                partition="train",
                batch_size=args.batch_size,
                repeat=True,
                seed=args.seed + cycle_number,
            )
        )
        (root / "status.json").write_text(
            json.dumps(
                {
                    "stage": "auxiliary",
                    "cycle": cycle_number + 1,
                    "behavior_kl": behavior_kl,
                }
            )
        )
        for _ in range(args.auxiliary_updates):
            auxiliary_metrics = trainer.auxiliary_update(
                critic_data.parse_batch(next(stream)), next(anchors)
            )
        snapshot_kl = sum(
            float(forward_kl(part["reference"], trainer.predict(part)))
            * len(part["chosen"])
            for part in tf.data.Dataset.from_tensor_slices(batch).batch(
                args.batch_size
            )
        ) / len(batch["chosen"])
        cycle.assign_add(1)
        manager.save(checkpoint_number=int(cycle.numpy()))
        metrics = {
            **{name: float(v) for name, v in policy_metrics.items()},
            **{name: float(v) for name, v in auxiliary_metrics.items()},
            "behavior_kl": behavior_kl,
            "snapshot_kl": snapshot_kl,
            "ppo_epochs": epoch + 1,
            "decisions": len(batch["chosen"]),
            "cycle": int(cycle.numpy()),
        }
        with writer.as_default():
            for name, metric in metrics.items():
                tf.summary.scalar(name, metric, step=int(cycle.numpy()))
        writer.flush()
        (root / "status.json").write_text(
            json.dumps(
                {
                    "stage": "cycle_complete",
                    **metrics,
                },
                indent=2,
            )
        )
        print(metrics, flush=True)
    network.save_weights(root / "final.weights.h5")
    (root / "status.json").write_text(
        json.dumps(
            {
                **json.loads((root / "status.json").read_text()),
                "stage": "complete",
                "cycle": int(cycle.numpy()),
            },
            indent=2,
        )
    )
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--initial", required=True, help="Joint weights")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--auxiliary-updates", type=int, default=32)
    parser.add_argument("--auxiliary-kl", type=float, default=1.0)
    parser.add_argument("--human-kl", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--awr-mix", type=float, default=1.0)
    parser.add_argument("--awr-temperature", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=941)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
