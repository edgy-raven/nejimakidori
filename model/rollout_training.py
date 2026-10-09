"""Budgeted settlement replay for joint training; payoff stays derived."""

import collections
import json
import math
import time

import numpy
import tensorflow as tf

from log_dataset import records, rollout_store
from model import critic, critic_utility, joint_settlement, objectives, outcomes


def win_advantage(differences):
    """Conservative paired win gap; ties carry no preference.

    A 95% Wilson interval on discordant trials shrinks their direction toward
    zero. Scale by their frequency among all trials, including tied outcomes.
    This is conditional evidence under the recorded hands and frozen policy.
    """
    count = sum(value != 0 for value in differences)
    if not count:
        return 0.0
    rate = sum(value > 0 for value in differences) / count
    center = (rate + 1.96**2 / (2 * count)) / (1 + 1.96**2 / count)
    radius = (
        1.96
        * math.sqrt(rate * (1 - rate) / count + 1.96**2 / (4 * count**2))
        / (1 + 1.96**2 / count)
    )
    return (
        (max(2 * (center - radius) - 1, 0) + min(2 * (center + radius) - 1, 0))
        * count
        / len(differences)
    )


class Replay:
    """Uniform positions, then policies and paired trials; screen split only."""

    def __init__(self, path, fraction=0.1, stale_weight=0.5):
        if not 0 <= stale_weight <= 1:
            raise ValueError("Stale rollout weight must be in [0, 1]")
        self.budget = Budget(fraction)
        started = time.monotonic()
        self.store = rollout_store.RolloutStore(path)
        self.positions = {}
        for identity, position in self.store.connection.execute(
            "SELECT DISTINCT roots.id, position FROM roots JOIN samples "
            "ON roots.id=samples.root WHERE stage='screen' ORDER BY roots.id"
        ):
            state, features = self.store.root(identity)
            if not rollout_store.eligible(
                features,
                disagreement="disagreement" in state and state["disagreement"],
            ):
                continue
            # Recorded games share the corpus split. Generated historical
            # roots without a source game ID retain their start-state split.
            partition = (
                records.game_partition(state["game_id"])
                if "game_id" in state
                else int(
                    rollout_store.fingerprint(
                        next(
                            row
                            for row in state["events"]
                            if row["type"] == "start_kyoku"
                        )
                    )[:8],
                    16,
                )
                % 10
            )
            if partition >= 7:
                continue
            rows = self.store.connection.execute(
                "SELECT actions, outcomes FROM samples WHERE root=? "
                "AND stage='screen' ORDER BY actions, seed, sample",
                (identity,),
            ).fetchall()
            order = (state["actor"] + numpy.arange(4)) % 4
            rows = [
                (json.loads(actions), json.loads(pair))
                for actions, pair in rows
            ]
            differences = collections.defaultdict(list)
            for actions, pair in rows:
                wins = [
                    any(
                        event["type"] == "hora"
                        and event["actor"] == state["actor"]
                        for event in row["events"]
                    )
                    for row in pair
                ]
                differences[tuple(actions)].append(int(wins[0]) - wins[1])
            advantages = {
                actions: win_advantage(values)
                for actions, values in differences.items()
            }
            self.positions.setdefault(position, []).append(
                {
                    "features": features,
                    "rows": [
                        (
                            numpy.array(actions, numpy.int32),
                            numpy.array(
                                [
                                    numpy.array(row["payments"])[:, order][
                                        :, :, order
                                    ][outcomes.PAYMENT_CELLS]
                                    for row in pair
                                ],
                                numpy.int32,
                            ),
                            numpy.float32(advantages[tuple(actions)]),
                        )
                        for actions, pair in rows
                    ],
                }
            )
        self.positions = list(self.positions.values())
        if not self.positions:
            self.close()
            raise ValueError("Rollout store has no eligible training positions")
        self.budget.replay_seconds += time.monotonic() - started

    def close(self):
        self.store.close()

    def distributed_batch(self, strategy, step):
        if not self.budget.ready():
            return None
        batches = self.batch(step, strategy.num_replicas_in_sync)
        return strategy.experimental_distribute_values_from_function(
            lambda ctx: tf.nest.map_structure(
                tf.convert_to_tensor, batches[ctx.replica_id_in_sync_group]
            )
        )

    def batch(self, step, replicas):
        generator = numpy.random.default_rng(941 + step)
        batches = []
        for _ in range(replicas):
            policies = self.positions[generator.integers(len(self.positions))]
            root = policies[generator.integers(len(policies))]
            actions, payments, advantage = root["rows"][
                generator.integers(len(root["rows"]))
            ]
            batches.append(
                (
                    {
                        name: value[None]
                        for name, value in root["features"].items()
                    },
                    actions,
                    {
                        "payment_points": payments,
                        "win_advantage": advantage,
                        # These immutable rollout policies precede the moving
                        # training actor, even if initialized from its weights.
                        "stale": numpy.ones(2, numpy.bool_),
                    },
                )
            )
        return batches


class Budget:
    """Charge setup and entire replay batches against ordinary batch seconds."""

    def __init__(self, fraction=0.1):
        if not 0 < fraction <= 0.1:
            raise ValueError("Rollout overhead fraction must be in (0, 0.1]")
        self.fraction = fraction
        self.normal_seconds = 0.0
        self.replay_seconds = 0.0
        self.reserve_seconds = 30.0
        self.updates = 0

    def ready(self):
        return (
            self.replay_seconds + self.reserve_seconds
            <= self.fraction * self.normal_seconds
        )

    def record(self, elapsed, replay):
        if replay:
            self.replay_seconds += elapsed
            self.reserve_seconds = max(self.reserve_seconds, 2 * elapsed)
            self.updates += 1
        else:
            self.normal_seconds += elapsed
            if self.updates == 0:
                self.reserve_seconds = max(self.reserve_seconds, 2 * elapsed)

    def metrics(self):
        return {
            "rollout_normal_seconds": self.normal_seconds,
            "rollout_charged_seconds": self.replay_seconds,
            "rollout_budget_fraction": self.fraction,
            "rollout_updates": self.updates,
        }


def candidate_inputs(network, batch):
    if batch is None:
        return None
    features, actions, _ = batch
    encoded, state = network.candidate_context(
        features, network.encoder(features)
    )
    matches = encoded["selected"][:, 1, None] == actions[None]
    tf.debugging.assert_equal(
        tf.reduce_sum(tf.cast(matches, tf.int32), 0), [1, 1]
    )
    selected = tf.argmax(tf.cast(matches, tf.int32), 0, output_type=tf.int32)
    public = {
        name: tf.repeat(value, 2, axis=0)
        for name, value in joint_settlement.public_inputs(features).items()
    }
    matches = (
        tf.cast(batch[2]["payment_points"][..., None], tf.float32)
        == joint_settlement.payment_values(public)[:, None]
    )
    tf.debugging.assert_equal(tf.reduce_all(tf.reduce_any(matches, -1)), True)
    return {
        **critic.public_inputs(
            features, tf.gather(encoded["selected"], selected)
        ),
        "labels": {
            **batch[2],
            "payment": tf.argmax(
                tf.cast(matches, tf.int32), -1, output_type=tf.int32
            ),
        },
        "state": tf.stop_gradient(tf.gather(state, selected)),
        "immediate_safe": tf.gather(encoded["immediate_safe"], selected),
        "immediate_wait": tf.stop_gradient(
            tf.gather(encoded["immediate_wait"], selected)
        ),
        "immediate_allowed": tf.gather(
            critic.immediate_decisions(features, encoded["selected"]), selected
        ),
        **public,
    }


def loss(predictor, inputs, stale_weight):
    if inputs is None:
        return 0.0, {}

    def mean(values, mask):
        weight = tf.where(inputs["labels"]["stale"], stale_weight, 1.0)
        if values.shape.rank == 2:
            weight = weight[:, None]
        # Keep the unweighted denominator: stale-only batches are downweighted.
        return objectives.mean_valid(values * weight, mask)

    metrics = joint_settlement.observed_terms(
        {**inputs["labels"], "pot": inputs["pot"]},
        {**inputs, **predictor.distributions(inputs)},
        mean=mean,
    )
    metrics["stale_fraction"] = tf.reduce_mean(
        tf.cast(inputs["labels"]["stale"], tf.float32)
    )
    return (
        metrics["payment_nll"],
        {
            ("rollout_weighted_" if name == "payment_nll" else "rollout_")
            + name: value
            for name, value in metrics.items()
        },
    )


def policy_loss(actor, batch, stale_weight):
    """Bounded expected win regret within a nominated pair, not money regret."""
    if batch is None:
        return 0.0, {}
    features, actions, labels = batch
    probabilities = tf.gather(
        critic_utility.candidate_probabilities(features, actor(features))[0],
        actions,
    )
    probabilities = tf.math.divide_no_nan(
        probabilities, tf.reduce_sum(probabilities)
    )
    regret = tf.maximum(labels["win_advantage"], 0.0) * probabilities[1]
    regret += tf.maximum(-labels["win_advantage"], 0.0) * probabilities[0]
    regret *= tf.reduce_mean(tf.where(labels["stale"], stale_weight, 1.0))
    return regret, {
        "rollout_policy_win_regret": regret,
        "rollout_pair_win_gap": tf.abs(labels["win_advantage"]),
    }
