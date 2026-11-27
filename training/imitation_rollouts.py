"""Balanced focused and natural human roots for hand continuations."""

import json

import numpy
import tensorflow as tf

from log_dataset import call_states, records, training_targets
from model import actions
from model import actor as actor_layers
from model import critic
from training import restoration, self_play
from training.corrections import collection, rollouts


def select_seeds(
    *, source, batch, predict, evaluate, seed, decisions_per_continuation, stop
):
    """Select compact balanced starts without simulating any hands."""
    count = int(batch["local_count"])
    if not count:
        return []
    observed = call_states.select(
        tf.nest.map_structure(lambda value: value.numpy(), batch["features"]),
        numpy.arange(count),
    )
    metadata = {
        name: value.numpy()[:count] for name, value in batch["metadata"].items()
    }
    if any(
        records.game_partition(game_id.decode()) >= 7
        for game_id in metadata["game_id"]
    ):
        raise ValueError("Imitation rollouts cannot use held-out games")
    heads = batch["head"].numpy()[:count]
    legal = batch["legal"].numpy()[:count]
    eligible = restoration.trainable_decisions(observed, heads, legal).numpy()
    natural = numpy.flatnonzero(eligible & batch["natural"].numpy()[:count])
    if not len(natural):
        return []
    generator = numpy.random.default_rng(seed)
    candidates = generator.permutation(numpy.flatnonzero(eligible))
    selected = []
    continuations = 0
    # A random order of all eligible rows induces a uniform random order of
    # the qualifying subset. Score only enough chunks to fill this budget.
    for begin in range(0, len(candidates), 32):
        if continuations * decisions_per_continuation >= count:
            break
        rows = candidates[begin : begin + 32]
        inputs = call_states.select(observed, rows)
        policy = self_play.policy_log_probabilities(
            predict(inputs), heads[rows], legal[rows]
        )
        preferred = actor_layers.greedy_choices(policy, heads[rows]).numpy()
        values = numpy.zeros((len(rows), 37), numpy.float32)
        discards = numpy.flatnonzero(heads[rows] == 0)
        if len(discards):
            values[discards] = evaluate(call_states.select(inputs, discards))
        for local, row in enumerate(rows):
            if continuations * decisions_per_continuation >= count:
                break
            if stop.exists():
                return []
            witness = source.witness(
                game_id=metadata["game_id"][row].decode(),
                kyoku_id=metadata["kyoku_id"][row].decode(),
                decision_index=int(metadata["decision_index"][row]),
                actor=int(metadata["actor"][row]),
                phase=int(observed["decision_phase"][row]),
            )
            choices, categories = collection.targeted_actions(
                features={
                    name: value[local]
                    for name, value in call_states.base(inputs).items()
                },
                head=int(heads[row]),
                legal=legal[row],
                policy=policy[local],
                preferred=int(preferred[local]),
                values=values[local],
                sakigiri_actions=numpy.asarray(
                    witness[0] if witness is not None else (), dtype=numpy.int32
                ),
            )
            if not choices:
                continue
            selected.extend(
                [
                    (
                        int(row),
                        {
                            "actions": choices,
                            "categories": categories,
                            "kind": "targeted",
                            "seed": seed + int(row),
                        },
                    ),
                    (
                        int(generator.choice(natural)),
                        {
                            "actions": [],
                            "categories": [],
                            "kind": "normal",
                            "seed": seed + count + len(selected),
                        },
                    ),
                ]
            )
            continuations += len(choices) + 2
    if not selected:
        return []
    normal = numpy.array([row for row, _ in selected[1::2]])
    policy = self_play.policy_log_probabilities(
        predict(call_states.select(observed, normal)),
        heads[normal],
        legal[normal],
    )
    for local, (row, selection) in enumerate(selected[1::2]):
        codes = numpy.flatnonzero(legal[row])
        probability = numpy.exp(policy[local, codes].astype(numpy.float64))
        selection["actions"] = generator.choice(
            codes, size=2, p=probability / probability.sum()
        ).tolist()
    return [
        {
            "game_id": metadata["game_id"][position].decode(),
            "kyoku_id": metadata["kyoku_id"][position].decode(),
            "decision_index": int(metadata["decision_index"][position]),
            "actor": int(metadata["actor"][position]),
            "phase": int(observed["decision_phase"][position]),
            "head": int(heads[position]),
            **selection,
            "affinity": training_targets.candidate_yaku_affinity(
                {
                    name: value[position]
                    for name, value in call_states.base(observed).items()
                }
            ),
        }
        for position, selection in selected
    ]


def collect(
    *,
    source,
    seeds,
    decision_target,
    predict,
    packing,
    directory,
    stop,
    reference_policy
):
    """Run the accumulated starts together, with one continuation/action."""
    statistics = {
        "roots": 0,
        "continuations": 0,
        "simulated_decisions": 0,
        "decision_target": decision_target,
        "targeted_roots": 0,
        "normal_roots": 0,
    }
    receipt = []
    reference_logs = []
    actions_in_order = []

    def streams():
        for selection in seeds:
            if stop.exists():
                return
            root = source.root(
                **{
                    name: selection[name]
                    for name in (
                        "game_id",
                        "kyoku_id",
                        "decision_index",
                        "actor",
                        "phase",
                    )
                }
            )
            reference_logs.append(json.loads(root.dumps())["game"]["logs"])
            wall_seeds = [
                selection["seed"]
                + (branch if selection["kind"] == "normal" else 0)
                for branch in range(len(selection["actions"]))
            ]
            receipt.append(
                {
                    name: value
                    for name, value in selection.items()
                    if name not in ("affinity", "seed")
                }
                | {"wall_seeds": [[start] for start in wall_seeds]}
            )
            statistics[selection["kind"] + "_roots"] += 1
            for category in selection["categories"]:
                statistics[category] = statistics.get(category, 0) + 1
            for action, wall_seed in zip(selection["actions"], wall_seeds):
                actions_in_order.append(action)
                yield self_play.collection_steps(
                    games=1,
                    seed=wall_seed,
                    collector=root.fork(1, wall_seed, until_hand_end=True),
                    forced_root=(root.decision, action),
                    with_labels=True,
                    root_only=True,
                    root_affinity=selection["affinity"],
                )

    completed = self_play.collect_streams(predict, streams(), max_active=128)
    statistics["simulated_decisions"] = sum(
        counts["simulated_decisions"] for _, counts in completed
    )
    retained = []
    for action, (rollout, _) in zip(actions_in_order, completed):
        assert len(rollout["chosen"]) == 1
        assert (rollout["chosen"] == action).all()
        retained.append(
            {
                "features": call_states.pack_numpy(
                    rollout["features"], packing
                ),
                "labels": rollout["labels"],
                "head": rollout["head"],
                "legal": rollout["legal"],
                "chosen": critic.chosen_actions(
                    rollout["features"],
                    {
                        name: numpy.where(
                            rollout["head"] == head, rollout["chosen"], -1
                        )
                        for head, name in enumerate(actions.TRAIN_POLICY_HEADS)
                    },
                ).numpy(),
                "payment_points": rollout["payment_points"],
            }
        )
    statistics["roots"] = len(receipt)
    statistics["continuations"] = sum(len(row["actions"]) for row in receipt)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "roots.json").write_text(json.dumps(receipt) + "\n")
    (directory / "statistics.json").write_text(json.dumps(statistics) + "\n")
    if not retained:
        return None, statistics
    rows = rollouts.concatenate(retained)
    offsets = numpy.cumsum([0] + [len(row["actions"]) for row in receipt])
    probabilities = reference_policy.distribution_many(
        [
            (
                events,
                [(len(events) - 1, selection["actor"], selection["phase"])],
                rows["head"][offset : offset + 1],
                rows["legal"][offset : offset + 1],
            )
            for events, selection, offset in zip(
                reference_logs, receipt, offsets[:-1]
            )
        ]
    )
    rows["reference_log_probability"] = numpy.repeat(
        probabilities, numpy.diff(offsets), axis=0
    )
    numpy.savez_compressed(
        file=directory / "outcomes.npz",
        **{
            "features/" + name: value
            for name, value in call_states.flatten(rows["features"]).items()
        },
        **{"label/" + name: value for name, value in rows["labels"].items()},
        **{
            name: value
            for name, value in rows.items()
            if name not in ("features", "labels")
        },
    )
    return rows, statistics


def batches(rows, human, group, batch_size):
    """Equal update counts across ranks, with zero-mass empty-rank padding."""
    count = 0 if rows is None else len(rows["chosen"])
    local_size = (batch_size + group.replicas - 1) // group.replicas
    maximum = int(group.maximum(tf.constant(count, tf.int32)))
    if rows is None:
        rows = {
            "reference_log_probability": numpy.zeros((1, 150), numpy.float32),
            "features": call_states.pack_numpy(
                call_states.select(
                    tf.nest.map_structure(
                        lambda value: value.numpy(), human["features"]
                    ),
                    [0],
                ),
                group.packing,
            ),
            **{
                name: tf.nest.map_structure(
                    lambda value: value.numpy()[:1], human[name]
                )
                for name in (
                    "head",
                    "legal",
                    "chosen",
                    "payment_points",
                    "labels",
                )
            },
        }
    for begin in range(0, maximum, local_size):
        selected = rollouts.select(
            rows,
            numpy.minimum(
                numpy.arange(begin, begin + local_size), max(count, 1) - 1
            ),
        )
        yield {
            **rollouts.unpack(selected, group.packing, None),
            "local_count": tf.constant(max(0, min(local_size, count - begin))),
            "global_count": tf.constant(0),
        }
