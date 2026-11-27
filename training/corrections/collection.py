"""Paired current-hand outcomes at selected frozen-policy decisions."""

import numpy
import tensorflow as tf

from log_dataset import (
    _native_collection,
    call_states,
    focus_sampling,
    training_data,
)
from model import actor as actor_layers
from model import critic, outcomes
from training import self_play, trainer

CATEGORIES = (
    "disagreement",
    "tenpai_threat",
    "keiten",
    "calls",
    "riichi",
    "sakigiri",
)


def root_scorer(network):
    """Return discard Q values, with illegal choices excluded from max-Q."""
    specs = {
        name: tf.TensorSpec([None, *spec["shape"]], spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    packing = training_data.PackedBatch(specs)

    @tf.function(reduce_retracing=True)
    def predict(inputs):
        features = call_states.unpack(inputs, packing)
        outputs = network.actor(features, training=False)
        state = network.candidate_context(features, outputs)
        selected = critic.active_candidates(features, outputs["selected"])
        values = trainer.action_values(
            network=network,
            features=features,
            selected=tf.boolean_mask(outputs["selected"], selected),
            state=tf.boolean_mask(state, selected),
            chunk_size=32,
        )[:, :37]
        return tf.where(
            tf.cast(features["discard_action_legal"], tf.bool),
            values,
            -float("inf"),
        )

    def evaluate(features):
        # Stable encoder shapes avoid recompiling for each sampled root count.
        count = len(features["decision_phase"])
        # Small self-play requests need small buckets, not 64 padded roots.
        chunk_size = min(64, 1 << (count - 1).bit_length())
        outputs = []
        for begin in range(0, count, chunk_size):
            indices = numpy.minimum(
                numpy.arange(begin, begin + chunk_size), count - 1
            )
            with tf.device("/CPU:0"):
                inputs = tf.nest.map_structure(
                    tf.convert_to_tensor,
                    call_states.pack_numpy(
                        call_states.select(features, indices), packing
                    ),
                )
            outputs.append(
                predict(inputs).numpy()[: min(chunk_size, count - begin)]
            )
        return numpy.concatenate(outputs)

    return evaluate


def action_pair(features, head, legal, policy, category, preferred):
    """Compare the policy choice with a publicly identifiable alternative."""
    codes = numpy.flatnonzero(legal)
    if category == "calls":
        if head not in (2, 3):
            return None
        improving = self_play.improving_call_choices(features)[head - 2]
        alternatives = codes[improving[codes]]
        if preferred != 0 and not improving[preferred]:
            return None
        if preferred != 0:
            return preferred, 0
    elif head not in (0, 1):
        return None
    elif category == "disagreement":
        # The discard-only critic scorer resolves this proposal after mining.
        return (preferred, -1) if head == 0 else None
    elif category in ("tenpai_threat", "keiten"):
        distance = min(
            int(features[name])
            for name in (
                "normal_self_shanten",
                "chiitoitsu_self_shanten",
                "kokushi_self_shanten",
            )
        )
        if category == "tenpai_threat" and (
            distance > 1
            or not any(
                focus_sampling.visible_threat(features, seat)
                for seat in range(1, 4)
            )
        ):
            return None
        if category == "keiten" and (
            distance > 2 or features["live_wall_count"] > 16
        ):
            return None
        shanten = features["discard_action_all_shanten"][codes % 37]
        keep = shanten == shanten.min()
        alternatives = codes[keep != keep[codes == preferred][0]]
    elif category == "riichi":
        if head != 1:
            return None
        alternatives = codes[(codes >= 37) != (preferred >= 37)]
        counterpart = preferred + (37 if preferred < 37 else -37)
        if counterpart in alternatives:
            return preferred, counterpart
    else:
        raise ValueError(f"Unknown correction category: {category}")
    if not len(alternatives):
        return None
    return preferred, int(alternatives[policy[alternatives].argmax()])


def snapshot(collector, batch, row, actions):
    """Own the decision buffers retained beyond the current collector step."""
    return {
        "snapshot": collector.snapshot(row),
        "features": call_states.select(batch["features"], [row]),
        "head": int(batch["head"][row]),
        "legal": batch["legal"][row].copy(),
        "actions": actions,
        "source": batch["indices"][row].copy(),
    }


def targeted_actions(
    *, features, head, legal, policy, preferred, values, sakigiri_actions
):
    """Union existing categories, exhaustively retaining advancing calls."""
    selected = set()
    categories = []
    for category in CATEGORIES:
        if category == "calls":
            if head not in (2, 3):
                continue
            improving = self_play.improving_call_choices(features)[head - 2]
            codes = numpy.flatnonzero(legal)
            alternatives = codes[improving[codes]]
            pair = [0, *alternatives] if len(alternatives) else None
        elif category == "sakigiri":
            pair = (
                sakigiri_actions
                if head == 0 and len(sakigiri_actions)
                else None
            )
        elif category == "disagreement":
            if head != 0:
                continue
            best = int(values.argmax())
            pair = (
                (preferred, best)
                if best != preferred
                and values[best] - values[preferred]
                >= 0.25 / outcomes.OBJECTIVE_ST3_SCALE
                else None
            )
        else:
            pair = action_pair(
                features=features,
                head=head,
                legal=legal,
                policy=policy,
                category=category,
                preferred=preferred,
            )
        if pair is not None:
            assert legal[list(pair)].all()
            selected.update(int(action) for action in pair)
            categories.append(category)
    return sorted(selected), categories


def collect(
    predict, root_evaluate, games, seed, *, retain, discover, reference_policy
):
    """Balance targeted starts with uniform normal starts from the same games.

    A category is assigned before mining. Normal reservoirs include every
    recoverable decision from all four seats, regardless of its category.
    Each targeted root has one normal root from the same game. Both get two
    continuations, and every rollout decision contributes to MMD.
    """
    generator = numpy.random.default_rng(seed + 1000000000)
    categories = (numpy.arange(games) + seed) % len(CATEGORIES)
    generator.shuffle(categories)
    proposals = {}
    normal = {}
    ordinary_seen = numpy.zeros(games, numpy.int64)
    seen = numpy.zeros(games, numpy.int64)

    def capture(collector, batch):
        with tf.device("/CPU:0"):
            preferred = actor_layers.greedy_choices(
                batch["behavior"], batch["head"]
            ).numpy()
        for row, game in enumerate(batch["indices"][:, 0]):
            ordinary_seen[game] += 1
            if generator.integers(ordinary_seen[game]) == 0:
                legal = numpy.flatnonzero(batch["legal"][row])
                probability = numpy.exp(
                    batch["behavior"][row, legal].astype(numpy.float64)
                )
                normal[int(game)] = snapshot(
                    collector,
                    batch,
                    row,
                    generator.choice(
                        legal, size=2, p=probability / probability.sum()
                    ),
                )
            if CATEGORIES[categories[game]] == "sakigiri":
                actions = batch["sakigiri_actions"][row]
                if batch["head"][row] != 0 or not len(actions):
                    continue
                pair = (int(generator.choice(actions[:-1])), int(actions[-1]))
                assert batch["legal"][row, list(pair)].all()
            else:
                pair = action_pair(
                    features={
                        name: value[row]
                        for name, value in call_states.base(
                            batch["features"]
                        ).items()
                    },
                    head=int(batch["head"][row]),
                    legal=batch["legal"][row],
                    policy=batch["behavior"][row],
                    category=CATEGORIES[categories[game]],
                    preferred=int(preferred[row]),
                )
            if pair is None:
                continue
            seen[game] += 1
            capacity = 8 if categories[game] == 0 else 1
            slot = (
                int(seen[game]) - 1
                if seen[game] <= capacity
                else int(generator.integers(seen[game]))
            )
            if slot >= capacity:
                continue
            root = snapshot(collector, batch, row, pair)
            retained = proposals.setdefault(int(game), [])
            if slot == len(retained):
                retained.append(root)
            else:
                retained[slot] = root

    statistics = discover(capture)
    roots = {
        game: retained[0]
        for game, retained in proposals.items()
        if categories[game] != 0
    }
    disagreement = [
        (game, root)
        for game in sorted(proposals)
        if categories[game] == 0
        for root in proposals[game]
    ]
    qualified = numpy.zeros(games, numpy.int64)
    if disagreement:
        features = call_states.concatenate(
            [root["features"] for _, root in disagreement]
        )
        values = root_evaluate(features)
        for row, (game, root) in enumerate(disagreement):
            preferred = root["actions"][0]
            best = int(values[row].argmax())
            gap = values[row, best] - values[row, preferred]
            if best == preferred or gap < 0.25 / outcomes.OBJECTIVE_ST3_SCALE:
                continue
            assert root["legal"][[preferred, best]].all()
            qualified[game] += 1
            if generator.integers(qualified[game]) == 0:
                root["actions"] = (preferred, best)
                roots[game] = root
    statistics["disagreement_scored"] = len(disagreement)
    statistics["disagreement_qualifying"] = int(qualified.sum())

    selected = sorted(roots)
    starts = [
        (game, kind, root)
        for game in selected
        for kind, root in (("targeted", roots[game]), ("normal", normal[game]))
    ]
    streams = []
    for game, kind, root in starts:
        for branch, action in enumerate(root["actions"]):
            # Targeted alternatives share a wall. Normal starts are independent
            # policy samples, including when they choose the same root action.
            continuation_seed = (
                seed
                + 1000000000
                + game
                + (games * (branch + 1) if kind == "normal" else 0)
            )
            streams.append(
                self_play.collection_steps(
                    games=1,
                    seed=continuation_seed,
                    collector=root["snapshot"].fork(
                        1, continuation_seed, until_hand_end=True
                    ),
                    forced_root=(
                        root["snapshot"].decision,
                        int(action),
                    ),
                    with_labels=True,
                    reference_policy=reference_policy,
                )
            )
    payments = []
    reference_probabilities = []
    for branch, (batch, _) in enumerate(
        self_play.collect_streams(predict, streams)
    ):
        row = numpy.flatnonzero(batch["paired_root"])
        assert len(row) == 1
        _, kind, root = starts[branch // 2]
        assert batch["chosen"][row[0]] == root["actions"][branch % 2]
        assert batch["head"][row[0]] == root["head"]
        retain(batch, kind)
        payments.append(batch["payment_points"][row[0]])
        if branch % 2 == 0:
            reference_probabilities.append(
                batch["reference_log_probability"][row[0]]
            )

    features = call_states.concatenate(
        [root["features"] for _, _, root in starts]
    )
    for category, name in enumerate(CATEGORIES):
        statistics[f"assigned/{name}"] = int((categories == category).sum())
        statistics[f"eligible/{name}"] = int(seen[categories == category].sum())
        statistics[f"selected/{name}"] = int(
            sum(categories[game] == category for game in selected)
        )
    statistics["selected_roots"] = len(starts)
    statistics["targeted_roots"] = len(selected)
    statistics["normal_roots"] = len(selected)
    statistics["root_continuations"] = 2 * len(starts)
    return {
        "features": features,
        "head": numpy.array(
            [root["head"] for _, _, root in starts], dtype=numpy.int32
        ),
        "reference_log_probability": numpy.asarray(
            reference_probabilities, numpy.float32
        ).reshape(-1, 150),
        "legal": numpy.array(
            [root["legal"] for _, _, root in starts], dtype=bool
        ).reshape(-1, 150),
        "actions": numpy.array(
            [root["actions"] for _, _, root in starts], dtype=numpy.int32
        ).reshape(-1, 2),
        "payment_points": numpy.array(payments, dtype=numpy.int32).reshape(
            -1, 2, int(outcomes.PAYMENT_CELLS.sum())
        ),
        "category": numpy.array(
            [
                categories[game] if kind == "targeted" else -1
                for game, kind, _ in starts
            ],
            dtype=numpy.int8,
        ),
        "source": numpy.array(
            [root["source"] for _, _, root in starts], dtype=numpy.int64
        ).reshape(-1, 4),
    }, statistics
