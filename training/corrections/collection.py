"""Paired current-hand outcomes at selected frozen-policy decisions."""

import numpy
import tensorflow as tf

from log_dataset import _native_collection, training_data
from model import actor as actor_layers
from model import critic, outcomes
from training import self_play, trainer

CATEGORIES = (
    "disagreement",
    "tenpai",
    "keiten",
    "calls",
    "riichi",
    "threat",
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
        features = packing.unpack(inputs)
        outputs, state = network.candidate_context(
            features, network.actor(features, training=False)
        )
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
        packed = packing.pack_numpy(features)
        count = len(features["decision_phase"])
        # Small self-play requests need small buckets, not 64 padded roots.
        chunk_size = min(64, 1 << (count - 1).bit_length())
        outputs = []
        for begin in range(0, count, chunk_size):
            indices = numpy.minimum(
                numpy.arange(begin, begin + chunk_size), count - 1
            )
            with tf.device("/CPU:0"):
                inputs = {
                    name: tf.convert_to_tensor(value[indices])
                    for name, value in packed.items()
                }
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
    elif category in ("tenpai", "keiten"):
        distance = min(
            int(features[name])
            for name in (
                "normal_self_shanten",
                "chiitoitsu_self_shanten",
                "kokushi_self_shanten",
            )
        )
        if category == "tenpai" and distance > 1:
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
    elif category == "threat":
        threatening = (features["riichi_state"][1:] > 0) | (
            (
                (features["meld_valid"][1:] > 0)
                & (features["meld_type"][1:] != 3)
            ).sum(axis=1)
            >= 2
        )
        if not threatening.any():
            return None
        tiles = numpy.take(numpy.r_[numpy.arange(34), 4, 13, 22], codes % 37)
        safety = features["genbutsu_to_seat"][1:][threatening][:, tiles].sum(0)
        preferred_safety = safety[codes == preferred][0]
        alternatives = codes[
            (
                safety == safety.max()
                if preferred_safety < safety.max()
                else safety < preferred_safety
            )
        ]
    else:
        raise ValueError(f"Unknown correction category: {category}")
    if not len(alternatives):
        return None
    return preferred, int(alternatives[policy[alternatives].argmax()])


def snapshot(collector, batch, row, actions):
    """Own the decision buffers retained beyond the current collector step."""
    return {
        "snapshot": collector.snapshot(row),
        "features": batch["features"][row].copy(),
        "head": int(batch["head"][row]),
        "legal": batch["legal"][row].copy(),
        "actions": actions,
        "source": batch["indices"][row].copy(),
    }


def collect(predict, root_evaluate, games, seed, *, retain, discover):
    """Balance targeted starts with uniform normal starts from the same games.

    A category is assigned before mining. Normal reservoirs include every
    recoverable decision from all four seats, regardless of its category.
    Each targeted root is accompanied by one normal root from that game. Both get
    two continuations, and every rollout decision contributes to MMD.
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
                pair = batch["sakigiri_pair"][row]
                if batch["head"][row] != 0 or (pair < 0).any():
                    continue
                assert batch["legal"][row, pair].all()
            else:
                pair = action_pair(
                    features=batch["features"][row],
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
        features = numpy.array(
            [root["features"] for _, root in disagreement],
            dtype=self_play.FEATURE_DTYPE,
        )
        values = root_evaluate(
            {name: features[name] for name in features.dtype.names}
        )
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
                    greedy=False,
                )
            )
    payments = []
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

    features = numpy.array(
        [root["features"] for _, _, root in starts],
        dtype=self_play.FEATURE_DTYPE,
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
        "features": {name: features[name] for name in features.dtype.names},
        "head": numpy.array(
            [root["head"] for _, _, root in starts], dtype=numpy.int32
        ),
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
