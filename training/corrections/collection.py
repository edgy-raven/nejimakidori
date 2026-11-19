"""Paired current-hand outcomes at selected frozen-policy decisions."""

import numpy
import tensorflow as tf
from model import actor as actor_layers
from model import outcomes
from training import self_play

CATEGORIES = ("disagreement", "tenpai", "keiten", "calls", "riichi", "threat")


def action_pair(features, head, legal, policy, category, preferred):
    """Compare the policy choice with a publicly identifiable alternative."""
    codes = numpy.flatnonzero(legal)
    if category == "calls":
        if head not in (2, 3):
            return None
        improving = self_play.improving_call_choices(features)[head - 2]
        calls = codes[improving[codes]]
        if not len(calls) or (preferred != 0 and not improving[preferred]):
            return None
        alternative = (
            int(calls[policy[calls].argmax()]) if preferred == 0 else 0
        )
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
        if not len(alternatives):
            return None
        alternative = int(alternatives[policy[alternatives].argmax()])
    elif category == "riichi":
        if head != 1:
            return None
        alternatives = codes[(codes >= 37) != (preferred >= 37)]
        if not len(alternatives):
            return None
        counterpart = preferred + (37 if preferred < 37 else -37)
        alternative = (
            counterpart
            if counterpart in alternatives
            else int(alternatives[policy[alternatives].argmax()])
        )
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
        if not len(alternatives):
            return None
        alternative = int(alternatives[policy[alternatives].argmax()])
    else:
        raise ValueError(f"Unknown correction category: {category}")
    return preferred, alternative


def collect(predict, root_evaluate, games, seed, *, discover=None):
    """Mine frozen games, then retain only one paired root per eligible game.

    Game categories are balanced before observing their decisions. Uniform
    reservoirs keep eight disagreement proposals, one for other categories.
    One qualifying disagreement is then chosen uniformly per assigned game.
    Both continuations retain the snapshot's concealed hands and resample its
    unseen wall with the same seed; subsequent actions use the frozen policy's
    serving-time greedy marginal selection, retaining its full probabilities.
    A supplied ``discover(capture)`` replays logged actions but passes current
    policy log probabilities to capture; it returns discovery statistics.
    """
    generator = numpy.random.default_rng(seed + 1000000000)
    categories = (numpy.arange(games) + seed) % len(CATEGORIES)
    generator.shuffle(categories)
    proposals = {}
    seen = numpy.zeros(games, numpy.int64)

    def capture(collector, batch):
        with tf.device("/CPU:0"):
            preferred = actor_layers.greedy_choices(
                batch["behavior"], batch["head"]
            ).numpy()
        for row, game in enumerate(batch["indices"][:, 0]):
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
            root = {
                "snapshot": collector.snapshot(row),
                "features": batch["features"][row].copy(),
                "head": int(batch["head"][row]),
                "legal": batch["legal"][row].copy(),
                "actions": pair,
                "source": batch["indices"][row].copy(),
            }
            retained = proposals.setdefault(int(game), [])
            if slot == len(retained):
                retained.append(root)
            else:
                retained[slot] = root

    if discover is None:
        _, statistics = self_play.collect(
            predict, games, seed, root_callback=capture, greedy=True
        )
    else:
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
        scored = root_evaluate(
            {name: features[name] for name in features.dtype.names}
        )
        for row, (game, root) in enumerate(disagreement):
            preferred = root["actions"][0]
            best = int(scored["q_values"][row].argmax())
            gap = (
                scored["q_values"][row, best]
                - scored["q_values"][row, preferred]
            )
            if best == preferred or gap < 0.25 / outcomes.OBJECTIVE_ST3_SCALE:
                continue
            assert root["legal"][[preferred, best]].all()
            assert preferred != best
            qualified[game] += 1
            if generator.integers(qualified[game]) == 0:
                root["actions"] = (preferred, best)
                roots[game] = root
    statistics["disagreement_scored"] = len(disagreement)
    statistics["disagreement_qualifying"] = int(qualified.sum())

    selected = sorted(roots)
    streams = []
    for game in selected:
        for action in roots[game]["actions"]:
            streams.append(
                self_play.collection_steps(
                    1,
                    seed + 1000000000 + game,
                    collector=roots[game]["snapshot"].fork(
                        1, seed + 1000000000 + game, until_hand_end=True
                    ),
                    forced_root=(
                        roots[game]["snapshot"].decision,
                        action,
                        1 << int(categories[game]),
                    ),
                    greedy=True,
                )
            )
    payments = []
    settlements = []
    for branch, (batch, _) in enumerate(
        self_play.collect_streams(predict, streams)
    ):
        row = numpy.flatnonzero(batch["paired_root"])
        assert len(row) == 1
        game = selected[branch // 2]
        assert batch["chosen"][row[0]] == roots[game]["actions"][branch % 2]
        assert batch["head"][row[0]] == roots[game]["head"]
        payments.append(batch["payment_points"][row[0]])
        settlements.append(batch["settlement"][row[0]])

    features = numpy.array(
        [roots[game]["features"] for game in selected],
        dtype=self_play.FEATURE_DTYPE,
    )
    for category, name in enumerate(CATEGORIES):
        statistics[f"assigned/{name}"] = int((categories == category).sum())
        statistics[f"eligible/{name}"] = int(seen[categories == category].sum())
        statistics[f"selected/{name}"] = int(
            sum(categories[game] == category for game in selected)
        )
    statistics["selected_roots"] = len(selected)
    statistics["root_continuations"] = 2 * len(selected)
    return {
        "features": {name: features[name] for name in features.dtype.names},
        "head": numpy.array(
            [roots[game]["head"] for game in selected], dtype=numpy.int32
        ),
        "legal": numpy.array(
            [roots[game]["legal"] for game in selected], dtype=bool
        ).reshape(-1, 150),
        "actions": numpy.array(
            [roots[game]["actions"] for game in selected], dtype=numpy.int32
        ).reshape(-1, 2),
        "payment_points": numpy.array(payments, dtype=numpy.int32).reshape(
            -1, 2, int(outcomes.PAYMENT_CELLS.sum())
        ),
        "settlement": numpy.array(settlements, dtype=numpy.int32).reshape(
            -1, 2, 12
        ),
        "category": categories[selected].astype(numpy.int8),
        "source_kind": numpy.full(
            len(selected), int(discover is not None), numpy.int8
        ),
        "source": numpy.array(
            [roots[game]["source"] for game in selected], dtype=numpy.int64
        ).reshape(-1, 4),
        "seed": seed + 1000000000 + numpy.array(selected, dtype=numpy.int64),
    }, statistics
