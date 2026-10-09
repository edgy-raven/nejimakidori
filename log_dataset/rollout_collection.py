"""Decision-time nominations and persistent, budgeted collection work."""

import io
import itertools
import json
import sqlite3

import numpy

from log_dataset import rollout_store


def nominations(packed, prediction, expert):
    phase = int(packed["decision_phase"])
    head, offset, size = (
        ("discard", 0, 37),
        ("response", 111, 150),
        ("kan_action", 260, 35),
        ("riichi", 37, 74),
    )[phase]
    key = head + ("_probabilities" if phase == 2 else "_policy_probabilities")
    probability = numpy.asarray(prediction[key])
    codes = numpy.arange(size)
    identifiers = codes + offset
    if phase == 1:
        identifiers[-1] = 261 + int(packed["trigger_tile_id"])
    values = {row["action"]: row for row in prediction["action_values"]}
    legal = numpy.array([action in values for action in identifiers])
    # Match the actor's marginal decision, then its conditional continuation.
    groups = []
    if phase == 3:
        groups = [codes // 37]
    elif phase == 1:
        groups = [
            numpy.where(
                codes == 0,
                0,
                numpy.where(codes < 112, 1, numpy.where(codes < 149, 2, 3)),
            ),
            numpy.where(codes == 0, 0, (codes - 1) // 37 + 1),
        ]
    eligible = legal.copy()
    for group in groups:
        totals = numpy.bincount(group, weights=probability * eligible)
        eligible &= group == numpy.argmax(totals)
    actor = int(
        identifiers[numpy.argmax(numpy.where(eligible, probability, -1))]
    )
    legal_values = [values[int(action)] for action in identifiers[legal]]
    result = {
        "actor": actor,
        "critic": max(legal_values, key=lambda row: row["projected_st3"])[
            "action"
        ],
        "point_ev": max(
            legal_values, key=lambda row: row["expected_points"][0]
        )["action"],
    }
    expert_key = (
        "discard_policy",
        "response_policy",
        "kan_action",
        "riichi_policy",
    )[phase]
    if expert_key in expert:
        code = int(expert[expert_key])
        action = int(identifiers[code])
        if not legal[code]:
            raise ValueError("Expert action is outside decision-time support")
        result["expert"] = action
    if int(packed["live_wall_count"]) < 16 and phase in (0, 1, 3):
        # Include a tenpai-preserving choice and the least immediate ron risk
        # among noten choices. Kan needs replacement-draw analysis.
        distances = {}
        for row in legal_values:
            action = row["action"]
            if phase in (0, 3):
                distances[action] = packed["discard_action_all_shanten"][
                    action if phase == 0 else (action - 37) % 37
                ]
            elif action == 111:
                distances[action] = (
                    round(float(packed["self_efficiency"][0]) * 8) + 1
                )
            elif 112 <= action < 223:
                distances[action] = packed["chi_discard_shanten"].reshape(-1)[
                    action - 112
                ]
            elif 223 <= action < 260:
                distances[action] = packed["pon_discard_shanten"][
                    0, action - 223
                ]
        tenpai = [
            row
            for row in legal_values
            if row["action"] in distances and distances[row["action"]] == 1
        ]
        noten = [
            row
            for row in legal_values
            if row["action"] in distances and distances[row["action"]] > 1
        ]
        if tenpai and noten:
            result["keiten"] = max(
                tenpai, key=lambda row: row["projected_st3"]
            )["action"]
            result["noten"] = min(
                noten,
                key=lambda row: (
                    sum(row["immediate_ron_probabilities"]),
                    -row["projected_st3"],
                ),
            )["action"]
    if phase == 3:
        for name, declaration in (("dama", False), ("riichi", True)):
            result[name] = max(
                (
                    row
                    for row in legal_values
                    if (row["action"] >= 74) == declaration
                ),
                key=lambda row: row["projected_st3"],
            )["action"]
        # Pair nominated discards where both declaration choices are legal.
        for action in sorted(set(result.values())):
            opposite = action + 37 if action < 74 else action - 37
            if opposite in values:
                result[f"riichi_dama_{opposite}"] = opposite
    disagreement = len(set(result.values())) > 1
    eligible = rollout_store.eligible(packed, disagreement=disagreement)
    if eligible and not disagreement and len(legal_values) > 1:
        result["probe"] = max(
            (row for row in legal_values if row["action"] != actor),
            key=lambda row: row["projected_st3"],
        )["action"]
    return {
        "nominations": result,
        "disagreement": disagreement,
        "eligible": eligible,
        "priority": max(row["projected_st3"] for row in legal_values)
        - min(values[action]["projected_st3"] for action in result.values()),
    }


class Queue:
    """One collector owns a queue; completed paired trials commit atomically."""

    def __init__(self, path):
        self.connection = sqlite3.connect(path, timeout=60)
        # Completed stores must open from read-only artifact mounts without
        # requiring SQLite to create WAL shared-memory sidecars.
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS roots (
                id TEXT PRIMARY KEY, state TEXT NOT NULL,
                features BLOB NOT NULL,
                policy TEXT NOT NULL, position TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS samples (
                root TEXT NOT NULL, actions TEXT NOT NULL,
                seed INTEGER NOT NULL,
                sample INTEGER NOT NULL, stage TEXT NOT NULL,
                outcomes TEXT NOT NULL,
                PRIMARY KEY(root, actions, seed, sample));
            CREATE TABLE IF NOT EXISTS collection (
                root TEXT PRIMARY KEY, mark TEXT NOT NULL,
                priority REAL NOT NULL,
                trials INTEGER NOT NULL DEFAULT 0);
        """)

    def close(self):
        self.connection.close()

    def mark(self, *, state, packed, prediction, expert, policy):
        mark = nominations(packed, prediction, expert)
        if not mark["eligible"] or len(set(mark["nominations"].values())) < 2:
            return None
        state = {**state, "phase": int(packed["decision_phase"])}
        position = rollout_store.fingerprint(state)
        state = {**state, "disagreement": mark["disagreement"]}
        identity = rollout_store.fingerprint(
            {"position": position, "policy": policy}
        )
        buffer = io.BytesIO()
        numpy.savez_compressed(buffer, **packed)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO roots VALUES (?, ?, ?, ?, ?)",
                (
                    identity,
                    json.dumps(state),
                    buffer.getvalue(),
                    json.dumps(policy),
                    position,
                ),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO collection(root,mark,priority) "
                "VALUES (?,?,?)",
                (
                    identity,
                    json.dumps(
                        {
                            **mark,
                            "prediction": {
                                name: prediction[name]
                                for name in (
                                    "action_values",
                                    "discard_policy_probabilities",
                                    "riichi_policy_probabilities",
                                    "response_policy_probabilities",
                                    "kan_action_probabilities",
                                )
                            },
                        }
                    ),
                    mark["priority"],
                ),
            )
        return identity

    def next(self, trials):
        row = self.connection.execute(
            "SELECT roots.id,state,features,policy,mark,trials FROM collection "
            "JOIN roots ON roots.id=collection.root WHERE trials<? "
            "ORDER BY trials,priority DESC,roots.id LIMIT 1",
            (trials,),
        ).fetchone()
        if row is None:
            return None
        with numpy.load(io.BytesIO(row[2])) as arrays:
            packed = dict(arrays)
        return {
            "id": row[0],
            "state": json.loads(row[1]),
            "packed": packed,
            "policy": json.loads(row[3]),
            "mark": json.loads(row[4]),
            "trial": row[5],
        }

    def complete(self, root, seed, results):
        with self.connection:
            for first, second in itertools.combinations(sorted(results), 2):
                self.connection.execute(
                    "INSERT INTO samples VALUES (?,?,?,?,?,?)",
                    (
                        root["id"],
                        json.dumps([first, second]),
                        seed,
                        root["trial"],
                        "screen",
                        json.dumps([results[first], results[second]]),
                    ),
                )
            self.connection.execute(
                "UPDATE collection SET trials=trials+1 WHERE root=?",
                (root["id"],),
            )
