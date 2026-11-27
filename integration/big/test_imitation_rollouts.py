"""Human roots supply single paired settlements without imitation labels."""

import json
import lzma
import pathlib
import sqlite3
import types

import numpy
import pytest
import tensorflow as tf

from integration import reference_policy
from log_dataset import (
    _native_collection,
    call_states,
    features,
    records,
    replay,
    root_states,
    training_data,
    training_targets,
)
from training import expert, imitation_rollouts, restoration, self_play
from training.corrections import collection


def write_roots(directory, events):
    directory.mkdir()
    events = events[
        next(
            i
            for i, event in enumerate(events)
            if event["type"] == "start_kyoku"
        ) :
    ]
    events = events[
        : next(
            i for i, event in enumerate(events) if event["type"] == "end_kyoku"
        )
        + 1
    ]
    prepared = root_states.prepare_hand("imitation-rollouts", events)
    with sqlite3.connect(directory / "index.sqlite") as connection:
        connection.execute("CREATE TABLE games (game_id TEXT, shard TEXT)")
        connection.execute(
            "INSERT INTO games VALUES (?, ?)",
            ("imitation-rollouts", "roots.sqlite"),
        )
    with sqlite3.connect(directory / "roots.sqlite") as connection:
        connection.execute(
            "CREATE TABLE hands (game_id TEXT, kyoku_id TEXT, "
            "roots BLOB, witnesses TEXT)"
        )
        connection.execute(
            "INSERT INTO hands VALUES (?, ?, ?, ?)",
            (
                "imitation-rollouts",
                f"{events[0]['bakaze']}{events[0]['kyoku']}-{events[0]['honba']}",
                lzma.compress(
                    json.dumps(list(prepared.items())).encode(), preset=3
                ),
                json.dumps(
                    [
                        (key, value["witness"])
                        for key, value in prepared.items()
                        if value["witness"] is not None
                    ]
                ),
            ),
        )


def test_human_hand_reconstruction_preserves_features_and_can_fork(tmp_path):
    coverage = set()
    for filename in (
        "payment_dealer_90fu.json",
        "payment_dealer_pao.json",
        "pass_speed_round.json",
        "recorded_nonred_pon.json",
    ):
        events = json.loads(
            pathlib.Path("integration/fixtures/model", filename).read_text()
        )
        events = events[
            next(
                i
                for i, event in enumerate(events)
                if event["type"] == "start_kyoku"
            ) :
        ]
        requested = numpy.arange(0, len(events), 7)
        expected_targets = training_targets.round_targets(
            replay.Replay.from_start(filename, events[0]), events
        )
        sparse_targets = training_targets.round_targets(
            replay.Replay.from_start(filename, events[0]),
            events,
            indices=requested,
        )
        for name in ("completion", "completion_score"):
            numpy.testing.assert_array_equal(
                sparse_targets[name][requested],
                expected_targets[name][requested],
            )
        numpy.testing.assert_array_equal(
            sparse_targets["payments"], expected_targets["payments"]
        )
        state = replay.Replay(filename, events)
        collector = _native_collection.NativeCollector.from_hand(
            json.dumps(events), 941
        )
        while (request := collector.next()) is not None:
            packed = call_states.from_request(request)
            for row, (actor, event) in enumerate(
                zip(request["seat"], request["event"])
            ):
                while state.event_index < event - 1:
                    state.apply(events[state.event_index + 1])
                phase = {value: key for key, value in features.PHASES.items()}[
                    int(packed["decision_phase"][row])
                ]
                expected = call_states.batch(
                    [features.pack(state, actor, phase)]
                )
                actual = call_states.select(packed, [row])
                for name in expected:
                    numpy.testing.assert_array_equal(
                        actual[name], expected[name], err_msg=name
                    )
                coverage.add(phase)
            collector.submit_recorded()
        assert collector.finish()[1]["hands"] == 1
        # Prepared records include post-call discards and staying dama,
        # even when ordinary simulation would combine or skip those requests.
        write_roots(tmp_path / filename, events)
        observations = list(
            replay.Replay("imitation-rollouts", events).observations()
        )
        prepared, labels = records.parse_batch(
            tf.constant(
                [records.serialize_observation(row) for row in observations]
            )
        )
        head, _, legal = restoration.policy_choices(prepared, labels)
        selected = numpy.flatnonzero(
            restoration.trainable_decisions(prepared, head, legal)
        )
        metadata = {
            name: numpy.array(
                [getattr(row, name) for row in observations],
                dtype="S" if name in ("game_id", "kyoku_id") else None,
            )
            for name in ("game_id", "kyoku_id", "decision_index", "actor")
        }
        source = root_states.Store(tmp_path / filename)
        for row in selected[::-1]:
            root = source.root(
                game_id=metadata["game_id"][row].decode(),
                kyoku_id=metadata["kyoku_id"][row].decode(),
                decision_index=int(metadata["decision_index"][row]),
                actor=int(metadata["actor"][row]),
                phase=int(prepared["decision_phase"][row]),
            )
            actual = call_states.select(
                call_states.from_request(root.fork(1, 941, False).next()),
                [root.decision],
            )
            expected = call_states.select(
                tf.nest.map_structure(lambda value: value.numpy(), prepared),
                [row],
            )
            for name in expected:
                numpy.testing.assert_array_equal(actual[name], expected[name])
        source.close()
    assert {"DISCARD", "RESPONSE", "KAN", "RIICHI"} <= coverage


def test_categories_keep_all_advancing_calls_and_unique_comparators():
    row = numpy.zeros((), call_states.ROW_DTYPE)
    row["self_efficiency"][0] = 4 / 8
    row["call_discard_legal"][0, :3] = 1
    row["call_discard_shanten"][0, :3] = [4, 3, 5]
    legal = numpy.zeros(150, bool)
    legal[:4] = True
    choices, categories = collection.targeted_actions(
        features=row,
        head=2,
        legal=legal,
        policy=numpy.zeros(150),
        preferred=3,
        values=numpy.zeros(37),
        sakigiri_actions=numpy.array([], dtype=numpy.int32),
    )
    assert choices == [0, 1, 2] and categories == ["calls"]


def test_single_root_outcomes_keep_human_features_and_exclude_holdouts(
    tmp_path,
):
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    observations = list(
        replay.Replay("imitation-rollouts", events).observations()
    )
    observation = next(
        row
        for row in observations
        if row.features["decision_phase"] == 1
        and self_play.improving_call_choices(row.features)[0].any()
    )
    raw = tf.train.Example.FromString(
        records.serialize_observation(observation)
    )
    raw.features.feature["expert/mortal_action"].bytes_list.value.append(
        numpy.int16(0).tobytes()
    )
    batch = expert.decode_records(tf.constant([raw.SerializeToString()] * 64))
    batch.update(
        local_count=tf.constant(64),
        natural=tf.constant([True, False] * 32),
        metadata={
            name: tf.constant([getattr(observation, name)] * 64)
            for name in ("game_id", "kyoku_id", "decision_index", "actor")
        },
    )
    write_roots(tmp_path / "roots", events)
    source = root_states.Store(tmp_path / "roots")
    packing = training_data.PackedBatch(call_states.base(batch["features"]))
    predicted = []

    def predict(features):
        predicted.append(len(features["decision_phase"]))
        return reference_policy.uniform_logits(features)

    seeds = imitation_rollouts.select_seeds(
        source=source,
        batch=batch,
        predict=predict,
        evaluate=lambda rows: numpy.zeros((len(rows["decision_phase"]), 37)),
        seed=941,
        decisions_per_continuation=32,
        stop=tmp_path / "STOP",
    )
    selection_rows = sum(predicted)
    assert selection_rows < 64  # Stop scoring once the pair budget is filled.
    exhaustive = imitation_rollouts.select_seeds(
        source=source,
        batch=batch,
        predict=reference_policy.uniform_logits,
        evaluate=lambda rows: numpy.zeros((len(rows["decision_phase"]), 37)),
        seed=941,
        decisions_per_continuation=0.001,
        stop=tmp_path / "STOP",
    )
    # Chunking must not cap the candidate search or discard later qualifiers.
    assert len(exhaustive[::2]) == len(exhaustive[1::2]) == 64
    assert sorted(row["seed"] for row in exhaustive[::2]) == list(
        range(941, 941 + 64)
    )
    rows, statistics = imitation_rollouts.collect(
        source=source,
        seeds=seeds,
        reference_policy=reference_policy.Policy(),
        decision_target=64,
        predict=predict,
        packing=packing,
        directory=tmp_path / "rollouts",
        stop=tmp_path / "STOP",
    )
    receipt = json.loads((tmp_path / "rollouts/roots.json").read_text())
    assert statistics["decision_target"] == 64
    assert statistics["simulated_decisions"] == sum(predicted) - selection_rows
    assert statistics["simulated_decisions"] > statistics["continuations"]
    predicted.clear()
    # Multiple batches' starts enter the same inference request, without
    # changing any seeded trajectory or carrying the previous overrun.
    repeated, repeated_statistics = imitation_rollouts.collect(
        source=source,
        seeds=seeds * 16,
        reference_policy=reference_policy.Policy(),
        decision_target=64 * 16,
        predict=predict,
        packing=packing,
        directory=tmp_path / "next-window",
        stop=tmp_path / "STOP",
    )
    assert predicted[0] >= len(seeds) * 16
    assert repeated_statistics["simulated_decisions"] == sum(predicted)
    assert repeated_statistics["simulated_decisions"] == (
        statistics["simulated_decisions"] * 16
    )
    numpy.testing.assert_array_equal(
        repeated["payment_points"], numpy.tile(rows["payment_points"], (16, 1))
    )
    assert statistics["roots"] == 2
    assert statistics["targeted_roots"] == statistics["normal_roots"] == 1
    assert statistics["continuations"] == sum(
        len(root["actions"]) for root in receipt
    )
    assert all(
        len(seeds) == 1 for root in receipt for seeds in root["wall_seeds"]
    )
    assert [root["kind"] for root in receipt] == ["targeted", "normal"]
    assert 0 in receipt[0]["actions"]
    assert all(
        seeds == receipt[0]["wall_seeds"][0]
        for seeds in receipt[0]["wall_seeds"]
    )
    assert not set(receipt[1]["wall_seeds"][0]) & set(
        receipt[1]["wall_seeds"][1]
    )
    assert len(rows["chosen"]) == statistics["continuations"]
    numpy.testing.assert_allclose(
        numpy.exp(rows["reference_log_probability"]).sum(axis=1), 1
    )
    numpy.testing.assert_array_equal(
        repeated["reference_log_probability"],
        numpy.tile(rows["reference_log_probability"], (16, 1)),
    )
    assert set(rows) == {
        "features",
        "labels",
        "head",
        "legal",
        "chosen",
        "payment_points",
        "reference_log_probability",
    }
    assert set(rows["labels"]) == set(batch["labels"])
    restored = call_states.unpack(
        tf.nest.map_structure(tf.constant, rows["features"]), packing
    )
    assert (
        restoration.trainable_decisions(restored, rows["head"], rows["legal"])
        .numpy()
        .all()
    )
    with numpy.load(tmp_path / "rollouts/outcomes.npz") as saved:
        numpy.testing.assert_array_equal(
            saved["payment_points"], rows["payment_points"]
        )
    # A rank without roots still joins every update with zero valid mass.
    group = types.SimpleNamespace(
        replicas=2, packing=packing, maximum=lambda count: len(rows["chosen"])
    )
    populated = list(imitation_rollouts.batches(rows, batch, group, 14))
    empty = list(imitation_rollouts.batches(None, batch, group, 14))
    assert len(populated) == len(empty)
    assert sum(int(part["local_count"]) for part in populated) == len(
        rows["chosen"]
    )
    assert all(int(part["local_count"]) == 0 for part in empty)
    assert all(len(part["chosen"]) == 7 for part in populated + empty)
    # A focused-only tail cannot supply an unbiased normal partner.
    unpaired = imitation_rollouts.select_seeds(
        source=source,
        batch={**batch, "natural": tf.constant([False] * 64)},
        predict=reference_policy.uniform_logits,
        evaluate=lambda rows: numpy.zeros((len(rows["decision_phase"]), 37)),
        seed=941,
        decisions_per_continuation=32,
        stop=tmp_path / "STOP",
    )
    assert unpaired == []
    # Held-out IDs are rejected before any native root is created.
    assert records.game_partition("expert") >= 7
    with pytest.raises(ValueError, match="held-out"):
        imitation_rollouts.select_seeds(
            source=source,
            batch={
                **batch,
                "metadata": {
                    **batch["metadata"],
                    "game_id": tf.constant(["expert"] * 64),
                },
            },
            predict=reference_policy.uniform_logits,
            evaluate=lambda rows: numpy.zeros(
                (len(rows["decision_phase"]), 37)
            ),
            seed=941,
            decisions_per_continuation=32,
            stop=tmp_path / "STOP",
        )
    source.close()
