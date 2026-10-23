"""Native sampling metadata agrees with the frozen public-feature rules."""

import json

import numpy

from integration import focus_reference
from log_dataset import _native_collection, features, replay


def test_focus_metadata_matches_public_feature_oracle():
    collector = _native_collection.NativeCollector(2, 12)
    generator = numpy.random.default_rng(12)
    while (request := collector.next()) is not None:
        legal = numpy.frombuffer(request["legal"], numpy.bool_).reshape(-1, 150)
        collector.submit(
            [int(generator.choice(numpy.flatnonzero(row))) for row in legal]
        )
    coverage = set()
    for game, events in enumerate(collector.finish()[0]):
        state = replay.Replay(str(game), json.loads(events))
        for _, actor, phase, _, _ in state.policy_decisions():
            packed, metadata = features.pack(
                state=state, actor=actor, phase=phase, include_focus=True
            )
            category = focus_reference.categories(packed)
            expected = (
                category,
                bool(category & 32)
                and bool(focus_reference.late_placement_focus(packed)),
                bool(category & 32)
                and focus_reference.placement_gap_focus(packed),
            )
            assert metadata == expected, (game, state.event_index, actor)
            coverage.update(bit for bit in range(6) if category & (1 << bit))
    assert coverage == set(range(6))


def test_generated_records_store_and_read_native_focus(tmp_path):
    import pathlib

    import tensorflow as tf

    from log_dataset import afk, scoring, training_data

    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    targets = scoring.round_labels(events)
    observations = list(
        replay.Replay(
            "focus-record",
            afk.retained_hands(events)[0],
        ).observations()
    )
    path = tmp_path / "focus.tfrecord.gz"
    with tf.io.TFRecordWriter(str(path), options="GZIP") as writer:
        for observation in observations:
            writer.write(
                training_data.serialize(
                    observation, targets[observation.kyoku_id]
                )
            )
    raw = list(tf.data.TFRecordDataset(str(path), compression_type="GZIP"))
    metadata = tf.io.parse_example(
        tf.stack(raw),
        {
            "meta/" + name: tf.io.FixedLenFeature([], tf.int64)
            for name in ("focus_categories", "late_placement", "placement_gap")
        },
    )
    numpy.testing.assert_array_equal(
        numpy.stack(list(metadata.values()), axis=-1),
        [observation.focus for observation in observations],
    )
    training_data.parse_batch(tf.stack(raw))
