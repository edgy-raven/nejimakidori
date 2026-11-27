"""Frozen replay scoring preserves decision context and request ordering."""

import hashlib
import json
import pathlib
import shutil
import unittest.mock

import numpy
import tensorflow as tf

from log_dataset import _native_collection, features, replay
from model import actions, actor
from training import fixed_reference


def test_frozen_producer_scores_reordered_and_repeated_decisions(tmp_path):
    class Policy(tf.Module):
        @tf.function
        def policy(self, candidate_hand_counts, trigger_tile_id):
            values = tf.pad(
                tf.cast(candidate_hand_counts[:, 0], tf.float32),
                [[0, 0], [0, 116]],
            ) + tf.one_hot(tf.cast(trigger_tile_id, tf.int32), 150)
            return {
                name: values[:, : actions.ACTOR_OUTPUTS[name][0]]
                for name in actions.TRAIN_POLICY_HEADS
            }

    module = Policy()
    source = tmp_path / "source/log_dataset"
    source.mkdir(parents=True)
    shutil.copyfile(
        _native_collection.__file__, source / "_native_collection.so"
    )
    signature = module.policy.get_concrete_function(
        **{
            name: tf.TensorSpec([None, *spec["shape"]], spec["dtype"])
            for name, spec in _native_collection.INPUTS.items()
            if name in ("candidate_hand_counts", "trigger_tile_id")
        }
    )
    tf.saved_model.save(
        module, str(tmp_path / "saved_model"), signatures={"policy": signature}
    )
    policy = fixed_reference.Policy(tmp_path / "saved_model", batch_size=3)
    events = json.loads(
        pathlib.Path(__file__)
        .parents[1]
        .joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    state = replay.Replay("frozen", events)
    requested = []
    packed = []
    for event, seat, phase, _, _ in state.policy_decisions():
        requested.append((event, seat, features.PHASES[phase]))
        packed.append(features.pack(state, seat, phase))
    selected = numpy.array([0, len(requested) - 1, 2, 1, 2, 0, 1])
    request = numpy.asarray(requested)[selected]
    inputs = {
        name: numpy.stack([packed[index][name] for index in selected])
        for name in _native_collection.INPUTS
    }
    expected = signature(**{name: inputs[name] for name in policy.fields})
    policy.policy = unittest.mock.Mock(
        wraps=policy.policy,
        structured_outputs=policy.policy.structured_outputs,
    )
    actual = policy.predict(state.events, request)
    for name in actions.TRAIN_POLICY_HEADS:
        numpy.testing.assert_array_equal(actual[name], expected[name])
    heads = numpy.asarray([0, 2, 3, 1, 4])[request[:, 2]]
    masks = actions.legal_choices(inputs)
    legal = numpy.stack(
        [
            numpy.pad(
                masks[actions.TRAIN_POLICY_HEADS[head]][row],
                (
                    0,
                    150
                    - actions.ACTOR_OUTPUTS[actions.TRAIN_POLICY_HEADS[head]][
                        0
                    ],
                ),
            )
            for row, head in enumerate(heads)
        ]
    )
    numpy.testing.assert_allclose(
        policy.distribution(state.events, request, heads, legal),
        actor.distribution(expected, heads, legal),
        atol=0,
    )
    assert len(policy.policy.call_args_list) == 6
    assert all(
        value.shape[0] == 3
        for call in policy.policy.call_args_list
        for value in call.kwargs.values()
    )
    # Independent root histories share neural batches, including repeated
    # requests and prefixes ending at different inclusive event positions.
    items = [
        (
            state.events[: int(row[0]) + 1],
            request[position : position + 1],
            heads[position : position + 1],
            legal[position : position + 1],
        )
        for position, row in enumerate(request)
    ]
    policy.policy.reset_mock()
    numpy.testing.assert_array_equal(
        policy.distribution_many(iter(items)),
        actor.distribution(expected, heads, legal).numpy(),
    )
    assert policy.policy.call_count == 3
    assert all(
        value.shape[0] == 3
        for call in policy.policy.call_args_list
        for value in call.kwargs.values()
    )
    assert (
        policy.provenance["files_sha256"][
            "source/log_dataset/_native_collection.so"
        ]
        == hashlib.sha256(
            pathlib.Path(_native_collection.__file__).read_bytes()
        ).hexdigest()
    )
    assert "saved_model/saved_model.pb" in policy.provenance["files_sha256"]
    assert set(policy.provenance) == {"files_sha256"}
