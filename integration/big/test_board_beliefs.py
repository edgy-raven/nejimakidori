"""Current hidden-state inference is independent of hypothetical actions."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import call_states, replay
from model import beliefs, network


def test_candidate_changes_preserve_beliefs_but_change_action_representations():
    tf.keras.utils.set_random_seed(941)
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/payment_dealer_90fu.json"
        ).read_text()
    )
    row = next(replay.Replay("board-beliefs", events).observations())
    features = call_states.base(call_states.batch([row.features]))
    model = network.JointPolicyPayment()
    model.beliefs.heads[beliefs.BELIEF_HEADS[0]].readout.kernel.assign(
        tf.random.stateless_normal((48, 4), [941, 1]) * 0.1
    )
    before = model.actor.policy(features)
    changed = {name: value.copy() for name, value in features.items()}
    # Replace hypothetical hands with other existing bank hands, leaving
    # current hand zero, all observed events and available copies untouched.
    occupied = numpy.flatnonzero(features["candidate_hand_valid"][0])[1:]
    assert len(occupied) > 1
    for name in (
        "candidate_hand_counts",
        "candidate_hand_red",
        "candidate_hand_yaku_possibility",
        "candidate_meld_tiles",
        "candidate_meld_types",
        "candidate_meld_red",
        "candidate_yaku_shanten",
        "candidate_fu",
    ):
        changed[name][:, occupied] = features[name][:, occupied[::-1]]
    for name in (
        "discard_action_normal_shanten",
        "discard_action_all_shanten",
        "discard_action_normal_ukeire_count",
        "discard_action_all_ukeire_count",
        "call_discard_shanten",
        "call_discard_ukeire_count",
        "call_efficiency",
        "kan_efficiency",
        "discard_retained_genbutsu",
    ):
        changed[name] += 1
    after = model.actor.policy(changed)
    for name in (
        "board",
        *beliefs.BELIEF_HEADS,
        "opponent_wait_logits",
        "ron_eligibility_logits",
        "candidate_ron_logits",
    ):
        numpy.testing.assert_array_equal(
            after[name], before[name], err_msg=name
        )
    assert numpy.max(numpy.abs(after["candidate"] - before["candidate"])) > 0.01
    assert not numpy.array_equal(
        after["discard_policy_logits"], before["discard_policy_logits"]
    )
    # Genuine new public evidence must still affect hidden-hand inference.
    changed = {name: value.copy() for name, value in features.items()}
    changed["riichi_state"][0, 1] = 2
    observed = model.actor.policy(changed)
    numpy.testing.assert_array_equal(
        tf.nn.softmax(observed["opponent_shanten_logits"], -1)[0, 0],
        [1, 0, 0, 0],
    )
    assert not numpy.array_equal(observed["board"], before["board"])
