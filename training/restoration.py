"""Expert records and policy labels for structural restoration."""

import tensorflow as tf

from log_dataset import call_states
from model import actions, actor


def policy_choices(features, labels):
    """Recover the active policy head, observed choice and legal mask."""
    choices = tf.stack(
        [
            tf.cast(labels[name], tf.int32)
            for name in actions.TRAIN_POLICY_HEADS
        ],
        axis=1,
    )
    tf.debugging.assert_equal(
        tf.reduce_sum(tf.cast(choices >= 0, tf.int32), axis=1), 1
    )
    head = tf.argmax(choices >= 0, axis=1, output_type=tf.int32)
    expert = tf.gather(choices, head, batch_dims=1)
    masks = actions.legal_choices(features)
    legal = tf.gather(
        tf.stack(
            [
                tf.pad(masks[name], [[0, 0], [0, 150 - masks[name].shape[-1]]])
                for name in actions.TRAIN_POLICY_HEADS
            ],
            axis=1,
        ),
        head,
        batch_dims=1,
    )
    tf.debugging.assert_equal(tf.gather(legal, expert, batch_dims=1), True)
    return head, expert, legal


def trainable_decisions(features, head, legal):
    """Identify choices still open to the actor at this decision."""
    # Declaring commits the cut. Staying dama leaves the ordinary discard
    # eligible even when the player could have declared instead.
    riichi_cut = (
        head == actions.TRAIN_POLICY_HEADS.index("discard_policy_logits")
    ) & (features["riichi_state"][:, 0] > 0)
    return (tf.math.count_nonzero(legal, axis=1) > 1) & ~riichi_cut


def policy_target_supported(features, head, target):
    """Whether the exact teacher action survives the actor's forced cut rule."""

    def rejected_cut(inputs, tile):
        safe = actor.final_wall_safe_cuts(
            inputs, tf.cast(inputs["discard_action_legal"], tf.bool)
        )
        return tf.reduce_any(safe, axis=1) & ~tf.gather(
            safe, tile, batch_dims=1
        )

    # Only ordinary and dama branches consume the current board's cut.
    discard = (head == 0) | ((head == 1) & (target < 37))
    rejected = discard & rejected_cut(features, target % 37)
    # The joint chi/pon label uses the chosen child's ordinary discard.
    owner = features["post_call_owner"]
    child_target = tf.gather(target, owner)
    chosen_child = (
        (tf.gather(head, owner) == 2)
        & (child_target > 0)
        & (child_target < 149)
        & (features["post_call_option"] == (child_target - 1) // 37)
    )
    child_rejected = chosen_child & rejected_cut(
        call_states.post(features), (child_target - 1) % 37
    )
    return ~(
        rejected
        | (
            tf.math.unsorted_segment_sum(
                tf.cast(child_rejected, tf.int32), owner, tf.size(head)
            )
            > 0
        )
    )
