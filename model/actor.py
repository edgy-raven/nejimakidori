"""Policy heads, legal masks and final candidate scoring."""

import tensorflow as tf

from log_dataset import call_states
from model import actions, beliefs, candidate_fusion


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class PolicyMask(tf.keras.layers.Layer):
    def __init__(self, head, **kwargs):
        super().__init__(**kwargs)
        self.head = head

    def get_config(self):
        return {**super().get_config(), "head": self.head}

    def call(self, values):
        logits, inputs = values
        mask = actions.legal_choices(inputs)[self.head]
        if self.head in ("discard_policy_logits", "riichi_policy_logits"):
            branches = []
            for start in range(
                0, 74 if self.head == "riichi_policy_logits" else 37, 37
            ):
                branch = logits[:, start : start + 37]
                legal = mask[:, start : start + 37]
                branch = tf.where(
                    (
                        (inputs["opponent_discard_count"][:, 0] == 0)
                        & (inputs["dealer_relative_seat"] == 0)
                    )[:, None],
                    opening_discard_symmetry(branch, inputs, legal),
                    branch,
                )
                if start == 0:
                    safe = final_wall_safe_cuts(inputs, legal)
                    # Preserve the branch mass. Even a suppressed legal cut
                    # must shift with the branch's additive logit offset.
                    total = tf.reduce_logsumexp(
                        tf.where(legal, branch, -1e4), axis=-1, keepdims=True
                    )
                    count = tf.reduce_sum(
                        tf.cast(safe, tf.float32), axis=-1, keepdims=True
                    )
                    branch = tf.where(
                        count > 0,
                        total
                        + tf.where(
                            safe, -tf.math.log(tf.maximum(count, 1.0)), -1e4
                        ),
                        branch,
                    )
                branches.append(branch)
            logits = tf.concat(branches, axis=-1)
        return tf.where(
            tf.reduce_any(mask, -1, keepdims=True),
            tf.where(mask, logits, -1e4),
            tf.zeros_like(logits),
        )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class PolicySummaries(tf.keras.layers.Layer):
    def call(self, outputs):
        riichi = tf.reshape(outputs["riichi_policy_logits"], (-1, 2, 37))
        response = outputs["response_policy_logits"]
        chii = tf.reshape(response[:, 1:112], (-1, 3, 37))
        pon = tf.reshape(response[:, 112:149], (-1, 1, 37))
        return {
            "riichi_action_logits": tf.reduce_logsumexp(riichi, -1),
            "response_action_logits": tf.stack(
                [
                    response[:, 0],
                    tf.reduce_logsumexp(chii, (1, 2)),
                    tf.reduce_logsumexp(pon, (1, 2)),
                    response[:, 149],
                ],
                axis=-1,
            ),
            "chi_pattern_logits": tf.reduce_logsumexp(chii, -1),
        }


def build_policy_heads(inputs, context, candidates, selected, predictions):
    """Condition each local action on decision and candidate predictions."""
    outputs = {}
    cut_evidence = [
        tf.keras.ops.sigmoid(tf.keras.ops.stop_gradient(predictions[name]))
        for name in ("route_affinity_logits", "candidate_ron_logits")
    ]
    final = FinalCandidates(dtype="float32")(
        [
            tf.keras.layers.Concatenate(dtype="float32")(
                [candidates, *cut_evidence]
            ),
            context,
            selected,
        ]
    )
    final = tf.keras.layers.Dense(
        512, activation="gelu", dtype="float32", name="final_pass"
    )(final)
    final = tf.keras.layers.LayerNormalization(
        center=False,
        scale=False,
        epsilon=1e-5,
        dtype="float32",
        name="final_pass_normalized",
    )(final)
    # Uncomputed actions are constant padding, not learned illegal logits.
    # Only the legal candidate scores participate in policy training.
    final = UnpackCandidates(dtype="float32")(
        [
            tf.keras.layers.Dense(1, dtype="float32", name="final_scores")(
                final
            ),
            selected,
            context,
        ]
    )[:, :, 0]
    for name, values in (
        ("discard_policy_logits", final[:, :37]),
        ("riichi_policy_logits", final[:, 37:111]),
        ("kan_action_logits", final[:, 260:295]),
    ):
        outputs[name] = PolicyMask(name, name=name, dtype="float32")(
            [values, inputs]
        )
    return outputs


def build_board_heads(inputs, context):
    win = tf.keras.layers.Dense(2, dtype="float32", name="win_decision_logits")(
        context
    )
    win = PolicyMask(
        "win_decision_logits", name="win_legality", dtype="float32"
    )([win, inputs])
    return {
        "win_decision_logits": win,
    }


def final_wall_safe_cuts(inputs, legal):
    """Return the forced uniform support, or an empty mask on free choices."""
    safe = legal & tf.gather(
        tf.reduce_all(actions.guaranteed_safe_tiles(inputs), axis=1),
        list(range(34)) + [4, 13, 22],
        axis=1,
    )
    # With an all-terminal/honor river, discarding a middle tile can forfeit
    # nagashi mangan. The feature contract does not record own called rivers,
    # so conservatively retain that possibility rather than infer it away.
    middle = tf.constant([i for i in range(27) if 0 < i % 9 < 8])
    nagashi_possible = ~tf.reduce_any(
        tf.gather(inputs["genbutsu_to_seat"][:, 0] > 0, middle, axis=1), axis=1
    )
    terminal_or_honor = tf.constant(
        [i >= 27 or i % 9 in (0, 8) for i in range(34)] + [False] * 3
    )
    safe &= ~nagashi_possible[:, None] | terminal_or_honor[None]
    return (
        safe
        & (inputs["live_wall_count"] == 0)[:, None]
        & ~tf.reduce_any(
            legal & (inputs["discard_action_all_shanten"] <= 1),
            axis=-1,
            keepdims=True,
        )
    )


def opening_discard_symmetry(logits, inputs, legal):
    """Pool equivalent legal cuts on the dealer’s first discard."""
    counts = inputs["candidate_hand_counts"][:, 0]
    suited = tf.reshape(counts[:, :27], (-1, 3, 9))
    non_dora = ~tf.reduce_any(
        tf.reshape(inputs["dora_multiplicity"][:, :27] > 0, (-1, 3, 9)),
        axis=-1,
    )
    terminals = tf.reshape(
        tf.stack(
            [
                non_dora
                & (suited[:, :, 0] == 1)
                & tf.reduce_all(suited[:, :, 1:5] == 0, -1),
                non_dora
                & (suited[:, :, 8] == 1)
                & tf.reduce_all(suited[:, :, 4:8] == 0, -1),
            ],
            axis=-1,
        ),
        (-1, 6),
    )
    floating = (
        tf.linalg.matmul(
            tf.cast(terminals, tf.float32),
            tf.one_hot([0, 8, 9, 17, 18, 26], 37),
        )
        > 0
    ) & legal
    symmetric = (
        non_dora
        & (suited[:, :, 0] == 1)
        & tf.reduce_all(suited == tf.reverse(suited, axis=[-1]), axis=-1)
        & tf.reduce_all(suited[:, :, 1:3] == 0, axis=-1)
        & ~tf.reduce_any(floating, axis=-1, keepdims=True)
    )
    winds = tf.constant([[0, 1, 2, 3]], dtype=tf.int8)
    groups = (
        tf.concat(
            [
                floating[:, None, :],
                tf.pad(
                    (counts[:, 31:34] == 1)
                    & (inputs["dora_multiplicity"][:, 31:34] == 0),
                    [[0, 0], [31, 3]],
                )[:, None, :],
                tf.pad(
                    (counts[:, 27:31] == 1)
                    & (inputs["dora_multiplicity"][:, 27:31] == 0)
                    & (winds != inputs["seat_wind"][:, 0, None])
                    & (winds != inputs["seat_wind"][:, 1, None])
                    & (winds != inputs["prevailing_wind"][:, None]),
                    [[0, 0], [27, 6]],
                )[:, None, :],
                symmetric[:, :, None]
                & tf.cast(
                    tf.one_hot([0, 9, 18], 37) + tf.one_hot([8, 17, 26], 37),
                    tf.bool,
                )[None],
            ],
            axis=1,
        )
        & legal[:, None, :]
    )
    # Log-space arithmetic preserves tiny probabilities and group mass.
    group_mean = tf.reduce_logsumexp(
        tf.where(groups, logits[:, None, :], -1e9), axis=-1
    ) - tf.math.log(
        tf.maximum(1.0, tf.reduce_sum(tf.cast(groups, tf.float32), axis=-1))
    )
    pooled = tf.reduce_sum(
        tf.where(groups, group_mean[:, :, None], 0.0), axis=1
    )
    return tf.where(tf.reduce_any(groups, axis=1), pooled, logits)


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class FinalCandidates(tf.keras.layers.Layer):
    def call(self, values):
        candidates, context, selected = values
        return tf.concat(
            [candidates, tf.gather(context, selected[:, 0])],
            axis=-1,
        )


def distribution(outputs, heads, legal):
    """Normalize over the exact native candidates used by the collector."""
    logits = tf.stack(
        [
            tf.pad(outputs[name], [[0, 0], [0, 150 - outputs[name].shape[-1]]])
            for name in actions.TRAIN_POLICY_HEADS
        ],
        axis=1,
    )
    logits = tf.gather(logits, heads, batch_dims=1)
    return tf.nn.log_softmax(tf.where(legal, logits, -1e9), axis=-1)


class CallPolicy(tf.keras.Model):
    """A call choice followed by the ordinary policy on accepted-call states."""

    def __init__(self, policy, hand_context, belief_layer):
        super().__init__(autocast=False)
        self.policy = policy
        head_names = (
            "route_affinity_logits",
            "route_constraint_loss",
            "candidate_ron_logits",
            "discard_policy_logits",
            "riichi_policy_logits",
            "kan_action_logits",
            "constraint_loss",
        )
        self.readout_inputs = (
            "candidate",
            "selected",
            "policy_context",
            "opponent_wait_logits",
            "ron_eligibility_logits",
            "belief_constraint_loss",
            "round_completion_yaku_logits_constraint_loss",
            "round_completion_score_logits_constraint_loss",
        )
        # Reapply the same head graph after attention. The policy model owns
        # its variables; this function introduces no second set of weights.
        self.readout = tf.keras.Function(
            [
                policy.input,
                {name: policy.output[name] for name in self.readout_inputs},
            ],
            {name: policy.output[name] for name in head_names},
            name="candidate_readout",
        )
        self.encode = tf.keras.Model(
            policy.input,
            {
                name: value
                for name, value in policy.output.items()
                if name not in head_names
            },
            name="decision_encoding",
            autocast=False,
        )
        self.encode_called = tf.keras.Model(
            policy.input,
            {
                name: policy.output[name]
                for name in (
                    "board",
                    "candidate",
                    "selected",
                    "structure",
                    "critic_rivers",
                )
            },
            name="called_state_encoding",
            autocast=False,
        )
        self.condition_inputs = (
            "board",
            *beliefs.BELIEF_HEADS,
            *beliefs.COMPLETION_HEADS,
            "belief_constraint_loss",
            "round_completion_yaku_logits_constraint_loss",
            "round_completion_score_logits_constraint_loss",
        )
        self.condition = tf.keras.Function(
            {name: policy.output[name] for name in self.condition_inputs},
            {
                name: policy.output[name]
                for name in (
                    "policy_context",
                    "belief_context",
                    "completion_context",
                )
            },
            name="belief_conditioning",
        )
        self.beliefs = belief_layer
        self.hand_context = hand_context
        self.fusion = candidate_fusion.CandidateFusion(name="candidate_fusion")
        self.call_choice = tf.keras.layers.Dense(
            6,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            dtype="float32",
            name="call_choice",
        )
        self.summaries = PolicySummaries(dtype="float32")

    def candidate_heads(self, features, outputs):
        outputs.update(
            self.readout(
                [
                    features,
                    {name: outputs[name] for name in self.readout_inputs},
                ]
            )
        )

    def call(self, features, training=False):
        current = call_states.base(features)
        outputs = self.encode(current, training=training)
        count = tf.shape(features["post_call_owner"])[0]
        capacity = tf.bitwise.left_shift(
            1,
            tf.cast(
                tf.math.ceil(
                    tf.math.log(tf.cast(tf.maximum(count, 1), tf.float32))
                    / tf.math.log(2.0)
                ),
                tf.int32,
            ),
        )
        # Bucket only device-side work; sparse transport remains unpadded.
        # One unselected dummy keeps shared convolutions defined when C=0.
        called_features = {
            name: tf.gather(
                tf.concat([value, current[name][:1]], axis=0),
                tf.minimum(tf.range(capacity), count),
            )
            for name, value in call_states.post(features).items()
        }
        called = self.encode_called(called_features, training=training)
        owners = tf.gather(
            tf.concat([features["post_call_owner"], [0]], axis=0),
            tf.minimum(tf.range(capacity), count),
        )
        # Considering our own call adds no evidence about concealed opponents.
        # Reuse the parent posterior; recompute completion potential on the
        # changed board with that same posterior as detached evidence.
        called.update(
            self.beliefs(
                inputs=called_features,
                hidden=called["board"],
                rivers=called["critic_rivers"],
                opponents={
                    name: tf.gather(outputs[name], owners)
                    for name in (
                        *beliefs.BELIEF_HEADS,
                        "belief_constraint_loss",
                    )
                },
            )
        )
        called.update(
            {
                name: tf.gather(outputs[name], owners)
                for name in ("opponent_wait_logits", "ron_eligibility_logits")
            }
        )
        called.update(
            self.condition(
                {name: called[name] for name in self.condition_inputs}
            )
        )
        child_rows = tf.where(
            (called["selected"][:, 0] < count) & (called["selected"][:, 1] < 37)
        )[:, 0]
        child_selected = tf.gather(called["selected"], child_rows)
        selected = tf.concat(
            [
                outputs["selected"],
                tf.stack(
                    [
                        tf.gather(
                            features["post_call_owner"], child_selected[:, 0]
                        ),
                        112
                        + 37
                        * tf.gather(
                            features["post_call_option"], child_selected[:, 0]
                        )
                        + child_selected[:, 1],
                    ],
                    axis=-1,
                ),
            ],
            axis=0,
        )
        order = tf.argsort(selected[:, 0] * 295 + selected[:, 1], stable=True)
        joint = tf.concat(
            [outputs["candidate"], tf.gather(called["candidate"], child_rows)],
            axis=0,
        )
        fused = tf.scatter_nd(
            order[:, None],
            self.fusion(
                [
                    tf.gather(joint, order),
                    tf.gather(selected[:, 0], order),
                    tf.shape(outputs["board"])[0],
                ]
            ),
            tf.shape(joint),
        )
        local_count = tf.shape(outputs["candidate"])[0]
        outputs["candidate"] = fused[:local_count]
        called["candidate"] = tf.tensor_scatter_nd_update(
            tf.cast(called["candidate"], tf.float32),
            child_rows[:, None],
            fused[local_count:],
        )
        self.candidate_heads(current, outputs)
        self.candidate_heads(called_features, called)
        conditional = tf.nn.log_softmax(
            called["discard_policy_logits"][:count], axis=-1
        )
        conditional = tf.tensor_scatter_nd_update(
            tf.fill((tf.shape(outputs["board"])[0], 4, 37), -1e4),
            tf.stack(
                [features["post_call_owner"], features["post_call_option"]],
                axis=-1,
            ),
            conditional,
        )
        legal = actions.legal_choices(current)["response_policy_logits"]
        branch_legal = tf.stack(
            [
                tf.reduce_any(part, axis=-1)
                for part in tf.split(legal, [1, 37, 37, 37, 37, 1], axis=-1)
            ],
            axis=-1,
        )
        choice = tf.nn.log_softmax(
            tf.where(
                branch_legal, self.call_choice(outputs["policy_context"]), -1e9
            ),
            axis=-1,
        )
        response = tf.concat(
            [
                choice[:, :1],
                tf.reshape(choice[:, 1:5, None] + conditional, (-1, 148)),
                choice[:, 5:],
            ],
            axis=-1,
        )
        outputs["response_policy_logits"] = tf.where(
            tf.reduce_any(legal, axis=-1, keepdims=True),
            tf.where(legal, response, -1e4),
            tf.zeros_like(response),
        )
        # Shared neural and hand context follows the accepted-call state.
        # Public event legality remains attached to the original call window.
        for rows, local in ((current, outputs), (called_features, called)):
            local["candidate_context"] = tf.concat(
                [
                    tf.cast(local["candidate"], tf.float32),
                    tf.gather(local["belief_context"], local["selected"][:, 0]),
                    tf.cast(
                        tf.gather(local["board"], local["selected"][:, 0]),
                        tf.float32,
                    ),
                    self.hand_context(
                        [rows, local["selected"], local["structure"]]
                    ),
                ],
                axis=-1,
            )
            local["candidate_completion"] = tf.gather(
                local["completion_context"], local["selected"][:, 0]
            )
        for name in (
            "candidate",
            "structure",
            "route_affinity_logits",
            "candidate_ron_logits",
            "candidate_context",
            "candidate_completion",
        ):
            outputs[name] = tf.gather(
                tf.concat(
                    [outputs[name], tf.gather(called[name], child_rows)], axis=0
                ),
                order,
            )
        outputs["selected"] = tf.gather(selected, order)
        outputs.update(self.summaries(outputs))
        return outputs


def greedy_choices(log_probability, heads):
    """Marginal branch selection on device, with lowest-index tie breaks."""
    chosen = tf.argmax(log_probability, axis=1, output_type=tf.int32)
    for head, sizes in (
        (1, (37, 37)),
        (2, (1, 37, 37, 37, 37, 1)),
    ):
        scores = log_probability[:, : sum(sizes)]
        branch = tf.argmax(
            tf.stack(
                [
                    tf.reduce_logsumexp(part, axis=1)
                    for part in tf.split(scores, sizes, axis=1)
                ],
                axis=1,
            ),
            axis=1,
            output_type=tf.int32,
        )
        scores = tf.where(
            tf.repeat(tf.range(len(sizes)), sizes)[None] == branch[:, None],
            scores,
            -float("inf"),
        )
        chosen = tf.where(
            heads == head,
            tf.argmax(scores, axis=1, output_type=tf.int32),
            chosen,
        )
    return chosen


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class UnpackCandidates(tf.keras.layers.Layer):
    def call(self, values):
        packed, selected, board = values
        return tf.scatter_nd(
            selected,
            packed,
            (tf.shape(board)[0], 295, packed.shape[-1]),
        )
