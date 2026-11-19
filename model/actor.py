"""Policy heads, action priors and final candidate scoring."""

import math

import tensorflow as tf

from model import actions, beliefs, encoder, losses, payment_scoring


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
                    branch = tf.where(
                        (inputs["live_wall_count"] == 0)[:, None],
                        late_noten_safety(branch, inputs, legal),
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
class PolicyPrior(tf.keras.layers.Layer):
    def __init__(self, head, **kwargs):
        super().__init__(**kwargs)
        self.head = head

    def get_config(self):
        return {**super().get_config(), "head": self.head}

    def call(self, values):
        logits, inputs = values
        mask = actions.legal_choices(inputs)[self.head]
        raw = tf.where(mask, logits, -1e4)
        if self.head == "discard_policy_logits":
            logits = opening_discard_symmetry(
                opening_discard_efficiency(logits, inputs, mask),
                inputs,
                mask,
            )
        elif self.head == "riichi_policy_logits":
            logits = tf.concat(
                [
                    opening_discard_symmetry(
                        opening_discard_efficiency(
                            logits[:, start : start + 37],
                            inputs,
                            mask[:, :37],
                        ),
                        inputs,
                        mask[:, start : start + 37],
                    )
                    for start in (0, 37)
                ],
                axis=-1,
            )
        if self.head == "discard_policy_logits":
            logits = late_noten_safety(logits, inputs, mask)
        elif self.head == "riichi_policy_logits":
            # Only the dama branch can contain noten choices.
            logits = tf.concat(
                [
                    late_noten_safety(logits[:, :37], inputs, mask[:, :37]),
                    logits[:, 37:],
                ],
                axis=-1,
            )
        target = tf.stop_gradient(
            tf.nn.log_softmax(tf.where(mask, logits, -1e4))
        )
        return tf.reduce_sum(
            tf.exp(target) * (target - tf.nn.log_softmax(raw)), axis=-1
        )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class PolicySummaries(tf.keras.layers.Layer):
    def call(self, values):
        outputs = values
        riichi = tf.reshape(outputs["riichi_policy_logits"], (-1, 2, 37))
        response = outputs["response_policy_logits"]
        chii = tf.reshape(response[:, 1:112], (-1, 3, 37))
        pon = tf.reshape(response[:, 112:149], (-1, 1, 37))
        return {
            **beliefs.opponent_summaries(outputs),
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


def build_policy_heads(inputs, hidden, candidates, selected, context):
    """Named auxiliary predictions followed by one action policy."""
    route = tf.keras.layers.Dense(3, dtype="float32", name="route_affinity")(
        candidates
    )
    raw_route = route
    route = RouteMask(dtype="float32")([route, inputs, selected])
    ron = tf.keras.layers.Dense(3, dtype="float32", name="candidate_ron")(
        candidates
    )
    outputs = {
        "route_affinity_logits": route,
        "route_constraint_loss": losses.ConstraintLoss(
            binary=True, dtype="float32"
        )([raw_route, route, selected[:, 0]]),
        "candidate_ron_logits": ron,
    }
    final = FinalCandidates()(
        [
            tf.keras.layers.Concatenate()(
                [
                    candidates,
                    tf.keras.layers.Activation("sigmoid")(route),
                    ron,
                ]
            ),
            context,
            selected,
        ]
    )
    final = tf.keras.layers.Dense(256, activation="gelu", name="final_pass")(
        final
    )
    final = tf.keras.layers.LayerNormalization(
        center=False,
        scale=False,
        epsilon=1e-5,
        dtype="float32",
        name="final_pass_normalized",
    )(final)
    # Uncomputed actions are constant padding, not learned illegal logits.
    # Only the legal candidate scores participate in policy training.
    final = UnpackCandidates()(
        [
            tf.keras.layers.Dense(1, dtype="float32", name="final_scores")(
                final
            ),
            selected,
            hidden,
        ]
    )[:, :, 0]
    for name, values in (
        ("discard_policy_logits", final[:, :37]),
        ("riichi_policy_logits", final[:, 37:111]),
        (
            "response_policy_logits",
            ResponseKan(dtype="float32")([final, inputs]),
        ),
        ("kan_action_logits", final[:, 260:295]),
    ):
        outputs[name] = PolicyMask(name, name=name, dtype="float32")(
            [values, inputs]
        )
        outputs[name + "_prior_loss"] = PolicyPrior(
            name, name=name + "_prior_loss", dtype="float32"
        )([values, inputs])
    return outputs


def build_board_heads(inputs, hidden, predictions):
    win = tf.keras.layers.Dense(2, dtype="float32", name="win_decision_logits")(
        hidden
    )
    outputs = {
        "win_decision_logits": PolicyMask(
            "win_decision_logits", name="win_legality", dtype="float32"
        )([win, inputs]),
    }
    outputs["win_decision_logits_prior_loss"] = PolicyPrior(
        "win_decision_logits", dtype="float32"
    )([outputs["win_decision_logits"], inputs])
    outputs.update({name: values for name, (values, _) in predictions.items()})
    auxiliary_logits = [probability for _, probability in predictions.values()]
    for name in (
        "round_completion_yaku_logits",
        "round_completion_score_logits",
    ):
        values = tf.keras.layers.Dense(
            math.prod(actions.ACTOR_OUTPUTS[name]), dtype="float32"
        )(hidden)
        values = tf.keras.layers.Reshape(
            actions.ACTOR_OUTPUTS[name], dtype="float32"
        )(values)
        raw = values
        if name == "round_completion_yaku_logits":
            values = payment_scoring.CompletionMask(dtype="float32")(
                [values, inputs]
            )
            auxiliary_logits.append(
                tf.keras.layers.Activation("sigmoid")(values)
            )
        else:
            values = payment_scoring.CompletionScoreMask(dtype="float32")(
                [values, inputs]
            )
            values = payment_scoring.CompletionScoreCoupling(dtype="float32")(
                [values, outputs["round_completion_yaku_logits"], inputs]
            )
            auxiliary_logits.append(tf.keras.layers.Softmax()(values))
        outputs[name + "_constraint_loss"] = losses.ConstraintLoss(
            binary=name == "round_completion_yaku_logits", dtype="float32"
        )([raw, values])
        outputs[name] = values
    # Completion probabilities carry the same constraints into the policy.
    # Other heads retain their established representation.
    context = tf.keras.layers.Dense(128, activation="gelu")(
        tf.keras.layers.Concatenate()(
            [hidden, encoder.numeric_features(auxiliary_logits)]
        )
    )
    context = tf.keras.layers.Concatenate()([hidden, context])
    return outputs, context


def late_noten_safety(logits, inputs, legal):
    """Mix in uniform proven-safe discards; fully override at exhaustion."""
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
    active = tf.reduce_any(safe, axis=-1) & ~tf.reduce_any(
        legal & (inputs["discard_action_all_shanten"] <= 1), axis=-1
    )
    strength = tf.where(
        active,
        1
        - tf.square(
            tf.clip_by_value(
                tf.cast(inputs["live_wall_count"], tf.float32) / 8, 0, 1
            )
        ),
        0.0,
    )[:, None]
    total = tf.reduce_logsumexp(
        tf.where(legal, logits, -1e4), axis=-1, keepdims=True
    )
    uniform = total - tf.math.log(
        tf.maximum(
            1.0,
            tf.reduce_sum(tf.cast(safe, tf.float32), axis=-1, keepdims=True),
        )
    )
    target = tf.where(safe, uniform, -1e4)
    mixed = tf.reduce_logsumexp(
        tf.stack(
            [
                logits + tf.math.log(tf.maximum(1 - strength, 1e-30)),
                target + tf.math.log(tf.maximum(strength, 1e-30)),
            ],
            axis=-1,
        ),
        axis=-1,
    )
    return tf.where(
        strength == 0, logits, tf.where(strength == 1, target, mixed)
    )


def opening_discard_efficiency(logits, inputs, legal):
    """Bounded efficiency bias, gone after five discards by any opponent."""
    score = -tf.cast(
        inputs["discard_action_all_shanten"], tf.float32
    ) + 0.35 * tf.math.log1p(
        tf.cast(inputs["discard_action_all_ukeire_count"], tf.float32)
    ) / math.log(
        137.0
    )
    strength = tf.maximum(
        0.0,
        1.0
        - tf.reduce_max(
            tf.cast(inputs["opponent_discard_count"][:, 1:], tf.float32),
            axis=-1,
            keepdims=True,
        )
        / 5.0,
    )
    bias = tf.clip_by_value(
        score
        - tf.reduce_max(tf.where(legal, score, -1e4), axis=-1, keepdims=True),
        -1.5,
        0.0,
    )
    return logits + tf.where(legal, strength * bias, 0.0)


def opening_discard_symmetry(logits, inputs, legal):
    """Exact dealer first-cut pooling; otherwise fade with opponent discards."""
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
    strength = tf.where(
        (inputs["opponent_discard_count"][:, 0] == 0)
        & (inputs["dealer_relative_seat"] == 0),
        1.0,
        tf.clip_by_value(
            1.0
            - tf.reduce_sum(
                tf.cast(inputs["opponent_discard_count"][:, 1:], tf.float32),
                axis=-1,
            )
            / 6.0,
            0.0,
            5.0 / 6.0,
        ),
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
    mixed = tf.reduce_logsumexp(
        tf.stack(
            [
                logits + tf.math.log(1.0 - strength[:, None]),
                pooled + tf.math.log(strength[:, None]),
            ],
            axis=-1,
        ),
        axis=-1,
    )
    return tf.where(
        tf.reduce_any(groups, axis=1) & (strength[:, None] > 0), mixed, logits
    )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class FinalCandidates(tf.keras.layers.Layer):
    def call(self, values):
        candidates, context, selected = values
        return tf.concat(
            [candidates, tf.gather(context, selected[:, 0])],
            axis=-1,
        )


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class ResponseKan(tf.keras.layers.Layer):
    def call(self, values):
        scores, inputs = values
        # A response kan refers to the triggering discard's base tile.
        return tf.concat(
            [
                scores[:, 111:260],
                tf.gather(
                    scores[:, 261:295],
                    tf.cast(inputs["trigger_tile_id"], tf.int32),
                    batch_dims=1,
                )[:, None],
            ],
            axis=-1,
        )


def policy_loss(labels, logits, mean=losses.mean_valid):
    nll = tf.add_n(
        [
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.maximum(tf.cast(labels[name], tf.int32), 0),
                logits=tf.cast(logits[name], tf.float32),
            )
            * tf.cast(labels[name] >= 0, tf.float32)
            for name in actions.TRAIN_POLICY_HEADS
        ]
    )
    return mean(
        nll,
        tf.reduce_any(
            tf.stack(
                [labels[name] >= 0 for name in actions.TRAIN_POLICY_HEADS]
            ),
            axis=0,
        ),
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


def greedy_choices(log_probability, heads):
    """Marginal branch selection on device, with lowest-index tie breaks."""
    chosen = tf.argmax(log_probability, axis=1, output_type=tf.int32)
    for head, levels in (
        (1, ((37, 37),)),
        (2, ((1, 111, 37, 1), (1, 37, 37, 37, 37, 1))),
    ):
        scores = log_probability[:, : sum(levels[0])]
        for sizes in levels:
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


def offline_loss_terms(inputs, outputs, mean=losses.mean_valid):
    metrics = {}
    metrics["policy_loss"] = policy_loss(inputs[1], outputs, mean=mean)
    metrics["policy_prior_loss"] = mean(
        tf.add_n(
            [
                outputs[name + "_prior_loss"]
                * tf.cast(inputs[1][name] >= 0, tf.float32)
                for name in actions.TRAIN_POLICY_HEADS
            ]
        ),
        tf.reduce_any(
            tf.stack(
                [inputs[1][name] >= 0 for name in actions.TRAIN_POLICY_HEADS]
            ),
            axis=0,
        ),
    )
    metrics["completion_yaku_loss"] = mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.maximum(
                tf.cast(inputs[1]["round_completion_yaku"], tf.float32), 0
            ),
            logits=outputs["round_completion_yaku_logits"],
        ),
        inputs[1]["round_completion_yaku"] >= 0,
    )
    metrics["completion_score_loss"] = mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.maximum(
                tf.cast(inputs[1]["round_completion_score"], tf.int32), 0
            ),
            logits=outputs["round_completion_score_logits"],
        ),
        inputs[1]["round_completion_score"] >= 0,
    )
    return metrics


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class UnpackCandidates(tf.keras.layers.Layer):
    def call(self, values):
        packed, selected, board = values
        return tf.scatter_nd(
            selected,
            packed,
            (tf.shape(board)[0], 295, packed.shape[-1]),
        )


class RouteMask(tf.keras.layers.Layer):
    def call(self, values):
        logits, inputs, selected = values
        hands = tf.stack(
            [
                selected[:, 0],
                tf.cast(
                    tf.gather_nd(
                        actions.candidate_hand_indices(inputs),
                        selected,
                    ),
                    tf.int32,
                ),
            ],
            axis=1,
        )
        allowed = payment_scoring.support(
            tf.gather_nd(tf.cast(inputs["candidate_melds"], tf.float32), hands)
        )
        # Outside routes include sequence-free honroutou/chinroutou.
        return tf.where(
            tf.stack(
                [
                    allowed[..., 1],
                    allowed[..., 18],
                    allowed[..., 10] | allowed[..., 11],
                ],
                axis=-1,
            ),
            logits,
            -1e9,
        )


def ron_risk_labels(labels, selected, structure):
    """Legal discard ron, including unknown amounts but excluding triple ron."""
    legal = tf.gather_nd(
        tf.transpose(labels["ron_legal"], [0, 2, 1]),
        tf.stack(
            [
                selected[:, 0],
                tf.minimum(tf.cast(structure[:, -1], tf.int32), 36),
            ],
            axis=1,
        ),
    )
    risk = legal == 1
    return risk & ~tf.reduce_all(risk, axis=1)[:, None]


def auxiliary_loss_terms(inputs, outputs, mean=losses.mean_valid):
    features, labels, _ = inputs
    selected = outputs["selected"]
    hand_indices = actions.candidate_hand_indices(features)
    affinity = tf.cast(
        tf.gather_nd(
            labels["candidate_yaku_affinity"],
            tf.stack(
                [
                    selected[:, 0],
                    tf.cast(tf.gather_nd(hand_indices, selected), tf.int32),
                ],
                axis=1,
            ),
        ),
        tf.float32,
    )
    terms = {
        "route_affinity_loss": losses.candidate_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.maximum(affinity, 0.0),
                logits=outputs["route_affinity_logits"],
            ),
            affinity >= 0,
            selected[:, 0],
            mean=mean,
        ),
    }
    risk = ron_risk_labels(labels, selected, outputs["structure"])
    terms["candidate_ron_loss"] = losses.candidate_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(risk, tf.float32),
            logits=outputs["candidate_ron_logits"],
        ),
        tf.broadcast_to(
            actions.discard_decisions(features, selected)[:, None],
            tf.shape(risk),
        ),
        selected[:, 0],
        mean=mean,
    )
    return terms
