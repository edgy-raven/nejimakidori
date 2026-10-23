"""Policy heads, action priors and final candidate scoring."""

import math

import tensorflow as tf

from model import actions, beliefs, encoder, losses, payment_scoring

POLICY_ENTROPY_WEIGHT = 0.01


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
        outputs, inputs = values
        riichi = tf.reshape(outputs["riichi_policy_logits"], (-1, 2, 37))
        response = outputs["response_policy_logits"]
        chii = tf.reshape(response[:, 1:112], (-1, 3, 37))
        pon = tf.reshape(response[:, 112:149], (-1, 1, 37))
        # Discard and riichi decisions have independently trained policies.
        # Use the active policy for the no-declaration branch.
        without = tf.where(
            (inputs["decision_phase"] == 3)[:, None],
            riichi[:, 0],
            outputs["discard_policy_logits"],
        )
        ready = inputs["discard_action_all_shanten"] == 1
        decision = tf.stack(
            [
                tf.reduce_logsumexp(tf.where(ready, without, -1e9), -1),
                tf.reduce_logsumexp(tf.where(~ready, without, -1e9), -1),
                tf.where(
                    inputs["decision_phase"] == 3,
                    tf.reduce_logsumexp(riichi[:, 1], -1),
                    -1e9,
                ),
            ],
            axis=-1,
        )
        return {
            **beliefs.opponent_summaries(outputs),
            "riichi_action_logits": tf.reduce_logsumexp(riichi, -1),
            "tenpai_decision_logits": decision,
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
    route = RouteMask(dtype="float32")([route, inputs, selected])
    ron = tf.keras.layers.Dense(3, dtype="float32", name="candidate_ron")(
        candidates
    )
    order = tf.keras.layers.Dense(1, dtype="float32", name="sakigiri_order")(
        candidates
    )
    outputs = {
        "route_affinity_logits": route,
        "candidate_ron_logits": ron,
        "sakigiri_scores": UnpackCandidates()([order, selected, hidden])[
            :, :, 0
        ],
    }
    final = FinalCandidates()(
        [
            tf.keras.layers.Concatenate()(
                [
                    candidates,
                    tf.keras.layers.Activation("sigmoid")(route),
                    ron,
                    order,
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
    outputs = {
        "win_decision_logits": PolicyMask(
            "win_decision_logits", name="win_legality", dtype="float32"
        )(
            [
                tf.keras.layers.Dense(
                    2,
                    dtype="float32",
                    name="win_decision_logits",
                )(hidden),
                inputs,
            ]
        )
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
        outputs[name] = values
    # Completion probabilities carry the same constraints into the policy.
    # Other heads retain their established representation.
    context = tf.keras.layers.Dense(128, activation="gelu")(
        tf.keras.layers.Concatenate()(
            [
                hidden,
                encoder.numeric_features(auxiliary_logits),
            ]
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
            [
                candidates,
                tf.gather(context, selected[:, 0]),
            ],
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


def policy_nll(labels, logits):
    losses = tf.add_n(
        [
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.maximum(tf.cast(labels[name], tf.int32), 0),
                logits=tf.cast(logits[name], tf.float32),
            )
            * tf.cast(labels[name] >= 0, tf.float32)
            for name in actions.TRAIN_POLICY_HEADS
        ]
    )
    return losses, tf.reduce_any(
        tf.stack([labels[name] >= 0 for name in actions.TRAIN_POLICY_HEADS]),
        axis=0,
    )


def policy_loss(labels, logits, mean=losses.mean_valid):
    return mean(*policy_nll(labels, logits))


def awr_policy_terms(nll, advantages, eligible, mean=losses.mean_valid):
    """Detached, capped positive regression weights on recorded actions."""
    log_weights = tf.minimum(tf.cast(advantages, tf.float32), tf.math.log(20.0))
    offset = tf.reduce_max(tf.where(eligible, log_weights, -1e30))
    replica = tf.distribute.get_replica_context()
    if replica is not None and replica.num_replicas_in_sync > 1:
        offset = tf.reduce_max(replica.all_gather(offset[None], axis=0))
    # A shared scale cancels from weighted CE and ESS. It prevents tiny
    # negative-advantage weights from underflowing, including their squares.
    weights = tf.stop_gradient(
        tf.exp(tf.where(eligible, log_weights - offset, -1e30))
    )
    weight_mean = mean(weights, eligible)
    return {
        "loss": mean(nll, eligible, weights),
        "weight_mean": weight_mean * tf.exp(offset),
        "weight_clipped_fraction": mean(
            tf.cast(
                advantages >= tf.math.log(20.0),
                tf.float32,
            ),
            eligible,
        ),
        "effective_sample_fraction": tf.math.divide_no_nan(
            tf.square(weight_mean), mean(tf.square(weights), eligible)
        ),
    }


def action_entropy(log_probability, head):
    """Explore continuations without rewarding their parent for having more.

    H(branch) + stop_gradient(P(branch)) * H(continuation | branch).
    The value remains joint entropy, but conditional uncertainty cannot push
    probability toward a call or declaration with more follow-up choices.
    """
    if head == "response_policy_logits":
        sizes = (1, 111, 37, 1)
    elif head == "riichi_policy_logits":
        sizes = (37, 37)
    else:
        return -tf.reduce_sum(
            tf.exp(log_probability) * log_probability, axis=-1
        )
    entropies = []
    for branch in tf.split(log_probability[:, : sum(sizes)], sizes, axis=-1):
        probability = tf.exp(branch)
        mass = tf.reduce_sum(probability, axis=-1)
        log_mass = tf.math.log(tf.maximum(mass, 1e-30))
        conditional = tf.math.divide_no_nan(
            -tf.reduce_sum(probability * (branch - log_mass[:, None]), axis=-1),
            mass,
        )
        entropies.append(
            -mass * log_mass + tf.stop_gradient(mass) * conditional
        )
    return tf.add_n(entropies)


def policy_entropy(labels, logits, mean=losses.mean_valid):
    entropies = []
    for name in actions.TRAIN_POLICY_HEADS:
        log_probability = tf.nn.log_softmax(
            tf.cast(logits[name], tf.float32), axis=-1
        )
        entropy = action_entropy(log_probability, name)
        entropies.append(tf.where(labels[name] >= 0, entropy, 0.0))
    return mean(
        tf.add_n(entropies),
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


def forward_kl(reference, current, mean=losses.mean_valid):
    reference = tf.stop_gradient(reference)
    return mean(
        tf.reduce_sum(tf.exp(reference) * (reference - current), axis=-1),
        tf.ones(tf.shape(current)[0]),
    )


def decision_kl(reference, current, labels):
    """Sum active heads per decision, then normalize globally across GPUs."""
    masks = tf.stack(
        [labels[name] >= 0 for name in actions.TRAIN_POLICY_HEADS],
        axis=-1,
    )
    divergences = []
    for name in actions.TRAIN_POLICY_HEADS:
        target = tf.stop_gradient(tf.nn.log_softmax(reference[name]))
        divergences.append(
            tf.reduce_sum(
                tf.exp(target) * (target - tf.nn.log_softmax(current[name])),
                axis=-1,
            )
        )
    return losses.mean_valid(
        tf.reduce_sum(
            tf.where(masks, tf.stack(divergences, axis=-1), 0.0), axis=-1
        ),
        tf.reduce_any(masks, axis=-1),
    )


def ppo_loss(
    *,
    current,
    behavior,
    chosen,
    advantages,
    clip,
    dual_clip,
    mean=losses.mean_valid,
):
    ratio = tf.exp(tf.gather(current - behavior, chosen, batch_dims=1))
    surrogate = tf.minimum(
        ratio * advantages,
        tf.clip_by_value(ratio, 1 - clip, 1 + clip) * advantages,
    )
    return -mean(
        tf.where(
            advantages < 0,
            tf.maximum(surrogate, dual_clip * advantages),
            surrogate,
        ),
        tf.ones_like(advantages),
    )


def offline_loss_terms(
    inputs, outputs, mean=losses.mean_valid, advantages=None
):
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
    metrics["policy_entropy"] = policy_entropy(inputs[1], outputs, mean=mean)
    metrics["policy_guidance_adjustment"] = tf.constant(0.0)
    if advantages is not None:
        nll, valid = policy_nll(inputs[1], outputs)
        guidance = awr_policy_terms(nll, advantages, valid, mean=mean)
        metrics.update(
            {"utility_" + name: value for name, value in guidance.items()}
        )
        metrics["policy_guidance_adjustment"] = (
            guidance["loss"] - metrics["policy_loss"]
        )
    for name in actions.TRAIN_POLICY_HEADS:
        prefix = name.removesuffix("_logits")
        valid = inputs[1][name] >= 0
        ranked_actions = tf.math.top_k(
            outputs[name], k=min(3, actions.ACTOR_OUTPUTS[name][0])
        ).indices
        recorded_action = tf.cast(inputs[1][name], tf.int32)
        matches = ranked_actions == recorded_action[:, None]
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.maximum(tf.cast(inputs[1][name], tf.int32), 0),
            logits=outputs[name],
        )
        # Counts make validation aggregation exact even when
        # the frequencies of calls and kan differ between batches/replicas.
        metrics[prefix + "_count"] = tf.reduce_sum(tf.cast(valid, tf.float32))
        metrics[prefix + "_correct"] = tf.reduce_sum(
            tf.cast(valid & matches[:, 0], tf.float32)
        )
        metrics[prefix + "_top3_correct"] = tf.reduce_sum(
            tf.cast(
                valid & tf.reduce_any(matches, axis=-1),
                tf.float32,
            )
        )
        metrics[prefix + "_nll_sum"] = tf.reduce_sum(tf.where(valid, nll, 0.0))
        if name == "discard_policy_logits":
            drawn = tf.where(
                inputs[0]["current_draw_is_red"] > 0,
                34 + tf.cast(inputs[0]["current_draw_tile_id"], tf.int32) // 9,
                tf.cast(inputs[0]["current_draw_tile_id"], tf.int32),
            )
            selected_draw = (inputs[0]["current_draw_valid"] > 0) & (
                recorded_action == drawn
            )
            for stratum, selected in {
                "focused": inputs[2]["focused"] > 0,
                "ordinary": inputs[2]["focused"] == 0,
                "drawn_tile": selected_draw,
                "other_tile": ~selected_draw,
            }.items():
                selected &= valid
                metrics[f"discard_{stratum}_count"] = tf.reduce_sum(
                    tf.cast(selected, tf.float32)
                )
                metrics[f"discard_{stratum}_correct"] = tf.reduce_sum(
                    tf.cast(selected & matches[:, 0], tf.float32)
                )
                metrics[f"discard_{stratum}_nll_sum"] = tf.reduce_sum(
                    tf.where(selected, nll, 0.0)
                )
    return metrics


def online_loss_terms(
    *, outputs, batch, current, count, clip, dual_clip, mean=losses.mean_valid
):
    return {
        "policy": ppo_loss(
            current=current,
            behavior=batch["behavior"][:count],
            chosen=batch["chosen"][:count],
            advantages=batch["advantages"][:count],
            clip=clip,
            dual_clip=dual_clip,
            mean=mean,
        ),
        "policy_prior_loss": mean(
            tf.gather(
                tf.stack(
                    [
                        outputs[name + "_prior_loss"][:count]
                        for name in actions.TRAIN_POLICY_HEADS
                    ],
                    axis=1,
                ),
                batch["head"][:count],
                batch_dims=1,
            ),
            tf.ones(count),
        ),
        "entropy": mean(
            tf.gather(
                tf.stack(
                    [
                        action_entropy(current, name)
                        for name in actions.TRAIN_POLICY_HEADS
                    ],
                    axis=1,
                ),
                batch["head"][:count],
                batch_dims=1,
            ),
            tf.ones(count),
        ),
    }


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


def order_loss(labels, scores, mean=losses.mean_valid):
    pair = tf.cast(labels["sakigiri_pair"], tf.int32)
    values = tf.gather(scores, tf.maximum(pair, 0), batch_dims=1)
    return mean(
        tf.nn.softplus(values[:, 1] - values[:, 0]),
        tf.reduce_all(pair >= 0, axis=1),
        tf.maximum(labels["sakigiri_weight"], 0.0),
    )


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
    terms["sakigiri_order_loss"] = order_loss(
        labels, outputs["sakigiri_scores"][:, :37], mean
    )
    return terms
