"""Structural families, scored yaku and guaranteed value stay distinct."""

import numpy
import riichienv
import tensorflow as tf

from integration import feature_reference
from log_dataset import _native_collection, call_states, scoring
from model import (
    actions,
    beliefs,
    candidate_predictions,
    critic,
    losses,
    network,
    payment_scoring,
)
from training import self_play


def test_belief_inputs_match_parent_and_accepted_call_forward_graph(
    monkeypatch,
):
    model = network.JointPolicyPayment()
    policy = model.actor.policy
    model.actor.call_choice.build(policy.output["policy_context"].shape)
    local = []
    weighted = []
    choices = []
    nodes = {}

    def capture(layer, buffer):
        original = layer.call

        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            buffer.append(dict(result) if isinstance(result, dict) else result)
            return result

        monkeypatch.setattr(layer, "call", call)

    candidate_heads = model.actor.candidate_heads

    def capture_candidates(features, outputs):
        candidate_heads(features, outputs)
        local.append(dict(outputs))

    monkeypatch.setattr(model.actor, "candidate_heads", capture_candidates)
    capture(policy.get_layer("shanten_weighted_ukeire"), weighted)
    capture(model.actor.call_choice, choices)

    @tf.function(autograph=False)
    def forward(rows):
        local.clear()
        weighted.clear()
        choices.clear()
        nodes.clear()
        outputs = model.actor.call(rows)
        state = model.candidate_context(rows, outputs)
        names = {
            *beliefs.BELIEF_HEADS,
            *beliefs.COMPLETION_HEADS,
            *actions.TRAIN_POLICY_HEADS,
            "route_affinity_logits",
            "candidate_ron_logits",
            "opponent_wait_logits",
            "ron_eligibility_logits",
            "policy_context",
        }
        for scope, result, joint in zip(
            ("parent", "child"), local, weighted, strict=True
        ):
            nodes.update(
                {(scope, key): result[key] for key in names & result.keys()}
            )
            nodes[scope, "shanten_weighted_ukeire"] = joint
        nodes["parent", "call_choice"] = choices[0]
        nodes["joint", "response_policy_logits"] = outputs[
            "response_policy_logits"
        ]
        nodes["joint", "critic_state"] = state
        return state

    forward.get_concrete_function(call_states.input_specs())
    by_tensor = {tensor.ref(): key for key, tensor in nodes.items()}
    assert len(by_tensor) == len(nodes)
    for (scope, target), tensor in nodes.items():
        # Walk data values to the first named boundary on each path. A size
        # or zero-like tensor carries no prediction value. StopGradient does.
        sources = set()
        pending = list(tensor.op.inputs)
        seen = set()
        while pending:
            parent = pending.pop()
            if parent.ref() in by_tensor:
                sources.add(by_tensor[parent.ref()])
            elif parent.ref() not in seen and parent.op.type not in (
                "Shape",
                "ShapeN",
                "Size",
                "Rank",
                "ZerosLike",
                "OnesLike",
            ):
                pending.extend(parent.op.inputs)
            seen.add(parent.ref())
        direct = beliefs.BELIEF_INPUTS.get(target, ())
        if scope == "child" and target in (
            *beliefs.BELIEF_HEADS,
            "opponent_wait_logits",
            "ron_eligibility_logits",
        ):
            expected = {("parent", target)}
        elif target == "response_policy_logits":
            expected = {
                ("parent" if name == "call_choice" else "child", name)
                for name in direct
            }
        elif target == "critic_state":
            expected = {
                (board, name)
                for board in ("parent", "child")
                for name in direct
            }
        else:
            expected = {(scope, name) for name in direct}
        assert sources == expected, (scope, target)
    assert beliefs.BELIEF_INPUTS.keys() <= {name for _, name in nodes}


def test_route_families_include_yakuman_shapes_and_kan_completions():
    for tiles, family in (
        ([0, 0, 0, 8, 8, 8, 9, 9, 9, 17, 17, 17, 18], "chanta"),
        ([0, 0, 0, 8, 8, 8, 9, 9, 9, 27, 27, 27, 18], "chanta"),
        ([1, 1, 1, 3, 3, 3, 10, 10, 10, 12, 12, 12, 23], "tanyao"),
        ([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8], "honitsu"),
    ):
        counts = numpy.bincount(tiles, minlength=34)
        values = scoring.hand_yaku_fractions(
            counts=counts,
            unavailable=counts,
            melds=[],
            player_wind=1,
            round_wind=0,
            names=("tanyao", "honitsu", "chanta"),
        )
        assert values[("tanyao", "honitsu", "chanta").index(family)] == 1
    # Two kans give 16 physical tiles at completion; the convenience scorer
    # would append the winning tile again because 16 % 3 == 1.
    flags = scoring.completion_flags(
        tiles=[36, 37, 38, 72, 73, 74, 108],
        melds=[
            riichienv.Meld(riichienv.MeldType.Ankan, [0, 1, 2, 3], False),
            riichienv.Meld(riichienv.MeldType.Ankan, [32, 33, 34, 35], False),
        ],
        player_wind=1,
        round_wind=0,
        require_all=False,
    )
    assert flags[scoring.NAMES.index("yakuman")] == 1
    assert flags[scoring.NAMES.index("honroutou")] == 0


def test_named_outside_requires_sequence_but_structural_route_does_not():
    melds = numpy.zeros((2, 34, 6), numpy.int8)
    melds[0, [0, 8, 9, 17], 1] = 1
    melds[1, 0, 0] = 2  # Two identical chii still occupy two group slots.
    allowed = payment_scoring.support(tf.constant(melds)).numpy()
    assert not allowed[0, [10, 19]].any()
    assert allowed[0, 11]
    assert not allowed[1, 14]  # Cannot fit three kans beside two chii.
    inputs = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["meld_tiles"][...] = 37
    inputs["candidate_meld_tiles"][...] = 34
    inputs["last_tedashi_tile_id"][...] = 37
    inputs["candidate_meld_tiles"][0, 0] = [0, 8, 9, 17]
    inputs["candidate_meld_types"][0, 0] = 1
    feature_reference.public_scoring_features(inputs)
    inputs = {name: tf.constant(value) for name, value in inputs.items()}
    route = candidate_predictions.RouteMask()(
        [tf.zeros((1, 3)), inputs, tf.constant([[0, 0]])]
    )
    assert route.numpy()[0, 2] == 0


def test_guaranteed_han_adds_public_dora_to_declared_yaku():
    inputs = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["meld_tiles"][...] = 37
    inputs["candidate_meld_tiles"][...] = 34
    inputs["last_tedashi_tile_id"][...] = 37
    inputs["meld_type"][0, 1, 0] = 4
    inputs["meld_tiles"][0, 1, 0] = 8
    inputs["dora_multiplicity"][0, 8] = 1
    inputs["riichi_state"][0, 1] = 2
    feature_reference.public_scoring_features(inputs)
    inputs = {name: tf.constant(value) for name, value in inputs.items()}
    ordinary_below_haneman = (payment_scoring.SCORE_HAN > 0) & (
        payment_scoring.SCORE_HAN < 6
    )
    completion = tf.nn.softmax(
        payment_scoring.CompletionScoreMask()(
            [tf.zeros((1, 4, 2, len(payment_scoring.SCORES))), inputs]
        )
    ).numpy()
    # Four dora plus riichi plus menzen tsumo: at least six han.
    assert not completion[0, 1, 1, ordinary_below_haneman].any()
    terminal = tf.nn.softmax(
        payment_scoring.masked_score_logits(
            logits=tf.zeros((1, 24, len(payment_scoring.SCORES))),
            dealer=tf.constant([0]),
            meld_fu=payment_scoring.fixed_meld_fu(inputs),
            opened=tf.zeros((1, 4), tf.bool),
            riichi_required=tf.ones((1, 4), tf.bool),
            bonus=tf.cast(inputs["public_bonus"], tf.int32),
            yakuman=tf.ones((1, 4), tf.bool),
        )
    ).numpy()
    # Public four dora plus riichi: ron >= five, tsumo >= six han.
    assert not terminal[0, :12][payment_scoring.CELLS[:12, 2] == 1][
        ..., (payment_scoring.SCORE_HAN > 0) & (payment_scoring.SCORE_HAN < 5)
    ].any()
    assert not terminal[0, 12:][payment_scoring.CELLS[12:, 2] == 1][
        ..., ordinary_below_haneman
    ].any()
    # A riichi-only ron remains legal: dora is not itself a yaku.
    assert payment_scoring.minimum_han_mask(tf.constant(1)).numpy()[
        payment_scoring.SCORES.index((1, 30))
    ]


def test_total_score_learns_public_evidence_and_shares_tsumo_predictions():
    inputs = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["meld_tiles"][...] = 37
    inputs["candidate_meld_tiles"][...] = 34
    inputs["last_tedashi_tile_id"][...] = 37
    feature_reference.public_scoring_features(inputs)
    inputs = {name: tf.constant(value) for name, value in inputs.items()}
    public = critic.public_inputs(inputs, tf.constant([[0, 0]]))
    predictor = critic.PaymentCritic(
        belief_offset=0, shanten_offset=critic.BELIEF_WIDTH + 4, player_width=8
    )
    result = predictor.distributions(
        {
            **public,
            "state": tf.zeros(
                (1, critic.BELIEF_WIDTH + critic.PREDICTED_TAIL_WIDTH + 64)
            ),
        }
    )
    assert result["score_logits"].shape == (1, 24, len(payment_scoring.SCORES))
    for winner in range(4):
        shares = numpy.flatnonzero(
            (payment_scoring.CELLS[:, 0] == 1)
            & (payment_scoring.CELLS[:, 2] == winner)
        )
        numpy.testing.assert_allclose(
            result["score_probability"].numpy()[:, shares],
            numpy.repeat(
                result["score_probability"].numpy()[:, shares[:1]], 3, axis=1
            ),
        )
    # Public values must affect predictions even when legal score support
    # stays identical: two vs four fixed fu both round to the same fu floor.
    state = tf.zeros(
        (1, critic.BELIEF_WIDTH + critic.PREDICTED_TAIL_WIDTH + 64)
    )
    values = []
    for fu in (2, 4):
        values.append(
            predictor.distributions(
                {**public, "state": state, "public_fu": tf.fill((1, 4), fu)}
            )["score_logits"].numpy()
        )
    numpy.testing.assert_array_equal(values[0] > -1e8, values[1] > -1e8)
    assert numpy.max(numpy.abs(values[0] - values[1])) > 1e-6
    numpy.testing.assert_allclose(
        tf.reduce_sum(result["score_probability"], -1), 1, atol=1e-6
    )


def test_yakuhai_masks_follow_visible_copies_claims_and_exact_winds():
    inputs = {
        name: numpy.zeros((6, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    inputs["meld_tiles"][...] = 37
    inputs["candidate_meld_tiles"][...] = 34
    inputs["last_tedashi_tile_id"][...] = 37
    inputs["seat_wind"][:] = [0, 1, 2, 3]
    # Two dead white dragons (including revealed indicators) prevent a pon.
    inputs["known_unavailable_counts"][:2, 31] = 2
    # The offered second copy is still claimable by everyone but seat 1.
    inputs["trigger_valid"][1] = 1
    inputs["trigger_tile_id"][1] = 31
    inputs["trigger_source"][1] = 1
    # A fixed white triplet is retained for its owner, unavailable to others.
    inputs["known_unavailable_counts"][2, 31] = 3
    for row, tiles in ((2, [31]), (3, [0, 9, 18, 29]), (4, [0, 9, 18, 27])):
        for slot, tile in enumerate(tiles):
            inputs["meld_type"][row, 0, slot] = 2
            inputs["meld_tiles"][row, 0, slot, :3] = tile
            inputs["known_unavailable_counts"][row, tile] = 3
    # Our concealed pair isn't dead: opponents could receive later discards.
    inputs["known_unavailable_counts"][5, 31] = 2
    inputs["candidate_hand_counts"][5, 0, 31] = 2
    raw_logits = tf.Variable(tf.zeros((6, 4, 2, len(scoring.NAMES))))
    with tf.GradientTape() as tape:
        logits = payment_scoring.CompletionMask()(
            [
                raw_logits,
                {name: tf.constant(value) for name, value in inputs.items()},
            ]
        )
        # The same constrained logits feed completion-yaku training loss.
        loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(logits), logits=logits
            )
        )
    probability = tf.sigmoid(logits).numpy()
    gradients = tape.gradient(loss, raw_logits).numpy()
    assert numpy.isfinite(gradients).all()
    numpy.testing.assert_array_equal(gradients[probability == 0], 0)
    assert numpy.any(gradients[1, 0, :, scoring.NAMES.index("haku")] != 0)
    assert numpy.any(gradients[0, 0, :, scoring.NAMES.index("hatsu")] != 0)
    possible = probability[:, :, 0, scoring.NAMES.index("haku")] > 0
    numpy.testing.assert_array_equal(
        possible[:3],
        [
            [False, False, False, False],
            [True, False, True, True],
            [True, False, False, False],
        ],
    )
    assert possible[5].all()
    for name in ("seat_wind", "round_wind"):
        assert probability[3, 0, :, scoring.NAMES.index(name)].sum() == 0
        assert (probability[4, 0, :, scoring.NAMES.index(name)] > 0).all()


def test_constraint_learning_corrects_shared_features_without_moving_targets():
    # A small shared representation feeds two heads. Hard masking alone
    # cannot change the forbidden raw logit; the explicit loss must do so.
    shared = tf.Variable([[0.5, 1.0, -0.5]])
    target = tf.Variable([[0.5, -1e9, -0.5]])
    for binary in (True, False):
        with tf.GradientTape(persistent=True) as tape:
            raw = shared * 2
            penalty = tf.reduce_mean(
                losses.constraint_divergence(raw, target, binary=binary)
            )
        gradient = tape.gradient(penalty, shared)
        assert gradient[0, 1] > 0
        assert numpy.isfinite(gradient).all()
        assert tape.gradient(penalty, target) is None
        updated = shared - 0.1 * gradient
        assert updated[0, 1] < shared[0, 1]
        assert (
            tf.reduce_mean(
                losses.constraint_divergence(updated * 2, target, binary=binary)
            )
            < penalty
        )


def test_sparse_policy_padding_cannot_train_score_offsets():
    tf.keras.utils.set_random_seed(941)
    model = network.JointPolicyPayment()
    features = self_play.initial_features(941)
    scores = model.actor.policy.get_layer("final_scores")
    original = model.actor(features, training=False)
    legal = actions.legal_choices(features)["discard_policy_logits"]
    assert tf.reduce_any(legal) and tf.reduce_any(~legal)
    for offset in (-1000.0, 1000.0):
        scores.bias.assign([offset])
        with tf.GradientTape() as tape:
            shifted = model.actor(features, training=False)
            penalty = tf.reduce_sum(shifted["constraint_loss"])
        assert tape.gradient(penalty, scores.bias) is None
        numpy.testing.assert_array_equal(
            shifted["constraint_loss"], original["constraint_loss"]
        )
        numpy.testing.assert_allclose(
            tf.nn.softmax(shifted["discard_policy_logits"]),
            tf.nn.softmax(original["discard_policy_logits"]),
            atol=1e-5,
        )
    # Win logits mean accept/decline conditional on a legal win opportunity.
    # Non-winning states cannot provide negative win-policy supervision.
    with tf.GradientTape() as tape:
        penalty = tf.reduce_sum(
            model.actor(features, training=False)["constraint_loss"]
        )
    win = model.actor.policy.get_layer("win_decision_logits")
    assert tape.gradient(penalty, win.trainable_variables) == [None, None]
