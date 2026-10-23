"""Same-completion yaku evidence imposes score tails, including exposed dora."""

import numpy
import tensorflow as tf

from integration import feature_reference
from log_dataset import _native_collection, scoring
from model import payment_scoring


def test_open_flush_requires_baiman_and_keeps_higher_score_estimates():
    public = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    public["meld_valid"][0, 2, :3] = 1
    public["meld_type"][0, 2, :3] = [1, 1, 2]
    public["meld_tile_valid"][0, 2, :3, :3] = 1
    public["meld_tile_ids"][0, 2, :3, :3] = [[0, 1, 2], [2, 3, 4], [8, 8, 8]]
    public["dora_multiplicity"][0, 8] = 1
    feature_reference.public_scoring_features(public)
    public = {name: tf.constant(value) for name, value in public.items()}
    # Explicit, already-consistent marginals isolate the value coupling.
    evidence = numpy.full((1, 4, 2, 23), -1e9, numpy.float32)
    for name, probability in (("chinitsu", 0.92), ("honitsu", 0.06)):
        evidence[0, 2, :, scoring.NAMES.index(name)] = numpy.log(
            probability / (1 - probability)
        )
    evidence = tf.Variable(evidence)
    raw = numpy.zeros((1, 4, 2, len(payment_scoring.SCORES)), numpy.float32)
    raw[..., payment_scoring.SCORES.index((4, 30))] = 8
    raw = tf.Variable(raw)
    layer = payment_scoring.CompletionScoreCoupling()
    with tf.GradientTape() as tape:
        masked = payment_scoring.CompletionScoreMask()([raw, public])
        logits = layer([masked, evidence, public])
        loss = -tf.nn.log_softmax(logits)[
            0, 2, 0, payment_scoring.SCORES.index((8, 0))
        ]
    gradient = tape.gradient(loss, [raw, evidence])
    assert all(numpy.isfinite(value).all() for value in gradient)
    assert all(tf.linalg.global_norm([value]) > 0 for value in gradient)
    probability = tf.nn.softmax(logits).numpy()
    numpy.testing.assert_allclose(probability.sum(-1), 1, atol=1e-6)
    baiman = payment_scoring.SCORE_HAN >= 8
    mangan = payment_scoring.SCORE_HAN >= 5
    numpy.testing.assert_allclose(
        probability[0, 2, :, baiman].sum(0), 0.92, atol=1e-6
    )
    # The mutually exclusive half/full flush alternatives both imply mangan.
    assert numpy.all(probability[0, 2, :, mangan].sum(0) >= 0.98 - 1e-6)
    assert not probability[0, 2, :, payment_scoring.SCORE_HAN < 4].any()
    # Coupling must not create zero-likelihood training targets in otherwise
    # feasible ordinary tiers: uncertain evidence still permits other hands.
    feasible = (masked.numpy()[0, 2] > -1e8) & (payment_scoring.SCORE_HAN > 0)
    assert probability[0, 2][feasible].min() > 0

    # No downward cap: strong pre-existing baiman evidence survives.
    high = raw.numpy()
    high[..., payment_scoring.SCORES.index((8, 0))] = 20
    masked = payment_scoring.CompletionScoreMask()([tf.constant(high), public])
    before = tf.nn.softmax(masked).numpy()[0, 2]
    after = tf.nn.softmax(layer([masked, evidence, public])).numpy()[0, 2]
    numpy.testing.assert_allclose(before, after, atol=1e-6)

    # Traced inference uses precisely the same coupling as eager training.
    traced = tf.function(layer.call)([masked, evidence, public])
    numpy.testing.assert_allclose(
        tf.nn.softmax(traced).numpy()[0, 2], after, atol=1e-6
    )


def test_closed_value_bonuses_and_true_yakuman_share_probability_mass():
    public = {
        name: numpy.zeros((1, *spec["shape"]), spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    public["riichi_state"][0, 1] = 2
    feature_reference.public_scoring_features(public)
    public = {name: tf.constant(value) for name, value in public.items()}
    evidence = numpy.full((1, 4, 2, 23), -1e9, numpy.float32)
    evidence[..., scoring.NAMES.index("chinitsu")] = numpy.log(0.8 / 0.2)
    evidence[..., scoring.NAMES.index("yakuman")] = numpy.log(0.1 / 0.9)
    logits = payment_scoring.CompletionScoreMask()(
        [tf.zeros((1, 4, 2, len(payment_scoring.SCORES))), public]
    )
    probability = tf.nn.softmax(
        payment_scoring.CompletionScoreCoupling()(
            [logits, tf.constant(evidence), public]
        )
    ).numpy()
    yakuman = payment_scoring.SCORE_HAN == 0
    numpy.testing.assert_allclose(
        probability[..., yakuman].sum(-1), 0.1, atol=1e-6
    )
    # Undeclared hands can still open: the guaranteed flush value is five.
    assert (
        probability[0, 0, 0, (payment_scoring.SCORE_HAN >= 5) | yakuman].sum()
        >= 0.9 - 1e-6
    )
    # Riichi fixes closed chinitsu: 6+1 ron, 6+1+1 tsumo.
    assert (
        probability[0, 1, 0, (payment_scoring.SCORE_HAN >= 6) | yakuman].sum()
        >= 0.9 - 1e-6
    )
    assert (
        probability[0, 1, 1, (payment_scoring.SCORE_HAN >= 8) | yakuman].sum()
        >= 0.9 - 1e-6
    )
    # Independent compatible labels are not added into an invented event.
    evidence.fill(-1e9)
    for name in ("sanankou", "sankantsu", "toitoi"):
        evidence[..., scoring.NAMES.index(name)] = numpy.log(0.8 / 0.2)
    probability = tf.nn.softmax(
        payment_scoring.CompletionScoreCoupling()(
            [logits, tf.constant(evidence), public]
        )
    ).numpy()
    assert numpy.isfinite(probability).all()
    numpy.testing.assert_allclose(probability.sum(-1), 1, atol=1e-6)
    assert not probability[..., yakuman].any()


def test_yaku_coupling_survives_extreme_logits_without_breaking_masks():
    public = {
        name: tf.constant(numpy.zeros((2, *spec["shape"]), spec["dtype"]))
        for name, spec in _native_collection.INPUTS.items()
    }
    rng = numpy.random.default_rng(13)
    evidence = payment_scoring.CompletionMask()(
        [tf.constant(rng.normal(0, 10, (2, 4, 2, 23)), tf.float32), public]
    )
    masked = payment_scoring.CompletionScoreMask()(
        [
            tf.constant(
                rng.normal(0, 100, (2, 4, 2, len(payment_scoring.SCORES))),
                tf.float32,
            ),
            public,
        ]
    )
    with tf.GradientTape() as tape:
        tape.watch([masked, evidence])
        logits = payment_scoring.CompletionScoreCoupling()(
            [masked, evidence, public]
        )
        loss = tf.reduce_sum(
            tf.nn.softmax(logits)
            * tf.range(len(payment_scoring.SCORES), dtype=tf.float32)
        )
    assert all(
        numpy.isfinite(value).all()
        for value in tape.gradient(loss, [masked, evidence])
    )
    probability = tf.nn.softmax(logits).numpy()
    assert numpy.isfinite(probability).all()
    numpy.testing.assert_allclose(probability.sum(-1), 1, atol=1e-6)
    assert not probability[masked.numpy() < -1e8].any()
