"""Recorded temperatures track capped-weight ESS without batch chasing."""

import numpy
from training import temperature


def test_separate_temperatures_track_rolling_ess_and_ignore_invalid_rows():
    gaps = numpy.random.default_rng(941).normal(size=(256, 2)) * [2, 6]
    samples = numpy.column_stack([gaps, numpy.ones(256)])
    samples = numpy.concatenate([samples, [[1e6, -1e6, 0]]])
    controller = temperature.RollingTemperature(16)
    for step in range(640):
        before = controller.beta.copy()
        updated = controller.observe(samples)
        assert updated == ((step + 1) % 32 == 0)
        assert numpy.all(controller.beta >= before * 0.8 - 1e-6)
        assert numpy.all(controller.beta <= before * 1.25 + 1e-6)
    numpy.testing.assert_allclose(controller.ess, 0.7, atol=0.002)
    numpy.testing.assert_allclose(
        controller.ess,
        temperature.effective_fraction(gaps, controller.beta),
        atol=1e-6,
    )
    assert controller.beta[1] > controller.beta[0] * 2
    controller = temperature.RollingTemperature(16)
    for _ in range(32):
        controller.observe(
            numpy.column_stack([numpy.zeros((256, 2)), numpy.ones(256)])
        )
    numpy.testing.assert_array_equal(controller.beta, [16, 16])
    numpy.testing.assert_array_equal(controller.minimum_ess, [1, 1])
