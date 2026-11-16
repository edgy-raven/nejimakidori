"""Slow ESS calibration over sampled recorded Q gaps, separately by source."""

import collections

import numpy

TARGET_ESS = 0.7


def effective_fraction(gaps, beta):
    log_weights = numpy.minimum(gaps / beta, numpy.log(20.0))
    weights = numpy.exp(log_weights - log_weights.max(axis=0))
    return weights.mean(axis=0) ** 2 / (weights**2).mean(axis=0)


class RollingTemperature:
    def __init__(self, initial):
        self.beta = numpy.full(2, initial, dtype=numpy.float32)
        self.samples = collections.deque(maxlen=128)
        self.updates = 0
        self.ess = numpy.ones(2)
        self.minimum_ess = numpy.ones(2)

    def observe(self, samples):
        self.samples.append(samples[samples[:, 2] > 0, :2])
        self.updates += 1
        if self.updates % 32:
            return False
        gaps = numpy.concatenate(self.samples)
        if len(gaps) < 1024:
            return False
        assert numpy.isfinite(gaps).all()
        grid = numpy.geomspace(0.01, 128.0, 65)
        for source in range(2):
            curve = effective_fraction(gaps[:, source, None], grid)
            self.minimum_ess[source] = curve.min()
            if numpy.ptp(curve) < 1e-6:
                self.ess[source] = curve[0]
                continue
            crossings = numpy.flatnonzero(curve <= TARGET_ESS)
            if len(crossings) and crossings[-1] < len(grid) - 1:
                lower = grid[crossings[-1]]
                upper = grid[crossings[-1] + 1]
                for _ in range(16):
                    middle = numpy.sqrt(lower * upper)
                    if effective_fraction(gaps[:, source], middle) < TARGET_ESS:
                        lower = middle
                    else:
                        upper = middle
                desired = upper
            else:
                desired = grid[numpy.argmin(numpy.abs(curve - TARGET_ESS))]
            self.beta[source] = numpy.clip(
                desired, self.beta[source] * 0.8, self.beta[source] * 1.25
            )
            self.ess[source] = effective_fraction(
                gaps[:, source], self.beta[source]
            )
        return True
