"""Observable decision categories with bounded, prevalence-based emphasis."""

import enum

import numpy


class Category(enum.IntFlag):
    FAST_CALLS = 1
    TANYAO = 2
    FLUSH = 4
    OUTSIDE = 8
    THREAT = 16
    PLACEMENT = 32


class Sampling(enum.IntEnum):
    NATURAL = 0
    FOCUSED = 1
    PLACEMENT_GAP = 2
    MIXED = 3


MIXTURE = (
    (Sampling.NATURAL, 2),
    (Sampling.FOCUSED, 1),
    (Sampling.PLACEMENT_GAP, 1),
)
NAMES = tuple(category.name.lower() for category in Category)
EMPHASIS = (1, 2, 2, 1, 1, 1)
PRIORITIES = numpy.array(
    [
        sum(
            weight
            for category, weight in zip(Category, EMPHASIS)
            if mask & category
        )
        for mask in range(1 << len(NAMES))
    ],
    numpy.int32,
)
