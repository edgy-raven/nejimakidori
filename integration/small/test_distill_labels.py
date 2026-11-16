"""The phase constraint applies to the main choice, not kan tile selection."""

import types

import numpy
from log_dataset import distill_labels


def test_phase_masks_preserve_upstream_kan_selection_and_input_masks():
    teacher = distill_labels.Teacher.__new__(distill_labels.Teacher)
    teacher.context = types.SimpleNamespace(phase="DISCARD")
    teacher.inference = lambda obs, masks, invisible: masks
    kan = numpy.zeros(46, dtype=bool)
    kan[27] = True
    main = numpy.zeros(46, dtype=bool)
    main[[4, 37, 42, 43, 44]] = True
    for phase, expected in (
        ("DISCARD", [4]),
        ("RIICHI", [4, 37]),
        ("KAN", [4, 37, 42]),
        ("WIN", [4, 37, 42, 43, 44]),
    ):
        teacher.context.phase = phase
        result = teacher.infer_phase([None, None], [kan, main], None)
        numpy.testing.assert_array_equal(result[0], kan)
        numpy.testing.assert_array_equal(numpy.flatnonzero(result[1]), expected)
    teacher.context.phase = "RESPONSE"
    main[:] = True
    result = teacher.infer_phase([None], [main], None)
    numpy.testing.assert_array_equal(
        numpy.flatnonzero(result[0]), [38, 39, 40, 41, 42, 45]
    )
    assert main.all()
