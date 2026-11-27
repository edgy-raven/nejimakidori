"""Independent NumPy decoding oracle for the packed record contract."""

import math

import numpy

from log_dataset import records


def decode(name, raw, definition):
    size = math.prod(definition["shape"])
    dtype = numpy.dtype(definition["dtype"])
    if name in records.PACKING:
        width, offset = records.PACKING[name]
        packed = numpy.frombuffer(
            raw.ljust((size * width + 7) // 8, b"\0"), dtype=numpy.uint8
        )
        bits = numpy.unpackbits(packed, bitorder="little")[: size * width]
        values = (bits.reshape(size, width) * (1 << numpy.arange(width))).sum(1)
        return (values - offset).astype(dtype).reshape(definition["shape"])
    return numpy.frombuffer(
        raw.ljust(size * dtype.itemsize, b"\0"), dtype=dtype
    ).reshape(definition["shape"])
