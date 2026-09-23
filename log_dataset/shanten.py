"""Structural Mahjong distances and available improving continuations."""

import ctypes
import fcntl
import functools
import os
import pathlib
import subprocess

import numpy

BASE_IDS = numpy.array(list(range(34)) + [4, 13, 22, 0])
NYANTEN = pathlib.Path(__file__).parent / "nyanten"
LIBRARY = NYANTEN / ".native" / "analysis.so"
ACTION_DTYPE = numpy.dtype(
    [
        ("shanten", numpy.int8, 34),
        ("ukeire_mask", numpy.uint64, 34),
        ("ukeire_count", numpy.uint16, 34),
        ("upgrade_mask", numpy.uint64, 34),
        ("upgrade_type_count", numpy.uint8, 34),
        ("upgrade_tile_count", numpy.uint16, 34),
        ("upgrade_weighted_gain", numpy.uint16, 34),
        ("completion_probability", numpy.float32, 34),
        ("completion_valid", numpy.uint8, 34),
    ],
    align=True,
)


@functools.lru_cache(maxsize=1)
def native():
    LIBRARY.parent.mkdir(exist_ok=True)
    with (LIBRARY.parent / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (
            not LIBRARY.exists()
            or LIBRARY.stat().st_mtime_ns
            < (NYANTEN / "normal_uke.cpp").stat().st_mtime_ns
        ):
            subprocess.run(
                [
                    "c++",
                    "-O3",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    "-I" + str(NYANTEN / "vendor"),
                    str(NYANTEN / "normal_uke.cpp"),
                    "-o",
                    str(LIBRARY.with_suffix(".tmp")),
                ],
                check=True,
            )
            os.replace(LIBRARY.with_suffix(".tmp"), LIBRARY)
    dll = ctypes.CDLL(str(LIBRARY))
    dll.nyanten_shanten.argtypes = [ctypes.c_char_p]
    dll.nyanten_shanten.restype = ctypes.c_uint8
    dll.nyanten_families.argtypes = [ctypes.c_char_p]
    dll.nyanten_families.restype = ctypes.c_uint32
    dll.nyanten_draw.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    dll.nyanten_draw.restype = ctypes.c_uint64
    dll.nyanten_analyze_modes.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    dll.nyanten_analyze_modes.restype = None
    dll.nyanten_tenpai_neededness.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    dll.nyanten_tenpai_neededness.restype = ctypes.c_bool
    return dll


def calculate_families(counts):
    values = native().nyanten_families(bytes(map(int, counts)))
    return tuple(((values >> shift) & 255) - 1 for shift in (0, 8, 16))


def calculate_shanten(counts, all_forms=True):
    if all_forms:
        return native().nyanten_shanten(bytes(map(int, counts))) - 1
    return calculate_families(counts)[0]


def analyze_actions(
    counts,
    tile_ids=None,
    known_unavailable_counts=None,
    modes=(0, 1),
    calculate_upgrades=False,
    calculate_completion=False,
):
    cuts = (
        numpy.flatnonzero(counts)
        if tile_ids is None
        else numpy.unique(BASE_IDS[numpy.asarray(tile_ids, dtype=int)])
    )
    values = numpy.zeros(len(modes), dtype=ACTION_DTYPE)
    native().nyanten_analyze_modes(
        bytes(map(int, counts)),
        bytes(
            map(
                int,
                (
                    counts
                    if known_unavailable_counts is None
                    else known_unavailable_counts
                ),
            )
        ),
        sum(1 << int(cut) for cut in cuts),
        calculate_upgrades,
        calculate_completion,
        bytes(modes),
        len(modes),
        values.ctypes.data,
    )
    return {
        mode: {
            name: (
                values[index][name].astype(numpy.int64)
                if values.dtype[name].base.kind in "iu"
                else values[index][name]
            )
            for name in ACTION_DTYPE.names
        }
        for index, mode in enumerate(modes)
    }


def improving(counts, unavailable=None, all_forms=True):
    value = native().nyanten_draw(bytes(map(int, counts)), int(all_forms))
    tiles = [tile for tile in range(34) if value & (1 << tile)]
    unseen = 4 - numpy.asarray(counts if unavailable is None else unavailable)
    return (value >> 34) - 1, tiles, int(unseen[tiles].sum())


TILE_NAMES = tuple(
    [f"{rank}{suit}" for suit in "mps" for rank in range(1, 10)]
    + ["E", "S", "W", "N", "P", "F", "C"]
    + ["5mr", "5pr", "5sr"]
)


def mjai_tile_code(tile):
    return TILE_NAMES.index(tile)


def base_id(tile):
    return int(
        BASE_IDS[mjai_tile_code(tile) if isinstance(tile, str) else tile]
    )


def is_red(tile):
    return (mjai_tile_code(tile) if isinstance(tile, str) else tile) >= 34


def tenpai_participation(
    counts, unavailable, tile_ids=None, max_draws=3, search_nodes=512
):
    """Native weighted original-copy retention on shortest tenpai routes."""
    cuts = (
        numpy.flatnonzero(counts)
        if tile_ids is None
        else numpy.unique(BASE_IDS[numpy.asarray(tile_ids, dtype=int)])
    )
    values = numpy.zeros(34, numpy.float32)
    complete = native().nyanten_tenpai_neededness(
        bytes(map(int, counts)),
        bytes(map(int, unavailable)),
        sum(1 << int(cut) for cut in cuts),
        max_draws,
        search_nodes,
        values.ctypes.data,
    )
    return values if complete else numpy.full(34, -1.0)
