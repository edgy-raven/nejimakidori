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


@functools.lru_cache(maxsize=1)
def native():
    LIBRARY.parent.mkdir(exist_ok=True)
    sources = sorted(NYANTEN.glob("*.cpp"))
    with (LIBRARY.parent / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not LIBRARY.exists() or LIBRARY.stat().st_mtime_ns < max(
            path.stat().st_mtime_ns
            for path in (
                pathlib.Path(__file__),
                *sources,
                *NYANTEN.rglob("*.hpp"),
            )
        ):
            subprocess.run(
                [
                    "c++",
                    "-O3",
                    "-DNDEBUG",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    "-I" + str(NYANTEN / "vendor"),
                    *map(str, sources),
                    "-o",
                    str(LIBRARY.with_suffix(".tmp")),
                ],
                check=True,
            )
            os.replace(LIBRARY.with_suffix(".tmp"), LIBRARY)
    dll = ctypes.CDLL(str(LIBRARY))
    dll.nyanten_shanten.argtypes = [ctypes.c_char_p]
    dll.nyanten_shanten.restype = ctypes.c_uint8
    dll.nyanten_preserves_completion.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_uint,
        ctypes.c_uint64,
    ]
    dll.nyanten_preserves_completion.restype = ctypes.c_bool
    dll.nyanten_families.argtypes = [ctypes.c_char_p]
    dll.nyanten_families.restype = ctypes.c_uint32
    dll.nyanten_yaku_shanten.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    dll.nyanten_yaku_shanten.restype = ctypes.c_uint32
    dll.nyanten_draw.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    dll.nyanten_draw.restype = ctypes.c_uint64
    dll.nyanten_tenpai_paths.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    dll.nyanten_tenpai_paths.restype = ctypes.c_bool
    return dll


def calculate_families(counts):
    values = native().nyanten_families(bytes(map(int, counts)))
    return tuple(((values >> shift) & 255) - 1 for shift in (0, 8, 16))


def calculate_shanten(counts, all_forms=True):
    if all_forms:
        return native().nyanten_shanten(bytes(map(int, counts))) - 1
    return calculate_families(counts)[0]


def calculate_yaku_shanten(counts, melds=()):
    """Exact tanyao, flush and outside distances; 14 is impossible.

    Counts exclude fixed melds. Each meld contains three or four base tile IDs.
    Distances are missing draws minus one, with physical copy limits but no
    public-availability constraint. Flush includes honitsu and chinitsu.
    Outside includes chanta, junchan, honroutou and chinroutou: every group
    and pair contains a terminal or honor. Closed seven pairs is included.
    """
    values = native().nyanten_yaku_shanten(
        bytes(map(int, counts)),
        bytes(
            int(tile)
            for meld in melds
            for tile in (*meld, *([34] * (4 - len(meld))))
        ),
        len(melds),
    )
    return tuple(((values >> shift) & 255) - 1 for shift in (0, 8, 16))


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
TILE_CODES = {tile: code for code, tile in enumerate(TILE_NAMES)}


def mjai_tile_code(tile):
    return TILE_CODES[tile]


def base_id(tile):
    return int(
        BASE_IDS[mjai_tile_code(tile) if isinstance(tile, str) else tile]
    )


def is_red(tile):
    return (mjai_tile_code(tile) if isinstance(tile, str) else tile) >= 34


def tenpai_paths(counts, unavailable, tile_ids=None):
    """Copy-weighted shortest routes, or None if the search is incomplete."""
    hands = numpy.empty((4096, 34), numpy.uint8)
    weights = numpy.empty(4096, numpy.uint64)
    size = ctypes.c_uint()
    complete = native().nyanten_tenpai_paths(
        bytes(map(int, counts)),
        bytes(map(int, unavailable)),
        (
            1 << 34
            if int(numpy.sum(counts)) % 3 == 1
            else sum(
                1 << int(cut)
                for cut in (
                    numpy.flatnonzero(counts)
                    if tile_ids is None
                    else numpy.unique(BASE_IDS[numpy.asarray(tile_ids, int)])
                )
            )
        ),
        3,
        512,
        hands.ctypes.data,
        weights.ctypes.data,
        len(hands),
        ctypes.byref(size),
    )
    return (hands[: size.value], weights[: size.value]) if complete else None
