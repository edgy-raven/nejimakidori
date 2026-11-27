# Hand analysis

[shanten.py](../shanten.py) exposes the C++ hand-analysis routines to Python.
Standard-hand shanten uses vendored Nyanten tables; the local wrapper adds
seven pairs, thirteen orphans, improving tiles and bounded completion searches.
[yaku_shanten.cpp](yaku_shanten.cpp) calculates tanyao, flush and outside-hand
distances.

## Build

A C++17 compiler is required. The first call through `shanten.py` builds
`.native/analysis.so`; you can trigger it explicitly from the repository root:

```sh
python -c 'from log_dataset import shanten; shanten.native()'
```

A file lock protects concurrent builders. Source, header or build-recipe changes
trigger recompilation in a fresh Python process. `.native/` is a build cache.
The [native feature engine](../native_collection/README.md) also uses these C++
calculations.

## Source and license

The six headers under `vendor/nyanten/standard/` come unchanged from
[Cryolite/nyanten](https://github.com/Cryolite/nyanten), commit
`581aa72ce53e0ffeb0ef8d1ed14df2521ca5fb9b`. The source archive SHA-256 is
`9c77ad73c09785d7cc28624a96d7645f2476028bd1d77ca5d9b0f9790232cb46`.

The generated lookup tables are required source headers. Preserve them and the
[upstream license notice](vendor/LICENSE.md). Upstream's separately named GPL
test files are not included in this vendored subset.
