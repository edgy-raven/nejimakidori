# Native hand analysis

[shanten.py](../shanten.py) loads the C++ implementation in
[analysis.cpp](analysis.cpp) through ctypes. It calculates shanten, improving
tiles, discard upgrades and bounded shortest-tenpai routes. Standard hands
use Nyanten; seven pairs and thirteen orphans are calculated by the local
wrapper.

The first call compiles `.native/analysis.so` with a C++17 compiler. A file
lock protects concurrent builders and the completed library replaces the old
one atomically. Changes to the wrapper, Python build recipe or vendored headers
trigger a rebuild in a fresh process. `.native/` is an ignored build cache.

## Vendored source

The six headers in `vendor/nyanten/standard/` are the complete dependency set
used by the wrapper. They are unchanged from
[Cryolite/nyanten](https://github.com/Cryolite/nyanten) commit
`581aa72ce53e0ffeb0ef8d1ed14df2521ca5fb9b`.
The source archive SHA-256 is
`9c77ad73c09785d7cc28624a96d7645f2476028bd1d77ca5d9b0f9790232cb46`.

Keep the generated lookup tables: they are required runtime headers, not
build leftovers. Upstream's all-family facade, special-hand headers, tests and
build system are not used here. The complete upstream
[license notice](vendor/LICENSE.md) is retained; its separately named GPL test
files are not included.
