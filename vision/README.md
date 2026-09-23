# Vision library

This directory contains only the vision worker, its runtime dependencies and
reference assets used by the live/friendly browser entry points.

Run the worker from the repository root with `python mahjongsoul.py --serve`.

`datasets/tile_classify/` is the small labeled tile reference bank used
during recognition. `static/` contains button, lobby, room and river
references plus the web tile atlas.

Offline exporters, calibration commands, benchmarks, validation copies and
generated overlays have been removed. Regression screenshots and tests live
outside this runtime directory; shared reference screenshots are linked back
here so tests use the same images as the vision worker.
