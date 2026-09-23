# Nejimakidori

Mahjong model training and inference, Mahjong Soul replay review and browser
control, and three isolated Discord-summoned friendly accounts.

- [Model card](MODEL_CARD.md): records, training and game evaluation.
- [Dataset pipeline](log_dataset/README.md): archive download, rank enrichment
  and corpus rebuild.
- [Training pipeline](train.py): rebuild-or-train, evaluate and export one
  candidate.
- [Services](service/README.md): model HTTP API and training dashboard.
- [Review](review/README.md): replay review and live display.
- [Live](live/README.md): browser controllers and vision.
- [Friendly bots](bots/README.md): isolated account fleet and Discord commands.
- [Audit archive](../nejimakidori-audits/README.md): dated investigations,
  benchmarks and supporting evidence; not operational instructions.

## Verification

Run from this repository root using the shared Python environment. Tests use
fixtures and temporary state; browser and HTTP scenarios need localhost access.
They do not need production accounts, model weights or GPUs.

```sh
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 \
  OMP_NUM_THREADS=2 python -m pytest -q
node --test integration/*.test.mjs
```

Python integration scenarios exercise real replay/features, data conversion,
learning/checkpoint/export, service concurrency, friendly fleet
isolation and live-controller recognition. Node scenarios exercise protocol,
review/cache/grading, HTTP controls and actual browser cache preservation.
Chrome cache tests use disposable profiles and synthetic login storage.

Software checks do not establish playing strength. Full GPU training, matched
playing-strength evaluation and upstream account acceptance are separate work.
