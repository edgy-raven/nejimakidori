# Application services

This directory owns the model HTTP API, live/review adapters and training
dashboard. It imports model implementation modules for feature encoding and
inference.
The root `train.py` owns corpus-to-candidate training and evaluation.
Neither imports the service.
The browser UI lives in [review](../review/README.md); controllers live in
[live](../live/README.md).

## Model API

From the repository root:

```sh
python -m pip install -r service/requirements.txt
MODEL_PATH=/path/to/release/saved_model python -m service.serve
python -m pytest integration/test_service_flows.py -q
```

The API defaults to `127.0.0.1:8792`; configure `HOST`, `PORT`,
`GPU_MEMORY_LIMIT_MB` and `ALLOWED_ORIGINS` as needed. Routes are `/`, `/health`,
`/predict`, `/predict-replay` and `/predict-live`.

Deploy a separate copy of `service/` beside the promoted model's frozen source:

```sh
cd /path/to/release
ln -s source/model model
PYTHONPATH=source MODEL_PATH=/path/to/release/saved_model python -m service.serve
```

Promotion neither bundles nor starts the service. Checkout edits do not change
running releases. Each request owns its game state; the API stores no account
or browser sessions.

Four workers share model weights across inference and adapter routes; extra
requests wait. Cancellation retains the worker until computation finishes.
Health checks stay on the HTTP loop and report `max_concurrent_requests: 4`.

Action selection uses raw logits; response probabilities are display data.
Native-legal ron, tsumo and kyuushu are accepted automatically without inference.
Kan declines reuse packed features; East-only context enters the initial pack.

Replay and live responses share discard options containing
`action`, `tile`, `tsumogiri`, `all_shanten`, `all_ukeire_count` and
`all_upgrade_count`. These reuse calculated model features; unused wait lists,
normal-form analysis and payment details are omitted. Replay context retains
`actual_shanten` for grading.

## Training dashboard

Start the monitor and dashboard against the same run directory:

```sh
python -m service.training_monitor --root /path/to/run
python -m service.dashboard --root /path/to/run
```

The dashboard at `127.0.0.1:8794/tensorboard/` reads atomically replaced
`monitor_status.json` and TensorBoard events. It shows completed/total games,
progress, games/hour and ETA for the current evaluator launch. Candidate and
reference results include placement rates, rank points and paired 95%
difference intervals; rate differences use percentage points.

TensorBoard's Custom Scalars charts place paired bootstrap difference bounds
on the reference baseline while plotting actual performance. Game charts use
completed games; training charts use updates, start collapsed and disable
smoothing. The regular Scalars tab retains all diagnostics. The monitor
backfills losses and throughput from existing logs; it does not control training.
Only full runs chart game strength. Run the service from its documented entry point.
