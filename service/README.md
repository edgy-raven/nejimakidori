# Application services

This directory owns the model HTTP API, live/review adapters and training
dashboard. It imports model implementation modules for feature encoding and
inference.
The `model` package owns training, evaluation and export. It does not import
the service.
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

The service requires the joint actor/critic export, including board heads.
An evaluation-only export with four policy heads cannot provide this contract.
Export completed joint weights and verify policy parity against the evaluated
SavedModel using a batched feature NPZ:

```sh
python export_policy.py --weights /path/to/final.weights.h5 \
  --policy /path/to/evaluated/saved_model --example /path/to/features.npz \
  --output /path/to/release
```

The output directory must be new. The export contains trained variables and
inference graphs only; critic scoring constraints derive from public features.
Package the matching `model/`, `log_dataset/`, `feature_vector.py` and
`service/` together under `source/` in the release:

```sh
cd /path/to/release
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

The model contract is `joint_policy_payment_critic`. Startup checks input names,
shapes and dtypes against the feature producer, plus the full output signature.
Board estimates remain available. `terminal_payment_probabilities` and
`structured_outcome` describe a mixture over the active legal policy, weighted
by its action probabilities. They exclude honba; settlement standard deviations
use the stated independent-cell approximation. They are not the old learned
state-payment head or a forecast conditioned on the recommended action.

Each prediction includes `attack_policy_probabilities` and
`defense_policy_probabilities`, each with 295 global candidate slots. These are
softmax distributions from the trained specialist decision heads, normalized
over legal candidates in the active decision phase; inactive slots are zero.
For per-tile riichi displays, combine corresponding discard slots in the
37–73 and 74–110 branches. These are policy preferences, not deal-in risks or
expected point values. The matching export exposes the two masked policy
logit arrays in addition to the final policy and critic.

Each prediction also contains `action_values`: the global candidate action ID,
four relative-seat expected point changes and three immediate ron probabilities.
Metadata defines the action ID ranges; response daiminkan uses
`261 + trigger_tile_id`. The exported signature retains each candidate's full
payment distribution. The service returns compact summaries of those arrays.
Critic decoding runs in chunks of 16 actions to bound intermediate memory.
`/predict` accepts discretionary decision phases (discard, response, kan,
riichi); forced wins are handled by `/predict-live` without model inference.

Discard policies have 37 tile classes (red fives remain distinct); riichi
policies have 74 classes for dama/riichi × tile. Karagiri and tsumogiri are one
policy/critic action. `discard_origin_logits` is a separately supervised
auxiliary with two branches (no declaration/declaration) and two origin classes
(tedashi/tsumogiri); it does not affect action selection. Execution uses the
drawn physical copy when its tile class is selected. Existing exports with
38/76 policy classes do not satisfy this source contract.

Replay and live responses share discard options containing
`action`, `tile`, `tsumogiri`, `all_shanten`, `all_ukeire_count` and
`all_upgrade_count`. `tsumogiri` describes canonical physical execution, not
a second policy choice. These reuse calculated model features; unused wait lists,
normal-form analysis and payment details are omitted. Replay context retains
`actual_shanten` for grading.

## Training dashboard

The monitor reads `model.train`'s `candidate/run_config.json` phase budget,
complete records in `training.log`, and per-phase candidate CSV metrics.

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

## Container deployment

`compose.yml` runs `service.serve` from source captured in the model image,
with the joint SavedModel mounted at `/model` and the NVIDIA GPU available.
Set host `MODEL_PATH` to the joint export's `saved_model` directory; the default
is the published full-budget `releases/joint-critic-300852` export. Rebuild the model image
with the matching feature producer for each release. Review caches include the
model's content revision, even when the mounted path stays `/model`.

The image compiles the bundled native shanten implementation and installs the
same `riichienv==0.4.10+duplicate` wheel used by training. `RIICHIENV_WHEELS`
selects its build context, defaulting to
`/home/fen/projects/riichienv/target/wheels`. Model serving does not mount the
legacy runtime, auxiliary kyuushu model or training records.
`TENSORBOARD_ROOT` selects the dashboard run directory.
The controller image runs both ranked and friendly browser controllers.

The image entry point adds wheel-provided CUDA libraries to the loader path.
On the current host, the installed NVIDIA userspace and CDI firmware paths
do not match the still-loaded 595.84 kernel module. The prepared runtime
override uses explicit GPU 0 devices and matching driver libraries without
modifying the host driver:

```sh
docker compose -p nejimakidori -f compose.yml \
  -f ../codex_env/critic-serving-gpu.compose.yml up -d --no-deps model
```

This is a cutover command, not a validation command. The override selects the
tested `nejimakidori-model:candidate-300852` image. Remove the host-specific
override after the installed and loaded driver versions have been reconciled;
its 595.84 libraries must match the loaded module.

The `web` service serves both the live page at `/live/` and replay review at
`/review/` on port 8793. `tensorboard` and `training-monitor` share the active
run directory and use port 8794. `reverse-ssh` forwards port 8793 and
`tensorboard-reverse-ssh` forwards port 8794 to Thoth.

Run the stack from the repository root after arranging a cutover from any
currently running host services:

```sh
docker compose -p nejimakidori up --build -d
docker compose -p nejimakidori ps
docker compose -p nejimakidori logs -f model ranked-controller web tensorboard
```

The ranked controller retains its current autoqueue setting and saved account
profile. Avoid starting its container while the existing ranked service is
still running.

## Joint-training TensorBoard

`python -m service.critic_dashboard --source <candidate>/tensorboard
--output <run>/dashboard` tails the trainer's events without restarting it.
Use `--once` for a single refresh. Point TensorBoard at `<run>/dashboard`.

The view shows training and validation imitation NLL/accuracy, board
representation loss and its components, critic loss/NLL/MSE/CRPS/severe-loss
NLL, and positions per second. Throughput uses 100-update wall-clock windows
with the effective batch from the candidate plan, including input loading
and validation/checkpoint pauses.
Combined critic loss is shown before its 0.1 training coefficient. Source
events remain intact; actor/AWR losses and raw counters are omitted from the
view. The projection refreshes every ten seconds and resumes without duplicate
steps. Custom Scalars groups the four areas; Scalars contains the same selected
series.

The current run's TensorBoard override, monitor PID and log are under
`/home/fen/projects/nejimakidori-audits/critic-current/joint-training/` as
`tensorboard.compose.yml`, `dashboard.pid`, and `dashboard.log`.

For the joint run's dashboard landing page, start
`python -m service.dashboard --joint --root <run>` alongside that projection.
`/tensorboard/` shows phase progress, exposures, a recent-speed ETA, headline
training/validation values, and three interactive charts. Hover shows raw values;
chart smoothing and a last-5,000-update range are available. Detailed TensorBoard
remains at `/tensorboard/charts/`. This view reads the candidate's plan/status
and the filtered scalar events; it does not modify or restart training.

The overview combines top-1 and top-3 imitation accuracy in one chart.
Imitation NLL and throughput charts are omitted; throughput is a status label. Top-3 uses the three highest-ranked choices in each active
policy head, with valid-decision counts summed across replicas and validation
batches. It begins at the metric-enabled checkpoint restart; older measurements
are not inferred. Board and critic loss components share one chart per group,
with component toggles. Normalization divides each component by its first
nonzero training observation (validation if training is absent), using the same
scale for both splits. Loss charts always use this normalization; hover
includes raw values. The accuracy and two loss charts are stacked vertically. This display normalization does not change training loss weights.
