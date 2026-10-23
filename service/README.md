# Serve the model

The HTTP service loads one SavedModel and handles stateless prediction requests.
Create a matching export with the [training guide](../model/README.md#export-and-evaluate),
then run from the repository root:

```sh
MODEL_PATH=/path/to/run/policy/saved_model python -m service.serve
curl --fail http://127.0.0.1:8792/health
```

`GET /health` reports the loaded revision. `GET /` lists its tensor schemas,
batch limit and request fields. Use the loaded model's schema when sending
feature tensors.

## Send a request

This example sends the bundled replay fixture to find decision positions for
seat 1. The service must already be running:

```sh
python - <<'PY' | curl --fail-with-body \
  http://127.0.0.1:8792/replay-decisions \
  -H 'Content-Type: application/json' --data-binary @-
import json
import pathlib

round_data = json.loads(
    pathlib.Path("integration/fixtures/review_reach_round.json").read_text()
)
print(json.dumps({"round": round_data, "seat": 1}))
PY
```

The response contains `event_counts`. Pass those cursors with the same `round`
and `seat` to `/predict-replay` to obtain annotations.

| Endpoint | Required JSON fields | Result |
| --- | --- | --- |
| `POST /predict` | `instances` | `inference`: predictions for a nonempty list of feature dictionaries. |
| `POST /predict-live` | `round`, `actor` | `decision` and `inference` for the current position. |
| `POST /replay-decisions` | `round`, `seat` | `event_counts`: decision cursors for that seat. |
| `POST /predict-replay` | `round`, `event_counts`, `seat` | `annotations` at the requested cursors. |
| `POST /replay-deal-ins` | `rounds`, `seat` | `deal_in_tiles` for the requested seat. |

Requests must contain exactly the listed fields. Round objects use the browser
replay format shown in the fixture, rather than raw MJAI event lists.
[log_dataset/review.py](../log_dataset/review.py) reconstructs them;
[web_adapter.py](web_adapter.py) builds decision responses.

Errors have the form `{"error": {"code": "...", "message": "..."}}`, sometimes
with additional `context`. Invalid requests or positions return 400, positions
without a decision return 409, and non-JSON requests return 415. Request bodies
are limited to 64 MiB.

## Configuration

| Variable | Meaning | Default |
| --- | --- | --- |
| `MODEL_PATH` | SavedModel directory. | Required; `/model` in the runtime image. |
| `HOST` | Bind address. | `127.0.0.1`; `0.0.0.0` in the runtime image. |
| `PORT` | HTTP port. | `8792` |
| `CUDA_VISIBLE_DEVICES` | GPUs available to TensorFlow. | Environment default. |
| `GPU_MEMORY_LIMIT_MB` | Memory limit per visible GPU. | Unset. |
| `ALLOWED_ORIGINS` | Comma-separated browser origins allowed by CORS. | Empty. |

The service has no authentication. Keep the default loopback binding for local
use; provide access control when exposing it remotely. CORS only controls
browser access.

## Decision semantics

Call selection proceeds through call type, chii pattern when needed, and
follow-up discard. Riichi selection chooses declaration first, then discard.
The full action retains its joint probability.

Replay cursors include legal call opportunities after opponents' discards.
`legal_response_actions` uses 0 for pass, 1 for chii, 2 for pon and 3 for open kan.
A subsequent draw establishes a recorded pass. If another player calls or wins,
the requested seat's intention is unknown: the annotation has
`response_censored: true` and empty `expert_decisions`. Do not grade that case
as a pass.
