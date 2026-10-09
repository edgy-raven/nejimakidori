# Model API

Start the service with the [v1 quick start](../README.md#run-v1). It accepts
stateless HTTP requests and returns JSON. The default local address is
`http://127.0.0.1:8792`.

## Requests

```sh
curl --fail http://127.0.0.1:8792/health
curl --fail http://127.0.0.1:8792/
```

`/health` reports the loaded model revision and contract. `/` describes the
model's actual input/output tensor names, shapes and dtypes, batch limit and
request fields. Use this schema when constructing tensor requests.

| Method and path | JSON body fields | Response fields |
| --- | --- | --- |
| `GET /health` | — | `ok`, `service`, `model`, `model_contract`, `revision`, `max_concurrent_requests` |
| `GET /` | — | Model metadata, tensor schemas and request/response descriptions. |
| `POST /predict` | `instances` | `inference` |
| `POST /predict-live` | `round`, `actor` | `decision`, `inference` |
| `POST /predict-replay` | `round`, `event_count` | `position`, `inference` |
| `POST /replay-deal-ins` | `rounds`, `seat` | `deal_in_tiles` |

POST requests require `Content-Type: application/json` and exactly the fields
listed above. `instances` is a nonempty list of feature dictionaries matching
the schema returned by `/`. Round objects use the browser replay format
consumed by [log_dataset.review](../log_dataset/review.py); they are not raw
MJAI event lists. See [web_adapter.py](web_adapter.py) for round conversion,
position metadata and decision responses.

For a saved request body:

```sh
curl --fail-with-body http://127.0.0.1:8792/predict-live \
  -H 'Content-Type: application/json' --data-binary @request.json
```

Request errors return `{"error": {"code": "...", "message": "..."}}`, with
optional `context`. Invalid requests or positions return 400; a position with
no decision returns 409; non-JSON requests return 415. Request bodies are
limited to 64 MiB. Four worker threads share one loaded model.

## Configuration

| Environment variable | Meaning | Default |
| --- | --- | --- |
| `MODEL_PATH` | Path to the matching SavedModel. | Required; `/model` in the runtime image. |
| `HOST` | Bind address. | `127.0.0.1`; `0.0.0.0` inside the runtime image. |
| `PORT` | HTTP port. | `8792` |
| `GPU_MEMORY_LIMIT_MB` | TensorFlow memory limit per visible GPU. | Unset |
| `ALLOWED_ORIGINS` | Comma-separated browser origins allowed by CORS. | Empty |

The service has no authentication layer. The quick start publishes it on
loopback; remote deployments need their own access control. CORS configures
browser access, not authentication.

## Decision semantics

Calls first select pass/chi/pon/kan by summed legal probability, then a chi
pattern where applicable, then a follow-up discard. Riichi first selects
whether to declare, then its discard. The full action retains its joint
probability for difficulty calculations.

The service reconstructs each request independently. It owns inference, not
browser profiles or review accounts. The exported model also exposes a
policy-only signature for frozen-policy evaluation; HTTP analysis uses the
full predictions.
