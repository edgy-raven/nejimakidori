# Model API

`service.serve` owns stateless inference for local controllers and remote review
clients. It does not own browser sessions, reviewer accounts or websites.

| Route | Contract |
| --- | --- |
| `/health` | Model revision and serving contract. |
| `/predict` | Batched visible tensors to predictions. |
| `/predict-live` | Round and actor to a legal decision and predictions. |
| `/predict-replay` | Recorded round and event cursor to a review position. |
| `/replay-deal-ins` | Recorded rounds and seat to legal ron tiles before discards. |

`MODEL_PATH`, `HOST`, `PORT`, `GPU_MEMORY_LIMIT_MB` and `ALLOWED_ORIGINS`
configure the service. Default standalone port: 8792. Four inference workers
share the model; each request owns its reconstructed game state. Shared cached
round conversion and reconstruction belong to `log_dataset.review`;
`service.web_adapter` translates model API requests and responses.

Calls select pass/chi/pon/kan by summed legal probability, then a chi pattern
when applicable, then its best discard. Riichi selects declaration versus no
declaration before its discard. Difficulty retains the complete action's joint
probability.

Source and weights must share a feature/output contract. Production local play
uses the frozen source supplied with its release. The `main` checkout contains
v1 training and inference code; phasic
development lives on `v2`. See the [root instructions](../README.md).
