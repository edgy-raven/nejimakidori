# Model catalog

An explorable reference for the **current working source**: public inputs,
network architecture, prediction contracts and the match/root training loop.
It is not an inventory of the checkpoint currently deployed in production.

## Browse

Serve this directory as static files and open `index.html`. The interface supports
feature search and grouping, architecture connections and individual blocks,
prediction inspection,
training stages and exact source excerpts. Hosting and proxy configuration belong
to the operations repository; the catalog itself has no host-specific settings.

## Contents

| File | Purpose |
| --- | --- |
| [index.html](index.html), [app.js](app.js), [style.css](style.css) | Static explorer |
| [build.py](build.py) | Schema extraction, source indexing and validation |
| [annotations.json](annotations.json) | Reviewed meanings, encodings and architecture narrative |
| [requirements.txt](requirements.txt) | Pinned build-time LaTeX-to-MathML renderer |
| `catalog.json` | Generated data consumed by the explorer |

The generator reads the native source schema and verifies exact names, order,
shapes and dtypes against the installed `_native_collection.INPUTS`. Every field
must have a description, encoding and group. Removed fields are absent from the current schema and catalog.
Field references include exact native producer and
Python consumer locations; dynamic board/river routing is identified explicitly.
The four candidate-hand index tensors are gather routes, excluded from learned
board covariates. Dora is represented by tile multiplicities; indicator-slot
IDs and their validity mask are not input fields.
Named references are source references, not proof of an independent learned
projection. Hidden-hand targets are excluded from the inference feature list.

The 17-field serving output contract is extracted from `model/inference.py`,
with serving actor membership and policy signatures from `model/actions.py`.
Internal belief and completion heads remain in the network but are not exported
unless a serving consumer needs them. Internal prediction features and training
loss terms are listed separately. These features feed downstream policy or
critic computations; removing them is an architectural ablation, not export
cleanup. Output row axes distinguish public positions from active
critic candidates. Model modules also include their classes, functions and
literal layer-construction source, so dimensions can be checked directly.

The architecture diagram contains 39 blocks and 71 explicit dataflow edges;
the 12-component overview remains available as a separate scale. Block details
expose 109 operators. Repeated suit, river, board and tile-attention blocks have
separate entries for each independently parameterized depth. Each operator
records shapes, residual connections, parameter sharing and a source location
scoped to the owning class or function. The four-player critic remains one
shared descriptor projection; future queued designs are not displayed as
implemented layers. The current board input projection has shape 9055 × 512.
Edges cite their implementation and distinguish states, input features, logits,
constraints and the initial-belief residual connection. The graph preserves
parallel actor heads and critic factors rather than chaining a module inventory.

Block descriptions embed selected equations as pre-rendered MathML. The
mathematics section contains rendered equations for
legal policy normalization, critic candidate mapping, marginalized observed
transfer likelihood, settlement utility, continuation, MMD, structural
constraints, shared player projections, tile attention and local encoder/policy/
critic operators. LaTeX remains authoring data only. Symbols have inline
MathML definitions. Actor and critic overlays on the block map include explicit
objective nodes and reviewed backward/stop edges. Each of the 108 gradient
paths is also a searchable atlas item with a stable phase/kind/endpoints ID,
a source citation and a rendered local VJP or stopped-path equation. Actor and
critic layers show each path as a visible gradient card connected to its source
and destination blocks; selecting the card opens its description. The initial
map shows critic gradients. Forward-only viewing remains available. Every block has a phase
status, including partial updates for grouped auxiliary heads, detached targets,
frozen critic weights and disconnected policy branches. Directional critic
probe stops are separate from ordinary differentiable likelihood paths. They
describe the current actor and critic passes; no alternative architecture is
implied. Equations are reviewed annotations tied to their implementation source;
review them when changing the corresponding mathematics.

All cited code is bundled as text from the public model, dataset/native and
training trees. No weights, host configuration, credentials or private operations
source are included. A content digest covers the annotations, generator and
bundled source. Historical production exports may use different inputs or layers.

## Regenerate and check

From the repository root, with the native extension installed:

```sh
python -m pip install -r model/catalog/requirements.txt
python -m model.catalog.build
python -m model.catalog.build --check
```

`--check` verifies exact reproducibility against the current source and fails on
missing tensor annotations, a stale native extension, missing output coverage,
broken architecture edges, cyclic block flow, missing block/operator coverage
or invalid source
references. It imports no TensorFlow,
loads no checkpoint and allocates no GPU. An alternate `--output PATH` can be used
when packaging the static files elsewhere. Regenerate after changing source,
annotations or source documentation.

MathML is generated and XML-validated at build time using the pinned
`latex2mathml` package. The browser needs no renderer dependency or CDN. Gradient
item identifiers and edge kinds are also validated.
Block equation selections, complete phase status coverage and reversed dataflow
connections for gradient edges are checked against the same producer contract.
