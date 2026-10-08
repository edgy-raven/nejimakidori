# Nejimakidori

Mahjong model, dataset builder, training pipeline and local Mahjong Soul client.
The reviewer website and hosted bot orchestration are separate projects.

## Local play with released weights

Download the pinned [production weights](weights/README.md) into a separate
weights directory outside this checkout. The bundle contains `saved_model/`,
its matching `source/`, and a checksum manifest. The current
training source is candidate development code; it is not interchangeable with
production inference source.

Build the local model runtime using Docker BuildKit and the release directory:

```sh
docker build --build-context release="$RELEASE_PATH" \
  -f docker/Dockerfile.model --target runtime -t nejimakidori-model .
docker run --rm -p 127.0.0.1:8792:8792 \
  --mount type=bind,src="$RELEASE_PATH/saved_model",dst=/model,readonly \
  nejimakidori-model
```

This starts CPU inference. On a machine with NVIDIA Container Toolkit, add
`--gpus '"device=GPU-UUID"'` to select the serving device. The image owns source;
weights are mounted read-only. Check `http://127.0.0.1:8792/health` before starting
the [local controller](controller/README.md). Browser profiles and credentials
belong in local state, outside Git.

The release checksum and per-file manifest are tracked under `weights/`.

## Dataset and training

[`dataset.py`](dataset.py) is the dataset job; [`train.py`](train.py) owns
training, verified export and evaluation. Packages own their implementations.

```sh
python dataset.py --archives /path/to/enriched-archives \
  --reserve /path/to/reserve.json --output /path/to/dataset
python train.py --data /path/to/dataset/data --output /path/to/run \
  --beliefs /path/to/compatible-beliefs.weights.h5 \
  --opponent /path/to/release/saved_model --devices GPU-UUID
```

The reserve file is `{"games": [{"game_id": "..."}]}`; an empty list excludes
nothing. Archives and training data are not distributed with model weights.
Belief initialization must match the candidate architecture; the production
SavedModel is an inference artifact, not a substitute for that initialization.
A new full training run still requires compatible belief weights. See
[dataset inputs](log_dataset/README.md) and [training contracts](model/README.md).

Build the candidate job image independently of the release runtime:

```sh
docker build -f docker/Dockerfile.model --target training \
  -t nejimakidori-training .
```

Mount input/output artifacts into that image, choose its GPUs explicitly, and
pass `python dataset.py ...` or `python train.py ...` as its command. Exported
candidates include their own frozen source and do not deploy automatically.

## Dependencies and checks

Python 3.13, a C++ compiler, Node.js 22 and Chromium are the development baseline.
The Dockerfile builds `riichienv==0.4.10+duplicate` from pinned upstream source
plus [`docker/riichienv.patch`](docker/riichienv.patch), using Rust 1.92. The patch
includes duplicate-game behavior, observation changes and its upstream tests.
Nyanten's vendored source and license are under `log_dataset/nyanten/vendor/`.

For a native environment, check out RiichiEnv commit
`479c1faeb33d082965eef8198f63261a79c0fce3`, apply the patch, and install it with
`python -m pip install /path/to/patched/RiichiEnv`. Then install:

```sh
python -m pip install -r model/requirements.txt \
  -r controller/vision/requirements.txt pytest
npm ci --prefix controller
python smoke.py small
python smoke.py big
```

Small checks cover model and local-client contracts. Big adds dataset,
training/export, evaluation, HTTP, vision and browser flows. Test processes use
CPU inference and temporary state; HTTP/browser checks need localhost access.

[Model card](MODEL_CARD.md) · [Model API](service/README.md)
