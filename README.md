# Nejimakidori v2

A four-player riichi mahjong AI that predicts moves, opponent hands and
current-hand outcomes. This repository contains the model, dataset builder,
HTTP API and local Mahjong Soul client.

This branch is v2 source. The [v1 release](weights/README.md) uses a different
architecture and must run with its matching source.

## Setup

Use a Python 3.13 environment with Rust 1.92 or newer, a C++17 compiler and Git.
From this repository's root, install the patched simulator and build the native
feature engine:

```sh
git clone https://github.com/smly/RiichiEnv.git ../riichienv
git -C ../riichienv checkout 479c1faeb33d082965eef8198f63261a79c0fce3
git -C ../riichienv apply "$PWD/docker/riichienv.patch"
python -m pip install ../riichienv
python -m pip install -r model/requirements.txt mahjong==2.0.0 pytest
python -m log_dataset.build_native_collection
```

Training also requires CUDA GPUs. The browser client needs Node.js 22, Chromium
and its [additional dependencies](controller/README.md#install-and-configure).

## Guides

| Task | Guide |
| --- | --- |
| Understand the model | [Architecture](model/ARCHITECTURE.md) |
| Prepare game records | [Datasets](log_dataset/README.md) |
| Train or restart a run | [Training and recovery](model/README.md) |
| Export and compare models | [Evaluation](model/README.md#export-and-evaluate) |
| Run the HTTP API | [Serving](service/README.md) |
| Play through Chromium | [Local client](controller/README.md) |
| Change native features | [Native interface](log_dataset/native_collection/README.md) |

## Checks

```sh
python smoke.py small
python smoke.py big
```

The big suite includes CPU performance checks. Client checks require its
installed dependencies; they can also be run separately with
`npm run test:controller --prefix controller`.
