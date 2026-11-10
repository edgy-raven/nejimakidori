# Nejimakidori

Nejimakidori (ねじまき鳥) is a four-player riichi mahjong model. This repository
contains the network, feature generation, dataset tools, training and
evaluation code, and released weights.

[Download weights](https://github.com/edgy-raven/nejimakidori/releases/latest) ·
[Training](training/README.md) · [Architecture](model/ARCHITECTURE.md)

## Download weights

Download the archive and checksum from the
[release](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-20261110):

```sh
mkdir -p artifacts
cd artifacts
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-20261110/nejimakidori-20261110.tar.gz
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-20261110/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf nejimakidori-20261110.tar.gz
cd ..
```

The archive includes `final.weights.h5` for training, a TensorFlow SavedModel,
matching source, and evaluation results. See [model files](weights/README.md)
for the contents and checksums.

## Build without Docker

Use Linux with Python 3.13, Rust 1.92 or newer, a C++17 compiler and Git.
Clone the source and install its dependencies:

```sh
git clone --branch nejimakidori-20261110 --depth 1 \
  https://github.com/edgy-raven/nejimakidori.git
cd nejimakidori
python3.13 -m venv .venv
. .venv/bin/activate

git clone https://github.com/smly/RiichiEnv.git ../riichienv
git -C ../riichienv checkout 479c1faeb33d082965eef8198f63261a79c0fce3
git -C ../riichienv apply "$PWD/docker/riichienv.patch"
python -m pip install ../riichienv
python -m pip install -r model/requirements.txt
python -m log_dataset.build_native_collection
```

Use the source and weights from the same release. Start with the
[dataset guide](log_dataset/README.md), then follow the
[training guide](training/README.md) to train and evaluate a checkpoint.

## Training container

The Docker image includes the patched simulator and native feature generator:

```sh
docker build -f docker/Dockerfile.model --target training \
  -t nejimakidori-training .
docker run --rm nejimakidori-training
```

The default command prints the trainer's options. Mount your datasets and
output directory, then pass a training command after the image name. NVIDIA
training requires NVIDIA Container Toolkit and `--gpus` on `docker run`.

## Guides

| Task | Guide |
| --- | --- |
| Understand model predictions and features | [Architecture](model/ARCHITECTURE.md) |
| Build a dataset | [Datasets](log_dataset/README.md) |
| Train, export and evaluate | [Training](training/README.md) |
| Run value-function experiments | [Experiments](training/EXPERIMENTS.md) |
| Modify native features or rollout collection | [Native interface](log_dataset/native_collection/README.md) |

## Checks

After installing the source dependencies:

```sh
python -m pip install pytest
python smoke.py small
python smoke.py big
```

The full suite includes CPU performance checks.
