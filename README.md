# Nejimakidori

Nejimakidori (ねじまき鳥) is a four-player riichi mahjong AI. It recommends
moves and estimates opponents' hands, winning chances and hand values.
You can run it as a local HTTP service, connect it to a Mahjong Soul client,
or use the training tools to develop your own model.

[Download](https://github.com/edgy-raven/nejimakidori/releases/latest) ·
[HTTP API](service/README.md) ·
[Mahjong Soul client](controller/README.md) ·
[Windows overlay](controller/overlay/README.md)

## Run the model

You need Git, Docker and curl. These commands run the model on CPU and expose
the API only on your computer:

```sh
git clone --branch nejimakidori-20261110 --depth 1 \
  https://github.com/edgy-raven/nejimakidori.git
cd nejimakidori

mkdir -p artifacts
cd artifacts
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-20261110/nejimakidori-20261110.tar.gz
curl -fLO https://github.com/edgy-raven/nejimakidori/releases/download/nejimakidori-20261110/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf nejimakidori-20261110.tar.gz
cd ..

docker build -f docker/Dockerfile.model --target runtime \
  -t nejimakidori-model .
docker run --rm --name nejimakidori-model \
  -p 127.0.0.1:8792:8792 \
  --mount type=bind,src="$PWD/artifacts/nejimakidori-20261110/saved_model",dst=/model,readonly \
  nejimakidori-model
```

Leave that process running. In another terminal, check that the model is ready:

```sh
curl --fail http://127.0.0.1:8792/health
```

For NVIDIA inference, install NVIDIA Container Toolkit and add `--gpus all`
to `docker run`. To select one GPU, also pass
`-e CUDA_VISIBLE_DEVICES=GPU-UUID`, using its UUID from `nvidia-smi -L`.

Next, follow the [client guide](controller/README.md) to play through Chromium,
or the [API guide](service/README.md) to request predictions from your own code.
The [Windows overlay](controller/overlay/README.md) displays advice without
playing moves automatically.

## Build without Docker

Use Linux with Python 3.13, Rust 1.92 or newer, a C++17 compiler and Git.
From the repository root:

```sh
python3.13 -m venv .venv
. .venv/bin/activate

git clone https://github.com/smly/RiichiEnv.git ../riichienv
git -C ../riichienv checkout 479c1faeb33d082965eef8198f63261a79c0fce3
git -C ../riichienv apply "$PWD/docker/riichienv.patch"
python -m pip install ../riichienv
python -m pip install -r model/requirements.txt mahjong==2.0.0
python -m log_dataset.build_native_collection
```

Download and extract the model as shown above, then start the service:

```sh
MODEL_PATH="$PWD/artifacts/nejimakidori-20261110/saved_model" \
  python -m service.serve
```

Use the source tag and weights from the same release. The release archive
includes a source snapshot and checksums; see [model files](weights/README.md).

## Training and development

| Task | Guide |
| --- | --- |
| Understand the predictions and model architecture | [Model](model/ARCHITECTURE.md) |
| Build a training dataset | [Datasets](log_dataset/README.md) |
| Train, export and evaluate a model | [Training](training/README.md) |
| Run value-function experiments | [Experiments](training/EXPERIMENTS.md) |
| Modify native features or rollout collection | [Native interface](log_dataset/native_collection/README.md) |

To run the checks, install pytest and the client dependencies:

```sh
python -m pip install pytest
npm ci --prefix controller
python smoke.py small
python smoke.py big
```

The full suite includes CPU performance checks. Client tests can also run with
`npm run test:controller --prefix controller`.
