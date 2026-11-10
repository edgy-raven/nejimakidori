# Training container

## Build

From the repository root:

```sh
docker build -f docker/Dockerfile.model --target training \
  -t nejimakidori-training .
docker run --rm nejimakidori-training
```

The default command prints candidate options. The image includes Python,
TensorFlow, the pinned patched simulator and the native feature generator.

## Run

Mount datasets, weights and an output directory, then pass a training command
after the image name. NVIDIA training requires NVIDIA Container Toolkit and
`--gpus` on `docker run`. See the [candidate command](../training/candidate/README.md).

## Files

| File | Purpose |
| --- | --- |
| [Dockerfile.model](Dockerfile.model) | Native build and training image |
| [riichienv.patch](riichienv.patch) | Required simulator changes |
| [model_entrypoint.py](model_entrypoint.py) | CUDA library discovery before executing the command |
