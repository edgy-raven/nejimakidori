# Source setup

Use Linux with Python 3.13, Rust 1.92 or newer, a C++17 compiler and Git.
Clone the source and install its dependencies:

```sh
git clone --branch main --depth 1 \
  https://github.com/edgy-raven/nejimakidori.git
cd nejimakidori
python3.13 -m venv .venv
. .venv/bin/activate

git clone https://github.com/smly/RiichiEnv.git ../riichienv
git -C ../riichienv checkout 479c1faeb33d082965eef8198f63261a79c0fce3
git -C ../riichienv apply "$PWD/docker/riichienv.patch"
python -m pip install ../riichienv
python -m pip install -r requirements.txt
python -m log_dataset.build_native_collection
```

Use weights compatible with the model architecture. Start with the
[dataset guide](log_dataset/README.md), then follow the
[training guide](training/README.md) to train and evaluate a checkpoint.
