# Nejimakidori

Nejimakidori (ねじまき鳥) is a four-player riichi mahjong AI. It learns from
recorded games and combines a move policy with predictions of opponents'
hands, deal-in risk, yaku and hand value. A local Mahjong Soul client can use
it to play through a visible browser window.

**[v1 source](https://github.com/edgy-raven/nejimakidori/tree/nejimakidori-v1)**
· **[Download v1](https://github.com/edgy-raven/nejimakidori/releases/tag/nejimakidori-v1)**
· **[Evaluation](#evaluation)**

`main` maintains v1. The release tag preserves the exact published snapshot.

## Run v1

You need Git, Docker with BuildKit, and the GitHub CLI (`gh`) for this example.
The model runs on CPU; an NVIDIA GPU is optional.

```sh
git clone https://github.com/edgy-raven/nejimakidori.git
cd nejimakidori

export WEIGHTS_PATH="$HOME/.local/share/nejimakidori/weights"
mkdir -p "$WEIGHTS_PATH"
gh release download nejimakidori-v1 \
  --repo edgy-raven/nejimakidori --dir "$WEIGHTS_PATH" \
  --pattern nejimakidori-v1.tar.gz --pattern SHA256SUMS
(cd "$WEIGHTS_PATH" && sha256sum -c SHA256SUMS)
tar -xzf "$WEIGHTS_PATH/nejimakidori-v1.tar.gz" -C "$WEIGHTS_PATH"
export RELEASE_PATH="$WEIGHTS_PATH/nejimakidori-v1"

docker build --build-context release="$RELEASE_PATH" \
  -f docker/Dockerfile.model --target runtime -t nejimakidori-model .
docker run --rm --name nejimakidori-model -p 127.0.0.1:8792:8792 \
  --mount type=bind,src="$RELEASE_PATH/saved_model",dst=/model,readonly \
  nejimakidori-model
```

The last command stays running. In another terminal:

```sh
curl --fail http://127.0.0.1:8792/health
```

Then follow the **[Mahjong Soul client setup](controller/README.md)** to open
Chrome, sign in and enable autoplay. The client requires a graphical desktop,
Node.js and Python. You join games manually.

For NVIDIA inference, configure NVIDIA Container Toolkit and add
`--gpus '"device=GPU-UUID"'` to `docker run`, replacing `GPU-UUID` with a value
from `nvidia-smi -L`. The image uses the source bundled with the release;
weights stay in the separate directory mounted at `/model`.

## How it works

The model sees the player's hand and public table state. Its shared encoder
feeds opponent-belief heads, attack and defense specialists, a policy, and a
critic that forecasts how the current hand will end. Legal-action masks
restrict the policy's choices.

Training starts with recorded decisions. Advantage-weighted regression (AWR)
gives more weight to decisions the critic expects to improve final rank
points. Additional labels teach hidden tile counts, immediate ron risk,
completion yaku and scored hand value. Paired one-hand rollouts can supply
further supervision. Inference uses the trained policy without running those
rollouts for each move.

The [architecture reference](model/PAYMENT_CRITIC.md) explains the prediction
heads, labels and objectives. Belief counts obey expected hand sizes and tile
copy limits; they do not sample complete hidden hands. Final-placement
estimates use an approximate continuation model.

## Evaluation

V1 was evaluated in two 10,000-game matches. Each uses 2,500 duplicate seed
groups, with v1 playing once in each seat against three copies of the reference.

| Reference checkpoint | V1 ST3 advantage per game | 95% interval |
| --- | ---: | ---: |
| 83570 | +0.02 | −2.97 to +2.86 |
| 133712 | +2.27 | −0.65 to +5.21 |

ST3 is the Mahjong Soul Saint 3 Jade South rank-point utility used for
training, combining final score and placement. Both intervals include zero;
neither comparison establishes a positive ST3 advantage. The
[weights bundle](weights/README.md) includes the complete match summaries.

## Work with the source

| Task | Guide |
| --- | --- |
| Download or inspect a checkpoint | [Weights](weights/README.md) |
| Use the HTTP inference service | [Model API](service/README.md) |
| Play locally in Mahjong Soul | [Client](controller/README.md) |
| Build a training dataset | [Dataset preparation](log_dataset/README.md) |
| Train, export and evaluate | [Training](model/README.md) |

The repository contains the model, data tools, inference service and standalone
client. The hosted reviewer and multi-account bot operations are maintained
separately.

For source development, use Python 3.13, a C++ compiler, Rust 1.92, Node.js 22
and Chromium. Install the patched simulator before the Python requirements:

```sh
git clone https://github.com/smly/RiichiEnv.git /tmp/nejimakidori-riichienv
git -C /tmp/nejimakidori-riichienv checkout \
  479c1faeb33d082965eef8198f63261a79c0fce3
git -C /tmp/nejimakidori-riichienv apply "$PWD/docker/riichienv.patch"
python -m pip install /tmp/nejimakidori-riichienv
python -m pip install -r model/requirements.txt \
  -r controller/vision/requirements.txt mahjong==2.0.0 pytest
npm ci --prefix controller
python smoke.py small
```

Run `python smoke.py big` for the complete suite, including dataset building,
training/export, evaluation, HTTP and browser checks. Tests use CPU inference;
HTTP and browser checks require localhost access. The Docker build installs
the patched simulator automatically. Third-party Nyanten source and notices
are in [log_dataset/nyanten](log_dataset/nyanten/README.md).
