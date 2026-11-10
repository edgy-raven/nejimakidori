"""Warm value fitting with repeated simulator continuations."""

import argparse
import hashlib
import json
import pathlib

import numpy
import tensorflow as tf

from model import actor, export, joint_settlement
from training import self_play


def weight_hash(variables):
    digest = hashlib.sha256()
    for variable in variables:
        digest.update(variable.numpy().tobytes())
    return digest.hexdigest()


def value_head(
    network,
    width=None,
    depth=0,
    dropout=0.0,
    linear=False,
    penalty=0.0,
    extra_features=0,
):
    layers = [
        network.actor.get_layer(name)
        for name in ("value_hidden", "state_value")
    ]
    config = layers[0].get_config()
    config["units"] = layers[0].units if width is None else width
    assert config["units"] >= layers[0].units
    head = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(layers[0].kernel.shape[0],)),
            tf.keras.layers.Dense.from_config(config),
            tf.keras.layers.Dense.from_config(layers[1].get_config()),
        ]
    )
    weights = head.get_weights()
    weights[0][:, : layers[0].units] = layers[0].kernel.numpy()
    weights[1][: layers[0].units] = layers[0].bias.numpy()
    weights[2] = numpy.concatenate(
        [
            layers[1].kernel.numpy(),
            numpy.zeros((config["units"] - layers[0].units, 1), numpy.float32),
        ]
    )
    weights[3] = layers[1].bias.numpy()
    head.set_weights(weights)
    if depth or linear:
        head.trainable = False
        inputs = tf.keras.Input(
            shape=(layers[0].kernel.shape[0] + extra_features,)
        )
        residual = inputs
        for layer in range(depth):
            residual = tf.keras.layers.Dense(
                128, activation="gelu", name=f"residual_hidden_{layer}"
            )(residual)
            if dropout:
                residual = tf.keras.layers.Dropout(dropout)(residual)
        residual = tf.keras.layers.Dense(
            units=1,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer=tf.keras.regularizers.L2(penalty),
            dtype="float32",
            name="residual_value",
        )(residual)
        head = tf.keras.Model(
            inputs,
            tf.keras.layers.Add()(
                [head(inputs[:, : layers[0].kernel.shape[0]]), residual]
            ),
        )
    return head


def continuation(root, copies, seed):
    selected = None

    def capture(collector, batch):
        nonlocal selected
        if selected is None:
            assert len(batch["head"]) % copies == 0
            selected = (
                numpy.arange(copies) * (len(batch["head"]) // copies)
                + root.decision
            )

    batch, _ = yield from self_play.collection_steps(
        games=copies,
        seed=seed,
        retain_features=False,
        collector=root.fork(copies, seed),
        root_callback=capture,
    )
    return {
        name: batch[name][selected]
        for name in ("settlement", "chosen", "head", "legal")
    }


def collect(args, network):
    predict_policy = self_play.predictor(network.actor)

    def predict(features):
        return {
            **predict_policy(
                {name: features[name] for name in features.dtype.names}
            ),
            "_feature_rows": numpy.arange(len(features)),
        }

    for begin in range(args.start, args.start + args.roots, args.batch_roots):
        roots = {}
        seen = numpy.zeros(args.batch_roots, numpy.int64)
        generator = numpy.random.default_rng(args.seed + begin)

        def capture(collector, batch):
            for row in range(len(batch["head"])):
                game = int(batch["indices"][row, 0])
                seen[game] += 1
                if generator.integers(seen[game]) == 0:
                    roots[game] = (
                        collector.snapshot(row),
                        batch["features"][row].copy(),
                    )

        self_play.collect(
            predict=predict,
            games=args.batch_roots,
            seed=args.seed + begin,
            retain_features=False,
            root_callback=capture,
        )
        assert len(roots) == args.batch_roots
        features = numpy.asarray(
            [roots[game][1] for game in range(args.batch_roots)],
            dtype=self_play.FEATURE_DTYPE,
        )
        results = self_play.collect_streams(
            predict,
            [
                continuation(
                    roots[game][0],
                    args.copies,
                    args.seed + 1000000 + (begin + game) * args.copies,
                )
                for game in range(args.batch_roots)
            ],
        )
        settlements = numpy.stack([row["settlement"] for row in results])
        returns = (
            joint_settlement.observed_payoffs(
                tf.convert_to_tensor(
                    settlements.reshape(-1, settlements.shape[-1])
                )
            )
            .numpy()
            .reshape(args.batch_roots, args.copies)
            / 100
        )
        inputs = {name: features[name] for name in features.dtype.names}
        outputs = network.actor(inputs, training=False)
        board = numpy.asarray(outputs["board"], numpy.float32)
        numpy.testing.assert_allclose(
            value_head(network)(board).numpy()[:, 0],
            outputs["state_value"].numpy(),
            rtol=1e-6,
            atol=1e-7,
        )
        for result in results:
            numpy.testing.assert_array_equal(
                result["head"], numpy.repeat(result["head"][:1], args.copies)
            )
            numpy.testing.assert_array_equal(
                result["legal"],
                numpy.repeat(result["legal"][:1], args.copies, axis=0),
            )
        numpy.savez_compressed(
            file=args.output / f"roots-{begin:06d}.npz",
            features=features,
            board=board,
            returns=returns,
            root=numpy.arange(begin, begin + args.batch_roots),
            head=numpy.asarray([row["head"][0] for row in results]),
            legal=numpy.asarray([row["legal"][0] for row in results]),
            chosen=numpy.asarray([row["chosen"] for row in results]),
        )
        print(
            json.dumps(
                {"stage": "collection", "root_end": begin + args.batch_roots}
            ),
            flush=True,
        )


def metrics(predictions, returns):
    target = returns.mean(axis=1)
    residual = predictions - target
    return {
        "mean_return_rmse_st3": float(
            numpy.sqrt(numpy.mean(residual**2)) * 100
        ),
        "single_return_rmse_st3": float(
            numpy.sqrt(numpy.mean((predictions[:, None] - returns) ** 2)) * 100
        ),
        "bias_st3": float(numpy.mean(residual) * 100),
        "explained_variance": float(
            1 - numpy.var(residual) / numpy.var(target)
        ),
        "mean_target_sampling_variance_st3_squared": float(
            numpy.mean(numpy.var(returns, axis=1, ddof=1))
            / returns.shape[1]
            * 10000
        ),
        "calibration": [
            {
                "count": len(rows),
                "predicted_st3": float(predictions[rows].mean() * 100),
                "observed_st3": float(target[rows].mean() * 100),
            }
            for rows in numpy.array_split(numpy.argsort(predictions), 5)
        ],
    }


def value_inputs(data):
    if data.dtype.names:
        return {name: data[name] for name in data.dtype.names}
    return data


def value_predictions(head, data, batch_size=4096):
    return numpy.concatenate(
        [
            head(
                value_inputs(data[begin : begin + batch_size]), training=False
            ).numpy()[:, 0]
            for begin in range(0, len(data), batch_size)
        ]
    )


def fit_head(
    head,
    board,
    targets,
    validation,
    args,
    batch_size=64,
    prediction_batch_size=4096,
    clipnorm=None,
):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.value_learning_rate, clipnorm=clipnorm
    )
    best = head.get_weights()
    best_loss = float(
        numpy.mean(
            (
                value_predictions(
                    head, board[validation], prediction_batch_size
                )
                - targets[validation]
            )
            ** 2
        )
    )
    best_epoch = 0
    history = [{"epoch": 0, "validation_mse": best_loss}]
    generator = numpy.random.default_rng(args.seed)

    @tf.function
    def step(inputs, target):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                tf.square(head(inputs, training=True)[:, 0] - target)
            )
            if head.losses:
                loss += tf.add_n(head.losses)
        optimizer.apply_gradients(
            zip(
                tape.gradient(loss, head.trainable_variables),
                head.trainable_variables,
                strict=True,
            )
        )

    for epoch in range(1, args.epochs + 1):
        rows = generator.permutation(numpy.flatnonzero(~validation))
        for begin in range(0, len(rows), batch_size):
            selected = rows[begin : begin + batch_size]
            step(value_inputs(board[selected]), targets[selected])
        loss = float(
            numpy.mean(
                (
                    value_predictions(
                        head, board[validation], prediction_batch_size
                    )
                    - targets[validation]
                )
                ** 2
            )
        )
        history.append({"epoch": epoch, "validation_mse": loss})
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best = head.get_weights()
        if epoch - best_epoch >= args.patience:
            break
    head.set_weights(best)
    return {"selected_epoch": best_epoch, "history": history}


def policy_probe(network, data, baseline, args):
    excluded = {id(variable) for variable in network.encoder.weights}
    for name in ("value_hidden", "state_value"):
        excluded.update(
            id(variable) for variable in network.actor.get_layer(name).weights
        )
    variables = [
        variable
        for variable in network.actor.trainable_variables
        if id(variable) not in excluded
    ]
    frozen = [
        variable for variable in network.weights if id(variable) in excluded
    ]
    frozen_hash = weight_hash(frozen)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-5, weight_decay=1e-4, clipnorm=1.0
    )

    @tf.function
    def step(features, heads, legal, chosen, returns, values):
        with tf.GradientTape() as tape:
            outputs = network.actor(features, training=False)
            probability = actor.distribution(outputs, heads, legal)
            terms = actor.awr_policy_terms(
                -tf.gather(probability, chosen, batch_dims=1),
                (returns - values) * 100 / 3,
                tf.ones_like(chosen, dtype=tf.bool),
            )
            entropy = -tf.reduce_mean(
                tf.reduce_sum(tf.exp(probability) * probability, axis=-1)
            )
            loss = terms["loss"] - 0.01 * entropy
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(
            [
                (gradient, variable)
                for gradient, variable in zip(gradients, variables, strict=True)
                if gradient is not None
            ]
        )

    generator = numpy.random.default_rng(args.seed)
    for _ in range(args.policy_passes):
        for selected in numpy.array_split(
            generator.permutation(data["returns"].size),
            int(numpy.ceil(data["returns"].size / 256)),
        ):
            rows, copies = numpy.divmod(selected, data["returns"].shape[1])
            step(
                features={
                    name: data["features"][name][rows]
                    for name in data["features"].dtype.names
                },
                heads=data["head"][rows],
                legal=data["legal"][rows],
                chosen=data["chosen"][rows, copies],
                returns=data["returns"][rows, copies],
                values=baseline[rows],
            )
    assert weight_hash(frozen) == frozen_hash


def load_data(path):
    shards = [numpy.load(shard) for shard in sorted(path.glob("roots-*.npz"))]
    return {
        name: numpy.concatenate([shard[name] for shard in shards])
        for name in shards[0].files
    }


def fit(args, network):
    data = load_data(args.data)
    numpy.testing.assert_array_equal(data["root"], numpy.arange(args.roots))
    # Each root is drawn from a separate base game; no game crosses splits.
    split = numpy.repeat(
        [0, 1, 2], [args.roots * 3 // 4, args.roots // 8, args.roots // 8]
    )[
        numpy.argsort(
            numpy.random.default_rng(args.seed).permutation(args.roots)
        )
    ]
    numpy.save(args.output / "split.npy", split)
    original = network.get_weights()
    value_variables = {
        id(variable)
        for name in ("value_hidden", "state_value")
        for variable in network.actor.get_layer(name).weights
    }
    frozen = [
        variable
        for variable in network.weights
        if id(variable) not in value_variables
    ]
    frozen_hash = weight_hash(frozen)
    original_values = value_head(network)(data["board"]).numpy()[:, 0]
    summary = {
        "roots": args.roots,
        "copies": data["returns"].shape[1],
        "target": "hand ST3 projection; opponent hands fixed per root",
        "baseline": {
            name: metrics(
                original_values[split == index], data["returns"][split == index]
            )
            for index, name in enumerate(("train", "validation", "test"))
        },
        "arms": {},
    }
    for arm in ("baseline", "single", "averaged"):
        network.set_weights(original)
        head = value_head(network)
        numpy.testing.assert_array_equal(
            head(data["board"]).numpy()[:, 0], original_values
        )
        if arm != "baseline":
            targets = data["returns"].mean(axis=1)
            if arm == "single":
                targets[split == 0] = data["returns"][split == 0, 0]
            summary["arms"][arm] = fit_head(
                head=head,
                board=data["board"][split != 2],
                targets=targets[split != 2],
                validation=split[split != 2] == 1,
                args=args,
            )
        else:
            summary["arms"][arm] = {"selected_epoch": 0}
        for name in ("value_hidden", "state_value"):
            network.actor.get_layer(name).set_weights(
                head.get_layer(name).get_weights()
            )
        assert weight_hash(frozen) == frozen_hash
        values = head(data["board"]).numpy()[:, 0]
        summary["arms"][arm]["test"] = metrics(
            values[split == 2], data["returns"][split == 2]
        )
        network.save_weights(args.output / (arm + ".value.weights.h5"))
        policy_probe(
            network,
            {name: value[split == 0] for name, value in data.items()},
            values[split == 0],
            args,
        )
        network.save_weights(args.output / (arm + ".policy.weights.h5"))
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(
            json.dumps({"stage": "fitted", "arm": arm, **summary["arms"][arm]}),
            flush=True,
        )


def sweep(args, network):
    data = load_data(args.data)
    split = numpy.load(args.split)
    # The original test games do not participate in this tuning sweep.
    board = data["board"][split != 2]
    targets = data["returns"][split != 2].mean(axis=1)
    validation = split[split != 2] == 1
    head = value_head(network)
    best = head.get_weights()
    original_loss = float(
        numpy.mean(
            (head(board[validation]).numpy()[:, 0] - targets[validation]) ** 2
        )
    )
    summary = {
        "original_validation_mse": original_loss,
        "best_validation_mse": original_loss,
        "selected": "original",
        "trials": [],
    }
    frozen_hash = weight_hash(network.weights)
    for learning_rate in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4):
        for batch_size in (64, 256, 768):
            head = value_head(network)
            args.value_learning_rate = learning_rate
            result = fit_head(
                head=head,
                board=board,
                targets=targets,
                validation=validation,
                args=args,
                batch_size=batch_size,
            )
            loss = float(
                numpy.mean(
                    (
                        head(board[validation]).numpy()[:, 0]
                        - targets[validation]
                    )
                    ** 2
                )
            )
            trial = {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "validation_mse": loss,
                **result,
            }
            summary["trials"].append(trial)
            if loss < summary["best_validation_mse"]:
                summary["best_validation_mse"] = loss
                summary["selected"] = {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epoch": result["selected_epoch"],
                }
                best = head.get_weights()
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in trial.items()
                        if key != "history"
                    }
                ),
                flush=True,
            )
    assert weight_hash(network.weights) == frozen_hash
    head.set_weights(best)
    for name in ("value_hidden", "state_value"):
        network.actor.get_layer(name).set_weights(
            head.get_layer(name).get_weights()
        )
    network.save_weights(args.output / "best.value.weights.h5")


def confirm(args, network):
    data = load_data(args.data)
    original = value_head(network)(data["board"]).numpy()[:, 0]
    excluded = {
        id(variable)
        for name in ("value_hidden", "state_value")
        for variable in network.actor.get_layer(name).weights
    }
    frozen = [
        variable for variable in network.weights if id(variable) not in excluded
    ]
    frozen_hash = weight_hash(frozen)
    network.load_weights(args.compare)
    assert weight_hash(frozen) == frozen_hash
    candidate = value_head(network)(data["board"]).numpy()[:, 0]
    target = data["returns"].mean(axis=1)
    differences = ((candidate - target) ** 2 - (original - target) ** 2) * 10000
    draws = numpy.random.default_rng(args.seed).integers(
        len(target), size=(10000, len(target))
    )
    interval = numpy.quantile(
        differences[draws].mean(axis=1), [args.alpha / 2, 1 - args.alpha / 2]
    )
    summary = {
        "roots": len(target),
        "copies": data["returns"].shape[1],
        "alpha": args.alpha,
        "original": metrics(original, data["returns"]),
        "candidate": metrics(candidate, data["returns"]),
        "mse_difference_st3_squared": float(differences.mean()),
        "paired_interval": interval.tolist(),
        "confirmed_improvement": bool(interval[1] < 0),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def value_board(data, scoring_features=False):
    if scoring_features:
        return numpy.concatenate(
            [
                data["board"],
                data["features"]["public_fu"].astype(numpy.float32) / 128,
                data["features"]["public_bonus"].astype(numpy.float32) / 13,
            ],
            axis=1,
        )
    return data["board"]


def architecture_search(args, network):
    data = load_data(args.data)
    split = numpy.load(args.split)
    board = data["board"][split != 2]
    targets = data["returns"][split != 2].mean(axis=1)
    validation = split[split != 2] == 1
    original = value_head(network)(board).numpy()[:, 0]
    original_loss = float(
        numpy.mean((original[validation] - targets[validation]) ** 2)
    )
    summary = {
        "original_validation_mse": original_loss,
        "best_validation_mse": original_loss,
        "selected": None,
        "trials": [],
        "original_training_mse": float(
            numpy.mean((original[~validation] - targets[~validation]) ** 2)
        ),
    }
    frozen_hash = weight_hash(network.weights)
    architectures = [
        {"width": width, "depth": 0, "dropout": 0.0}
        for width in (128, 256, 512)
    ] + [
        {"width": 128, "depth": depth, "dropout": dropout}
        for depth in (1, 2, 3)
        for dropout in (0.0, 0.1)
    ]
    if args.stage == "regularized":
        architectures = [
            {
                "width": 128,
                "depth": 0,
                "dropout": 0.0,
                "linear": True,
                "penalty": penalty,
            }
            for penalty in (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0)
        ]
    if args.scoring_features:
        assert args.stage == "regularized"
        board = value_board(data, scoring_features=True)[split != 2]
        for architecture in architectures:
            architecture["extra_features"] = 8
    for architecture in architectures:
        for learning_rate in (
            args.residual_learning_rates
            if architecture["depth"] or args.stage == "regularized"
            else (1e-6, 1e-5, 1e-4)
        ):
            for batch_size in (64, 256):
                trial = {
                    **architecture,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "seeds": [],
                }
                heads = []
                for seed in (941, 942, 943):
                    tf.keras.utils.set_random_seed(seed)
                    head = value_head(network, **architecture)
                    numpy.testing.assert_allclose(
                        head(board).numpy()[:, 0],
                        original,
                        rtol=1e-5,
                        atol=1e-6,
                    )
                    args.seed = seed
                    args.value_learning_rate = learning_rate
                    result = fit_head(
                        head=head,
                        board=board,
                        targets=targets,
                        validation=validation,
                        args=args,
                        batch_size=batch_size,
                    )
                    trial["seeds"].append(
                        {
                            "seed": seed,
                            **result,
                            "training_mse": float(
                                numpy.mean(
                                    (
                                        head(
                                            board[~validation], training=False
                                        ).numpy()[:, 0]
                                        - targets[~validation]
                                    )
                                    ** 2
                                )
                            ),
                            "validation_mse": min(
                                row["validation_mse"]
                                for row in result["history"]
                            ),
                        }
                    )
                    heads.append(head)
                trial["validation_mse"] = float(
                    numpy.mean(
                        [row["validation_mse"] for row in trial["seeds"]]
                    )
                )
                summary["trials"].append(trial)
                if trial["validation_mse"] < summary["best_validation_mse"]:
                    summary["best_validation_mse"] = trial["validation_mse"]
                    summary["selected"] = {
                        **architecture,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                    }
                    for seed, head in zip((941, 942, 943), heads, strict=True):
                        head.save(args.output / f"seed-{seed}.keras")
                (args.output / "summary.json").write_text(
                    json.dumps(summary, indent=2)
                )
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in trial.items()
                            if key != "seeds"
                        }
                    ),
                    flush=True,
                )
    assert weight_hash(network.weights) == frozen_hash


def learning_curve(args, network):
    """Nested development subsets; confirmation panels never enter this fit."""
    data = load_data(args.data)
    split = numpy.load(args.split)
    validation_rows = numpy.flatnonzero(split == 1)
    training_rows = numpy.flatnonzero(split == 0)
    assert max(args.training_roots) <= len(training_rows)
    original = value_predictions(value_head(network), data["board"])
    frozen_hash = weight_hash(network.weights)
    summary = {
        "validation_roots": len(validation_rows),
        "original": metrics(
            original[validation_rows], data["returns"][validation_rows]
        ),
        "trials": [],
    }
    for seed in (941, 942, 943):
        ordered = numpy.random.default_rng(seed).permutation(training_rows)
        for count in args.training_roots:
            rows = numpy.concatenate([ordered[:count], validation_rows])
            validation = numpy.arange(len(rows)) >= count
            for depth, dropout, rate in ((0, 0.0, 1e-6), (2, 0.1, 1e-5)):
                tf.keras.utils.set_random_seed(seed)
                head = value_head(network, depth=depth, dropout=dropout)
                numpy.testing.assert_allclose(
                    value_predictions(head, data["board"][rows]),
                    original[rows],
                    rtol=1e-5,
                    atol=1e-6,
                )
                args.seed = seed
                args.value_learning_rate = rate
                result = fit_head(
                    head=head,
                    board=data["board"][rows],
                    targets=data["returns"][rows].mean(axis=1),
                    validation=validation,
                    args=args,
                    batch_size=64,
                )
                predictions = value_predictions(head, data["board"][rows])
                trial = {
                    "seed": seed,
                    "training_roots": count,
                    "depth": depth,
                    "dropout": dropout,
                    "learning_rate": rate,
                    "training_rows": ordered[:count].tolist(),
                    "validation_rows": validation_rows.tolist(),
                    **result,
                    "original_training": metrics(
                        original[ordered[:count]],
                        data["returns"][ordered[:count]],
                    ),
                    "training": metrics(
                        predictions[:count], data["returns"][ordered[:count]]
                    ),
                    "validation": metrics(
                        predictions[count:], data["returns"][validation_rows]
                    ),
                }
                summary["trials"].append(trial)
                (args.output / "summary.json").write_text(
                    json.dumps(summary, indent=2)
                )
                print(
                    json.dumps(
                        {
                            name: trial[name]
                            for name in (
                                "seed",
                                "training_roots",
                                "depth",
                                "selected_epoch",
                                "training",
                                "validation",
                            )
                        }
                    ),
                    flush=True,
                )
    assert weight_hash(network.weights) == frozen_hash


def encoder_search(args, network):
    data = load_data(args.data)
    split = numpy.load(args.split)
    features = data["features"][split != 2]
    targets = data["returns"][split != 2].mean(axis=1)
    validation = split[split != 2] == 1
    value_model = tf.keras.Model(
        network.actor.input, network.actor.output["state_value"][:, None]
    )
    selected_variables = {id(variable) for variable in value_model.weights}
    frozen = [
        variable
        for variable in network.weights
        if id(variable) not in selected_variables
    ]
    frozen_hash = weight_hash(frozen)
    original_weights = network.get_weights()
    original = value_predictions(value_model, features[validation], 64)
    original_loss = float(numpy.mean((original - targets[validation]) ** 2))
    summary = {
        "original_validation_mse": original_loss,
        "best_validation_mse": original_loss,
        "selected": None,
        "trials": [],
        "target": "V only; separate encoder copy",
    }
    for learning_rate in (1e-6, 3e-6, 1e-5):
        trial = {"learning_rate": learning_rate, "seeds": []}
        weights = []
        for seed in (941, 942, 943):
            network.set_weights(original_weights)
            tf.keras.utils.set_random_seed(seed)
            args.seed = seed
            args.value_learning_rate = learning_rate
            numpy.testing.assert_allclose(
                value_predictions(value_model, features[validation], 64),
                original,
                rtol=1e-6,
                atol=1e-7,
            )
            result = fit_head(
                head=value_model,
                board=features,
                targets=targets,
                validation=validation,
                args=args,
                batch_size=64,
                prediction_batch_size=64,
                clipnorm=1.0,
            )
            assert weight_hash(frozen) == frozen_hash
            trial["seeds"].append(
                {
                    "seed": seed,
                    **result,
                    "validation_mse": min(
                        row["validation_mse"] for row in result["history"]
                    ),
                }
            )
            weights.append(value_model.get_weights())
        trial["validation_mse"] = float(
            numpy.mean([row["validation_mse"] for row in trial["seeds"]])
        )
        summary["trials"].append(trial)
        if trial["validation_mse"] < summary["best_validation_mse"]:
            summary["best_validation_mse"] = trial["validation_mse"]
            summary["selected"] = {"learning_rate": learning_rate}
            for seed, values in zip((941, 942, 943), weights, strict=True):
                network.set_weights(original_weights)
                value_model.set_weights(values)
                assert weight_hash(frozen) == frozen_hash
                network.save_weights(
                    args.output / f"seed-{seed}.value.weights.h5"
                )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(trial), flush=True)
    network.set_weights(original_weights)


def confirm_predictions(original, candidates, returns, args):
    target = returns.mean(axis=1)
    original_errors = (original - target) ** 2
    differences = numpy.stack(
        [
            (candidate - target) ** 2 - original_errors
            for candidate in candidates.values()
        ]
    )
    generator = numpy.random.default_rng(args.seed)
    repetitions = max(10000, int(numpy.ceil(200 / args.alpha)))
    # Keep at least 100 draws in each confidence tail without a huge index
    # allocation. Each bootstrap resamples whole independent game roots.
    sampled = numpy.concatenate(
        [
            differences[
                :,
                generator.integers(
                    len(target),
                    size=(min(2000, repetitions - begin), len(target)),
                ),
            ].mean(axis=-1)
            * 10000
            for begin in range(0, repetitions, 2000)
        ],
        axis=1,
    )
    summary = {
        "original": metrics(original, returns),
        "alpha_per_seed": args.alpha,
        "bootstrap_repetitions": repetitions,
        "seeds": [],
    }
    for row, (seed, candidate) in enumerate(candidates.items()):
        interval = numpy.quantile(
            sampled[row],
            [args.alpha / 2, 1 - args.alpha / 2],
        )
        relative_gain = float(-differences[row].mean() / original_errors.mean())
        summary["seeds"].append(
            {
                "seed": seed,
                "candidate": metrics(candidate, returns),
                "relative_mse_gain": relative_gain,
                "paired_interval_st3_squared": interval.tolist(),
                "passed": bool(interval[1] < 0 and relative_gain >= 0.02),
            }
        )
    summary["sustained_panel_improvement"] = all(
        row["passed"] for row in summary["seeds"]
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def confirm_heads(args, network):
    data = load_data(args.data)
    original = value_head(network)(data["board"]).numpy()[:, 0]
    candidates = {}
    for seed in (941, 942, 943):
        head = tf.keras.models.load_model(args.compare / f"seed-{seed}.keras")
        candidates[seed] = head(
            value_board(data, args.scoring_features)
        ).numpy()[:, 0]
    confirm_predictions(original, candidates, data["returns"], args)


def confirm_encoder(args, network):
    data = load_data(args.data)
    value_model = tf.keras.Model(
        network.actor.input, network.actor.output["state_value"][:, None]
    )
    original = value_predictions(value_model, data["features"], 64)
    selected = {id(variable) for variable in value_model.weights}
    frozen = [
        variable for variable in network.weights if id(variable) not in selected
    ]
    frozen_hash = weight_hash(frozen)
    candidates = {}
    for seed in (941, 942, 943):
        network.load_weights(args.compare / f"seed-{seed}.value.weights.h5")
        assert weight_hash(frozen) == frozen_hash
        candidates[seed] = value_predictions(value_model, data["features"], 64)
    confirm_predictions(original, candidates, data["returns"], args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "collect",
            "fit",
            "sweep",
            "confirm",
            "architecture",
            "learning-curve",
            "encoder",
            "regularized",
            "confirm-heads",
            "confirm-encoder",
        ),
    )
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--data", type=pathlib.Path)
    parser.add_argument("--split", type=pathlib.Path)
    parser.add_argument("--compare", type=pathlib.Path)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--roots", type=int, default=1024)
    parser.add_argument("--batch-roots", type=int, default=16)
    parser.add_argument("--copies", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2040010000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--value-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--residual-learning-rates",
        type=float,
        nargs="+",
        default=[1e-6, 1e-5, 1e-4],
    )
    parser.add_argument(
        "--training-roots",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 12288],
    )
    parser.add_argument("--scoring-features", action="store_true")
    parser.add_argument("--policy-passes", type=int, default=8)
    args = parser.parse_args()
    assert args.roots % args.batch_roots == 0
    assert args.copies >= 2
    args.output.mkdir(parents=True, exist_ok=True)
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(args.seed)
    network = export.load_network(
        args.weights, self_play.initial_features(args.seed)
    )
    if args.stage == "collect":
        collect(args, network)
    elif args.stage == "fit":
        fit(args, network)
    elif args.stage == "sweep":
        sweep(args, network)
    elif args.stage == "confirm":
        confirm(args, network)
    elif args.stage in ("architecture", "regularized"):
        architecture_search(args, network)
    elif args.stage == "learning-curve":
        learning_curve(args, network)
    elif args.stage == "encoder":
        encoder_search(args, network)
    elif args.stage == "confirm-heads":
        confirm_heads(args, network)
    else:
        confirm_encoder(args, network)


if __name__ == "__main__":
    main()
