"""Fit the current-hand critic twice on the completed checkpoint match."""

import json
import os
import subprocess
import sys
import time

import numpy
import tensorflow as tf
from training import match_records
from training import trainer as trainers
from training.candidate import replay


def run(*, stage, root, network, strategy, args, report, total_exposures):
    report(stage + "_q-building", {})
    data = root / (stage + "-q-data")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.match_records",
            "--matches",
            str(root / (stage + "-evaluation") / "matches"),
            "--output",
            str(data),
            "--workers",
            str(args.collection_workers),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )
    return fit(
        stage=stage,
        root=root,
        data=data,
        network=network,
        strategy=strategy,
        args=args,
        report=report,
        total_exposures=total_exposures,
    )


def fit(*, stage, root, data, network, strategy, args, report, total_exposures):
    positions = json.loads((data / "dataset.json").read_text())["positions"]
    with strategy.scope():
        trainer = trainers.DistributedCriticTrainer(
            trainers.CriticTrainer(
                network, network.encoder.input, args.learning_rate
            ),
            strategy,
        )
    exposures = 0
    for pass_number in (1, 2):
        report(
            stage + "_q-fit",
            {
                "q_pass": pass_number,
                "q_passes": 2,
                "q_exposures": exposures,
                "q_budget": 2 * positions,
                "q_total_exposures": total_exposures,
                "q_positions_per_second": 0,
            },
        )
        seen = 0
        started = time.monotonic()
        for tensors in match_records.batches(
            data, args.batch_size, args.seed + pass_number
        ):
            tensors = tf.nest.map_structure(
                lambda value: value.numpy(), tensors
            )
            batch = {
                "features": trainer.feature_packing.pack_numpy(
                    tensors["features"]
                ),
                "chosen": numpy.asarray(tensors["chosen"], dtype=numpy.int32),
                "payment_points": numpy.asarray(
                    tensors["payment_points"], dtype=numpy.int32
                ),
            }
            count = len(batch["chosen"])
            metrics = trainer.step(
                replay.pad_batch(batch, args.batch_size),
                tf.constant(count, tf.int32),
            )
            exposures += count
            total_exposures += count
            seen += count
            now = time.monotonic()
            report(
                stage + "_q-fit",
                {
                    "q_pass": pass_number,
                    "q_passes": 2,
                    "q_exposures": exposures,
                    "q_budget": 2 * positions,
                    "q_total_exposures": total_exposures,
                    "q_positions_per_second": count / (now - started),
                    **{
                        "q/" + name: float(value)
                        for name, value in metrics.items()
                    },
                },
            )
            started = time.monotonic()
        if seen != positions:
            raise ValueError(f"Q dataset count mismatch: {seen} != {positions}")
        checkpoint = root / f"{stage}-q-{pass_number}.weights.h5"
        network.save_weights(checkpoint)
        pending = root / "latest.weights.h5.pending"
        pending.symlink_to(checkpoint.name)
        pending.replace(root / "latest.weights.h5")
    return total_exposures
