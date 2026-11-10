"""Stable candidate metrics shared by TensorBoard and the live dashboard."""

import json
import pathlib
import time

import tensorboard.plugins.custom_scalar.layout_pb2 as scalar_layout
import tensorboard.plugins.custom_scalar.summary as scalar_summary
import tensorflow as tf

from training import online_training


class Monitor:
    def __init__(self, root, positions, plan, resume=None):
        self.root = pathlib.Path(root)
        self.positions = positions
        self.phases = (
            []
            if plan["mixed_only"]
            else ["distill", "record-awr"][int(plan["skip_distill"]) :]
        ) + [f"mixed-awr-{index}" for index in range(1, 6)]
        self.started = self.previous = time.monotonic()
        self.previous_exposures = 0
        if resume is not None:
            self.started -= resume["elapsed_seconds"]
            self.previous_exposures = resume["exposures"]
        self.rate = 0.0
        self.writer = tf.summary.create_file_writer(
            str(self.root / "tensorboard")
        )
        layout = scalar_layout.Layout()
        for category, charts in {
            "Progress": {
                "Pass and total fractions": r"progress/(phase|total)_fraction$",
                "Training positions": r"progress/total_exposures$",
            },
            "Distillation": {"Production KL": r"distill/teacher_kl$"},
            "AWR health": {
                "Policy loss": r"awr/policy$",
                "Value RMSE (ST3)": r"awr/value_rmse_st3$",
                "Value bias (ST3)": r"awr/value_bias_st3$",
                "Settlement NLL": r"awr/payment_nll$",
                "Entropy": r"awr/entropy$",
                "Weight health": r"awr/(effective_sample_fraction|weight_clipped_fraction)$",
                "Learning rate": r"(awr|distill)/learning_rate$",
                "Previous-policy pull": r"awr/(mirror_kl|weighted_mirror_loss)$",
                "Local gradient norm": r"gradient/local_norm$",
            },
            "Replay": {
                "Source shares": r"replay/(record|generated)_fraction$",
                "Targeted shares within source": r"replay/(record|generated)_target_fraction$",
                "Pool occupancy": r"replay/.*_pool_positions$",
            },
            "Self play": {
                "Collection positions/s": r"collection/positions_per_second$",
                "Roots and branches": r"collection/(selected_roots|root_continuations)$",
                "Games": r"progress/games$",
            },
            "Runtime": {
                "Training positions/s": r"runtime/positions_per_second$",
                "End-to-end positions/s": r"runtime/overall_positions_per_second$",
                "Elapsed hours": r"runtime/elapsed_hours$",
            },
        }.items():
            group = layout.category.add(title=category)
            for title, pattern in charts.items():
                group.chart.add(title=title).multiline.tag.append(pattern)
        with self.writer.as_default():
            tf.summary.text("run/plan", json.dumps(plan, indent=2), step=0)
            tf.summary.experimental.write_raw_pb(
                scalar_summary.pb(layout).SerializeToString(), step=0
            )
        self.writer.flush()

    def report(self, *, stage, metrics, exposures, cohort, games):
        now = time.monotonic()
        phase = next(
            (
                name
                for name in self.phases
                if stage == name or stage.startswith(name + "_")
            ),
            None,
        )
        phase_index = (
            self.phases.index(phase) + 1 if phase else len(self.phases)
        )
        phase_exposures = max(
            0,
            min(self.positions, exposures - (phase_index - 1) * self.positions),
        )
        # Collection/evaluation pauses are not mislabeled as slow optimizer work.
        if exposures > self.previous_exposures:
            self.rate = (exposures - self.previous_exposures) / (
                now - self.previous
            )
        status = {
            "stage": stage,
            "phase_name": phase or "complete",
            "phase_index": phase_index,
            "phase_exposures": phase_exposures,
            "phase_budget": self.positions,
            "phase_fraction": phase_exposures / self.positions,
            "exposures": exposures,
            "total_exposures": exposures,
            "budget_exposures": len(self.phases) * self.positions,
            "total_fraction": exposures / (len(self.phases) * self.positions),
            "cohort": cohort,
            "games": games,
            "elapsed_seconds": now - self.started,
            "window_positions_per_second": self.rate,
            "updated_at": time.time(),
            **{name: float(value) for name, value in metrics.items()},
        }
        online_training.save(self.root / "status.json", status)
        with (self.root / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(status) + "\n")
        with self.writer.as_default(step=exposures):
            tf.summary.text("run/stage", stage)
            for name in (
                "total_exposures",
                "total_fraction",
                "phase_fraction",
                "phase_index",
                "games",
            ):
                tf.summary.scalar("progress/" + name, status[name])
            tf.summary.scalar(
                "runtime/elapsed_hours", status["elapsed_seconds"] / 3600
            )
            tf.summary.scalar(
                "runtime/overall_positions_per_second",
                exposures / status["elapsed_seconds"],
            )
            if exposures > self.previous_exposures:
                tf.summary.scalar("runtime/positions_per_second", self.rate)
            for name, value in metrics.items():
                group = "distill" if phase == "distill" else "awr"
                if "/" in name:
                    tag = name
                elif stage.endswith("_cohort") or stage.endswith("_collecting"):
                    tag = "collection/" + name
                else:
                    tag = group + "/" + name
                tf.summary.scalar(tag, value)
            if "value_loss" in metrics:
                tf.summary.scalar(
                    "awr/value_rmse_st3", tf.sqrt(metrics["value_loss"]) * 100
                )
                tf.summary.scalar(
                    "awr/value_bias_st3", metrics["value_bias"] * 100
                )
        self.writer.flush()
        self.previous, self.previous_exposures = now, exposures

    def close(self):
        self.writer.close()
