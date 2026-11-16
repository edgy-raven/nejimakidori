"""Stable candidate metrics shared by TensorBoard and the live dashboard."""

import json
import pathlib
import time

import tensorboard.plugins.custom_scalar.layout_pb2 as scalar_layout
import tensorboard.plugins.custom_scalar.summary as scalar_summary
import tensorflow as tf


def write_json(path, value):
    """Replace run metadata atomically so readers never see partial JSON."""
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Monitor:
    def __init__(self, root, positions, plan):
        self.root = pathlib.Path(root)
        self.positions = positions
        self.phases = ["record-imitation"] + [
            f"mixed-policy-{index}" for index in range(1, 6)
        ]
        self.started = self.previous = time.monotonic()
        self.previous_exposures = 0
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
            "Policy health": {
                "Actor objectives": (
                    r"policy/(policy|expert_loss|self_play_loss)$"
                ),
                "Label Q gaps (ST3)": r"policy/(expert|distill)_q_gap_st3$",
                "Self-play gain (ST3)": r"policy/self_play_gain_st3$",
                "Settlement NLL": r"policy/payment_nll$",
                "Belief-Q direction": r"policy/(tenpai|wait)_direction_loss$",
                "Entropy": r"policy/entropy$",
                "Recorded targets": r"policy/(expert|distill)_weighted_ce$",
                "Recorded ESS": (
                    r"policy/(expert|distill)_effective_sample_fraction$"
                ),
                "Adaptive beta": r"policy/(expert|distill)_beta$",
                "Rolling ESS": r"policy/(expert|distill)_rolling_ess$",
                "Minimum attainable ESS": (
                    r"policy/(expert|distill)_minimum_rolling_ess$"
                ),
                "Clipped weights": (
                    r"policy/(expert|distill)_weight_clipped_fraction$"
                ),
                "Learning rate": r"policy/learning_rate$",
                "Policy anchors": r"policy/(mirror_kl|magnet_kl)$",
                "Weighted anchors": r"policy/weighted_(mirror|magnet)_loss$",
                "Risk penalty (ST3)": r"policy/risk_penalty_st3$",
                "Local gradient norm": r"gradient/local_norm$",
            },
            "Replay": {
                "Source shares": r"replay/(record|generated)_fraction$",
                "Targeted shares within source": (
                    r"replay/(record|generated)_target_fraction$"
                ),
                "Pool occupancy": r"replay/.*_pool_positions$",
            },
            "Match critic fitting": {
                "Settlement likelihood": r"q/payment_nll$",
                "Constraints": r"q/critic_constraint_loss$",
                "Settlement support": r"q/settlement_supported_fraction$",
                "Local gradient norm": r"q/gradient/local_norm$",
                "Learning rate": r"q/learning_rate$",
            },
            "Self play": {
                "Collection positions/s": r"collection/positions_per_second$",
                "Roots and branches": (
                    r"collection/(selected_roots|root_continuations)$"
                ),
                "Games": r"progress/games$",
            },
            "Belief health": {
                "Supervision": r"policy/(belief_loss|opponent_.*_loss)$",
                "Wait calibration": r"policy/(joint_wait|tenpai_wait)_brier$",
                "Wait recall": r"policy/tenpai_wait_recall_at_5$",
                "Count constraints": (
                    r"policy/count_(size_error|copy_overflow)$"
                ),
                "Constraint loss": r"policy/belief_constraint_loss$",
            },
            "Runtime": {
                "Step breakdown (seconds)": (
                    r"runtime/(input|root|update)_seconds$"
                ),
                "Training positions/s": r"runtime/positions_per_second$",
                "End-to-end positions/s": (
                    r"runtime/overall_positions_per_second$"
                ),
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
        # Collection/evaluation pauses are not counted as slow optimizer work.
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
        write_json(self.root / "status.json", status)
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
                if name.startswith("q/"):
                    tf.summary.scalar(
                        name, value, step=int(metrics["q_total_exposures"])
                    )
                    continue
                if name.startswith("q_"):
                    continue
                if "/" in name:
                    tag = name
                elif stage.endswith("_cohort") or stage.endswith("_collecting"):
                    tag = "collection/" + name
                else:
                    tag = "policy/" + name
                tf.summary.scalar(tag, value)
        self.writer.flush()
        self.previous, self.previous_exposures = now, exposures

    def close(self):
        self.writer.close()
