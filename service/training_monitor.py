"""Read complete progress records and resume TensorBoard without duplicates."""

import csv
import io
import json
import pathlib
import re

import tensorboard.compat.proto.event_pb2
import tensorboard.compat.proto.summary_pb2
import tensorboard.plugins.custom_scalar.layout_pb2
import tensorboard.plugins.custom_scalar.summary
import tensorboard.plugins.text.summary
import tensorboard.summary.writer.event_file_writer

CHARTS = {
    "Training": {
        "Loss": ["total_loss", "policy_loss"],
        "Discard agreement": ["discard_match", "discard_top3"],
        "Policy ramp": [
            "training_parameters/offline_payoff_weight",
            "training_parameters/offline_policy_mix",
        ],
        "Value error": ["payoff_critic_loss", "payoff_baseline_loss"],
    },
    "Attack and defense": {
        "Policy imitation": [
            "expert_attack_imitation_loss",
            "expert_defense_imitation_loss",
        ],
    },
    "Board representation": {
        "Placement": ["round_placement_loss", "final_placement_loss"],
        "Payments": ["payment_cross_entropy", "payment_occurrence_loss"],
        "Opponent hand": ["opponent_shanten_loss", "opponent_hand_count_loss"],
        "Opponent ukeire": [
            "opponent_ukeire_loss",
            "tenpai_ukeire_loss",
            "other_ukeire_loss",
        ],
        "Completion yaku": ["completion_yaku_loss"],
        "Tenpai wait recall at 5": ["tenpai_wait_recall_at_5"],
    },
    "Runtime": {
        "Learning rate": ["training_parameters/learning_rate"],
        "Gradient norm": [
            "training_parameters/replica_gradient_norm",
            "training_parameters/replica_actor_gradient_norm",
            "training_parameters/replica_value_gradient_norm",
        ],
        "Positions per second": ["training_parameters/examples_per_second"],
    },
}


class Monitor:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.writers = {}
        self.recorded = set()
        status = self.root / "monitor_status.json"
        if status.exists():
            self.recorded = set(
                tuple(item)
                for item in json.loads(status.read_text())["recorded"]
            )

        self.write_layout()

    def write_layout(self):
        layout = tensorboard.plugins.custom_scalar.layout_pb2.Layout()
        if (self.root / "games/summary.json").exists():
            category = layout.category.add(title="Games", closed=False)
            for metric in ("first", "fourth", "rank_points"):
                chart = category.chart.add(
                    title={
                        "first": "First-place rate — higher is better",
                        "fourth": "Last-place rate — lower is better",
                        "rank_points": "Rank points per game — higher is better",
                    }[metric]
                )
                chart.margin.series.add(
                    value=f"{metric}/candidate",
                    lower=f"{metric}/lower",
                    upper=f"{metric}/upper",
                )
                chart.margin.series.add(
                    value=f"{metric}/reference",
                    lower=f"{metric}/reference",
                    upper=f"{metric}/reference",
                )
        for title, charts in CHARTS.items():
            category = layout.category.add(title=title, closed=False)
            for name, tags in charts.items():
                chart = category.chart.add(title=name)
                chart.multiline.tag.extend(f"^{tag}$" for tag in tags)
        writer = tensorboard.summary.writer.event_file_writer.EventFileWriter(
            str(self.root / "tensorboard/layout")
        )
        writer.add_event(
            tensorboard.compat.proto.event_pb2.Event(
                summary=tensorboard.compat.proto.summary_pb2.Summary.FromString(
                    tensorboard.plugins.custom_scalar.summary.pb(
                        layout
                    ).SerializeToString()
                )
            )
        )
        writer.close()

    def poll(self, now):
        config_path = self.root / "candidate/run_config.json"
        if not config_path.exists():
            status = {
                **json.loads((self.root / "pipeline_status.json").read_text()),
                "recorded": sorted(self.recorded),
                "updated_at": now,
            }
            record_log = self.root / "records.log"
            if status["stage"] == "records" and record_log.exists():
                progress = re.findall(
                    r"records shards=(\d+)/(\d+) examples=(\d+) "
                    r"elapsed_seconds=([\d.]+)\n",
                    record_log.read_text(),
                )
                if progress:
                    completed, total, examples, elapsed = progress[-1]
                    status["records"] = {
                        "completed_shards": int(completed),
                        "total_shards": int(total),
                        "examples": int(examples),
                    }
                    status["progress_percent"] = (
                        100 * int(completed) / int(total)
                    )
                    status["eta_seconds"] = (
                        float(elapsed)
                        * (int(total) - int(completed))
                        / int(completed)
                    )
            return self.publish_status(status)
        config = json.loads(config_path.read_text())
        if ("configuration", 0) not in self.recorded:
            descriptions = {"training_parameters": config}
            writer = (
                tensorboard.summary.writer.event_file_writer.EventFileWriter(
                    str(self.root / "tensorboard/configuration")
                )
            )
            for name, fields in descriptions.items():
                writer.add_event(
                    tensorboard.compat.proto.event_pb2.Event(
                        wall_time=now,
                        step=0,
                        summary=tensorboard.compat.proto.summary_pb2.Summary.FromString(
                            tensorboard.plugins.text.summary.pb(
                                name,
                                "```json\n"
                                + json.dumps(fields, indent=2)
                                + "\n```",
                            ).SerializeToString()
                        ),
                    )
                )
            writer.close()
            self.recorded.add(("configuration", 0))
        budget = config["phase_budget"]
        total = sum(row["epochs"] * row["steps_per_epoch"] for row in budget)
        raw = (self.root / "training.log").read_text()
        phase = None
        epoch = 0
        step = 0
        throughput = 0
        metrics = {}
        for line in raw[: raw.rfind("\n") + 1].splitlines():
            if line.startswith("phase="):
                phase = line.split()[0].split("=")[1]
            elif line.startswith("Epoch "):
                epoch = int(line.split()[1].split("/")[0]) - 1
            elif line.startswith("step="):
                values = dict(re.findall(r"(\w+)=([-+\d.e]+)", line))
                local_step = int(values["step"])
                position = next(
                    i for i, row in enumerate(budget) if row["phase"] == phase
                )
                step = (
                    sum(
                        row["epochs"] * row["steps_per_epoch"]
                        for row in budget[:position]
                    )
                    + epoch * budget[position]["steps_per_epoch"]
                    + local_step
                )
                throughput = float(values["examples_per_second"])
                key = (phase, step)
                if key not in self.recorded:
                    if phase not in self.writers:
                        self.writers[phase] = (
                            tensorboard.summary.writer.event_file_writer.EventFileWriter(
                                str(self.root / "tensorboard" / phase / "live")
                            )
                        )
                    writer = self.writers[phase]
                    scalars = {}
                    for charts in CHARTS.values():
                        for tags in charts.values():
                            for tag in tags:
                                name = (
                                    "loss"
                                    if tag == "total_loss"
                                    else tag.removeprefix(
                                        "training_parameters/"
                                    )
                                )
                                if name in values:
                                    scalars[tag] = float(values[name])
                    writer.add_event(
                        tensorboard.compat.proto.event_pb2.Event(
                            wall_time=now,
                            step=step,
                            summary=tensorboard.compat.proto.summary_pb2.Summary(
                                value=[
                                    tensorboard.compat.proto.summary_pb2.Summary.Value(
                                        tag=name,
                                        simple_value=value,
                                    )
                                    for name, value in scalars.items()
                                ]
                            ),
                        )
                    )
                    writer.flush()
                    self.recorded.add(key)
        if phase:
            history = self.root / "candidate" / (phase + ".csv")
            if history.exists():
                raw = history.read_text()
                rows = list(
                    csv.DictReader(io.StringIO(raw[: raw.rfind("\n") + 1]))
                )
                if rows:
                    metrics = {
                        name: float(value)
                        for name, value in rows[-1].items()
                        if name != "epoch" and value not in ("NA", "", None)
                    }
        plan = json.loads((self.root / "plan.json").read_text())
        comparison = self.root / "games/summary.json"
        if comparison.exists():
            summary = json.loads(comparison.read_text())
            key = ("comparison", summary["seed_groups"])
            if key not in self.recorded:
                self.write_layout()
                writer = tensorboard.summary.writer.event_file_writer.EventFileWriter(
                    str(self.root / "tensorboard/games")
                )
                for metric in ("first", "fourth", "rank_points"):
                    baseline = summary["models"]["old"][metric]
                    low, high = summary["paired_bounds"][metric]
                    values = {
                        "reference": baseline,
                        "candidate": summary["models"]["new"][metric],
                        "lower": baseline + low,
                        "upper": baseline + high,
                    }
                    writer.add_event(
                        tensorboard.compat.proto.event_pb2.Event(
                            wall_time=now,
                            step=summary["seed_groups"],
                            summary=tensorboard.compat.proto.summary_pb2.Summary(
                                value=[
                                    tensorboard.compat.proto.summary_pb2.Summary.Value(
                                        tag=f"{metric}/{name}",
                                        simple_value=value,
                                    )
                                    for name, value in values.items()
                                ]
                            ),
                        )
                    )
                writer.close()
                self.recorded.add(key)
        status = {
            **json.loads((self.root / "pipeline_status.json").read_text()),
            "training": {
                "step": step,
                "total_steps": total,
                "metrics": metrics,
            },
            "eta_seconds": (
                (total - step) * config["batch_size"] / throughput
                if throughput
                else None
            ),
            "recorded": sorted(self.recorded),
            "updated_at": now,
        }
        if status["stage"] in {"games", "complete"}:
            status.update(self.game_progress(plan))
        return self.publish_status(status)

    def game_progress(self, plan):
        comparison = self.root / "games/summary.json"
        game_log = self.root / "games.log"
        progress = (
            re.findall(
                r"games=(\d+)/(\d+) elapsed_seconds=([\d.]+)\n",
                game_log.read_text(),
            )
            if game_log.exists()
            else []
        )
        game_file = self.root / "games/games.jsonl"
        completed = (
            game_file.read_bytes().count(b"\n") if game_file.exists() else 0
        )
        requested = (
            int(progress[-1][1]) if progress else plan["evaluation_games"]
        )
        rate = None
        if len(progress) > 1:
            # Use the latest uninterrupted launch, including resume.
            begin = len(progress) - 1
            while begin and float(progress[begin - 1][2]) < float(
                progress[begin][2]
            ):
                begin -= 1
            elapsed = float(progress[-1][2]) - float(progress[begin][2])
            if elapsed > 0:
                rate = (
                    int(progress[-1][0]) - int(progress[begin][0])
                ) / elapsed
        status = {
            "games": {
                "completed": completed,
                "total": requested,
                "per_hour": None if rate is None else rate * 3600,
                "comparison": (
                    json.loads(comparison.read_text())
                    if comparison.exists()
                    else None
                ),
            }
        }
        status["progress_percent"] = 100 * completed / requested
        status["eta_seconds"] = (
            max(0, requested - completed) / rate if rate else None
        )
        return status

    def publish_status(self, status):
        temporary = self.root / "monitor_status.json.partial"
        temporary.write_text(json.dumps(status))
        temporary.replace(self.root / "monitor_status.json")
        return status

    def close(self):
        for writer in self.writers.values():
            writer.close()


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    watcher = Monitor(args.root)
    try:
        while True:
            watcher.poll(time.time())
            if args.once:
                break
            time.sleep(5)
    finally:
        watcher.close()
