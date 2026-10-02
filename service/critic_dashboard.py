"""Project joint-training events into a small, live TensorBoard dashboard."""

import argparse
import json
import pathlib
import re
import time

import tensorboard.backend.event_processing.event_accumulator
import tensorboard.backend.event_processing.event_file_loader
import tensorboard.compat.proto.event_pb2
import tensorboard.compat.proto.summary_pb2
import tensorboard.plugins.custom_scalar.layout_pb2
import tensorboard.plugins.custom_scalar.summary
import tensorboard.summary.writer.event_file_writer
import tensorboard.util.tensor_util

CHARTS = {
    "1 Imitation": {
        "NLL": "policy_loss",
        "Accuracy": "imitation_accuracy",
        "Top-3 accuracy": "imitation_top3_accuracy",
    },
    "2 Board representation": {
        "Combined loss": "board_loss",
        "Shanten": "opponent_shanten_loss",
        "Hand-end placement": "round_placement_loss",
        "Final placement": "final_placement_loss",
        "Ukeire": "opponent_ukeire_loss",
        "Yaku": "completion_yaku_loss",
        "Opponent hand": "opponent_hand_count_loss",
    },
    "3 Critic": {
        "Combined loss (before 0.1 weight)": "payment_auxiliary_loss",
        "Payment NLL": "payment_nll",
        "Point MSE": "point_mse",
        "Distribution CRPS": "payment_crps",
        "Severe-loss NLL": "loss_tail_nll",
        "Immediate risk": "risk_loss",
        "Discard ranking": "ranking_loss",
        "Bonus NLL": "bonus_nll",
        "Scenario NLL": "scenario_nll",
        "Han-fu NLL": "score_nll",
    },
}


class Monitor:
    def __init__(self, source, output):
        pathlib.Path(output).mkdir(parents=True, exist_ok=True)
        self.source = pathlib.Path(source)
        self.batch_size = json.loads(
            (self.source.parent / "plan.json").read_text()
        )["effective_batch"]
        self.loaders = {}
        self.previous = None
        self.tags = {
            f"{split}/{metric}": f"{category}/{title}/{split}"
            for category, charts in CHARTS.items()
            for title, metric in charts.items()
            for split in ("train", "validation")
        }
        saved = tensorboard.backend.event_processing.event_accumulator.EventAccumulator(
            str(output), size_guidance={"scalars": 1}
        )
        saved.Reload()
        self.recorded = {
            tag: saved.Scalars(tag)[-1].step for tag in saved.Tags()["scalars"]
        }
        self.writer = (
            tensorboard.summary.writer.event_file_writer.EventFileWriter(
                str(output)
            )
        )
        layout = tensorboard.plugins.custom_scalar.layout_pb2.Layout()
        for category, charts in CHARTS.items():
            group = layout.category.add(title=category, closed=False)
            for title in charts:
                chart = group.chart.add(title=title)
                chart.multiline.tag.extend(
                    "^" + re.escape(f"{category}/{title}/{split}") + "$"
                    for split in ("train", "validation")
                )
        chart = layout.category.add(
            title="4 Throughput", closed=False
        ).chart.add(title="Positions/sec (100-update wall-clock windows)")
        chart.multiline.tag.append("^4 Throughput/positions_per_second$")
        self.writer.add_event(
            tensorboard.compat.proto.event_pb2.Event(
                wall_time=time.time(),
                summary=tensorboard.compat.proto.summary_pb2.Summary.FromString(
                    tensorboard.plugins.custom_scalar.summary.pb(
                        layout
                    ).SerializeToString()
                ),
            )
        )

    def scalar(self, tag, value, event):
        if tag in self.recorded and event.step <= self.recorded[tag]:
            return
        self.writer.add_event(
            tensorboard.compat.proto.event_pb2.Event(
                wall_time=event.wall_time,
                step=event.step,
                summary=tensorboard.compat.proto.summary_pb2.Summary(
                    value=[
                        tensorboard.compat.proto.summary_pb2.Summary.Value(
                            tag=tag, simple_value=value
                        )
                    ]
                ),
            )
        )
        self.recorded[tag] = event.step

    def update(self):
        for path in sorted(self.source.glob("events.out.tfevents.*")):
            if path not in self.loaders:
                self.loaders[path] = (
                    tensorboard.backend.event_processing.event_file_loader.EventFileLoader(
                        str(path)
                    )
                )
            for event in self.loaders[path].Load():
                for value in event.summary.value:
                    if value.tag not in self.tags:
                        continue
                    self.scalar(
                        self.tags[value.tag],
                        float(
                            tensorboard.util.tensor_util.make_ndarray(
                                value.tensor
                            )
                        ),
                        event,
                    )
                    if value.tag == "train/policy_loss":
                        if self.previous is None:
                            self.previous = (event.step, event.wall_time)
                        elif event.step - self.previous[0] >= 100:
                            self.scalar(
                                "4 Throughput/positions_per_second",
                                (event.step - self.previous[0])
                                * self.batch_size
                                / (event.wall_time - self.previous[1]),
                                event,
                            )
                            self.previous = (event.step, event.wall_time)
        self.writer.flush()


def run(args):
    monitor = Monitor(args.source, args.output)
    while True:
        monitor.update()
        if args.once:
            break
        time.sleep(10)
    monitor.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--once", action="store_true")
    run(parser.parse_args())
