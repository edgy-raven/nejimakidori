"""Held-out critic charts stay separate from training progress."""

import json

import tensorboard.backend.event_processing.event_accumulator

from service import training_monitor


def test_monitor_records_validation_once_without_advancing_training(tmp_path):
    (tmp_path / "candidate").mkdir()
    (tmp_path / "candidate/run_config.json").write_text(
        json.dumps(
            {
                "batch_size": 512,
                "phase_budget": [
                    {
                        "phase": "natural_start",
                        "epochs": 1,
                        "steps_per_epoch": 1000,
                    }
                ],
            }
        )
    )
    (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "pipeline_status.json").write_text('{"stage": "training"}')
    (tmp_path / "training.log").write_text(
        "phase=natural_start\nEpoch 1/1\n"
        "step=100 examples_per_second=512 loss=1 policy_loss=0.7\n"
        "validation_step=100 placement_observed_mse=0.38 "
        "placement_prior_mse=0.39 points_observed_mse=0.09 "
        "points_prior_mse=0.1 policy_loss=0.8 imitation_accuracy=0.7\n"
        "validation_step=200 placement_observed_mse=0.37"
    )
    monitor = training_monitor.Monitor(tmp_path)
    for now in (100, 101):
        status = monitor.poll(now)
        assert status["training"]["step"] == 100
        assert ("validation", 100) in monitor.recorded
        assert ("validation", 200) not in monitor.recorded
    monitor.close()
    events = (
        tensorboard.backend.event_processing.event_accumulator.EventAccumulator(
            str(tmp_path / "tensorboard/validation/live")
        )
    )
    events.Reload()
    points = events.Scalars("validation/points_observed_mse")
    assert len(points) == 1
    assert points[0].step == 100
    assert abs(points[0].value - 0.09) < 1e-6
    assert "total_loss" not in events.Tags()["scalars"]
