"""Learning-rate schedule and shared gradient application."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class OfflineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, peak=2e-4):
        self.config = dict(total_steps=total_steps, peak=peak)
        self.warmup_steps = max(1, total_steps * 2 // 100)
        self.decay_start = total_steps * 3 // 4
        self.decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=peak,
            decay_steps=total_steps - self.decay_start,
            alpha=0.1,
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.where(
            step < self.warmup_steps,
            self.config["peak"] * (0.1 + 0.9 * step / self.warmup_steps),
            self.decay(tf.maximum(step - self.decay_start, 0.0)),
        )

    def get_config(self):
        return self.config


def apply_gradients(optimizer, tape, loss, variables):
    gradients = [
        (gradient, variable)
        for gradient, variable in zip(tape.gradient(loss, variables), variables)
        if gradient is not None
    ]
    tf.debugging.assert_all_finite(
        tf.linalg.global_norm([gradient for gradient, _ in gradients]),
        "Nonfinite training gradient",
    )
    optimizer.apply_gradients(gradients)
