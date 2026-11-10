"""Run candidate AWR training."""

from training.candidate import run

if __name__ == "__main__":
    run.run(run.argument_parser().parse_args())
