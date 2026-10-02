"""Job: Shared supervised learning with explicit valid-label means."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument(
        "--dealership-advice-strength", type=float, default=0.25
    )
    arguments = parser.parse_args()
    if arguments.dealership_advice_strength < 0:
        parser.error("dealership advice strength must be nonnegative")
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES", "0" if arguments.experiment else "0,2,3,4"
    )
    from model import train

    train.run(
        arguments.data,
        arguments.output,
        arguments.experiment,
        arguments.dealership_advice_strength,
    )


if __name__ == "__main__":
    main()
