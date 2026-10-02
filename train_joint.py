"""Job: Joint imitation and payment-critic training on five GPUs."""

import argparse

from model import critic_full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--policy-initialization", required=True)
    parser.add_argument(
        "--utility",
        help="JSON with every=1, reference_weight, reference_every, "
        "temperature, placement_scale, "
        "variance_scale and calibrated 19-cell correlation; "
        "omitted disables AWR",
    )
    parser.add_argument(
        "--batch-size", type=int, default=critic_full.BATCH_SIZE
    )
    parser.add_argument(
        "--mode", choices=tuple(critic_full.PHASES), default="full"
    )
    parser.add_argument(
        "--phase-batches",
        nargs=3,
        type=int,
        metavar=("NATURAL_START", "FOCUSED", "NATURAL_FINISH"),
        help="Total batch budgets for the three phases, not additional batches",
    )
    critic_full.run(parser.parse_args())


if __name__ == "__main__":
    main()
