"""Job: Import allowlisted RiichiLab opponents as explicitly assigned dan seats."""

import argparse

from log_dataset import riichilab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-bots", nargs="+", type=int, required=True)
    parser.add_argument(
        "--teachers",
        nargs="+",
        required=True,
        help="Explicit BOT_ID:DAN pairs, e.g. 120:8",
    )
    parser.add_argument("--output", required=True)
    riichilab.run(parser.parse_args())


if __name__ == "__main__":
    main()
