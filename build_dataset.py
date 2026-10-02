"""Job: Rebuild a queryable training corpus from enriched MJAI archives."""

import argparse

from log_dataset import cache, rebuild


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--stage", choices=("all", *cache.DEPENDENCIES), default="all"
    )
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--experiment", action="store_true")
    arguments = parser.parse_args()
    arguments.limit_games = 2 if arguments.experiment else None
    if arguments.experiment:
        arguments.workers = 1
    arguments.games_per_shard = 32
    rebuild.run(arguments)


if __name__ == "__main__":
    main()
