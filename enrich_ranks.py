"""Job: Create a fixed player-game selection from original Tenhou databases."""

import argparse

from log_dataset import player_ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", required=True)
    parser.add_argument("--databases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    player_ranks.run(args.archives, args.databases, args.output)


if __name__ == "__main__":
    main()
