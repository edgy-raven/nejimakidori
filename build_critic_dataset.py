"""Job: Streamable full-corpus critic records with same-hand scoring labels."""

import argparse

from model import critic_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", required=True)
    parser.add_argument("--extra-archives", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--reserve", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit-games", type=int)
    critic_data.run(parser.parse_args())


if __name__ == "__main__":
    main()
