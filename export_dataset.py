"""Job: Stream normalized preprocessing components into training TFRecords."""

import argparse

from log_dataset import materialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    materialize.write(arguments.database, arguments.output, arguments.workers)


if __name__ == "__main__":
    main()
