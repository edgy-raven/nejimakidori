"""Job: Download and verify the fixed Houou MJAI corpus."""

import argparse

from log_dataset import download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    for path in download.run(parser.parse_args().output):
        print(path)


if __name__ == "__main__":
    main()
