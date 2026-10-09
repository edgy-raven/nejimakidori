"""Prepare source archives and build the joint training dataset."""

import argparse
import fcntl
import json
import os
import pathlib
import types

from log_dataset import download, player_ranks, riichilab


def run(args):
    root = pathlib.Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = vars(args).copy()
        del plan["workers"]
        for name in ("archives", "databases", "output", "reserve"):
            if plan[name] is not None:
                plan[name] = str(pathlib.Path(plan[name]).resolve())
        plan["extra_archives"] = [
            str(pathlib.Path(path).resolve()) for path in args.extra_archives
        ]
        if (root / "pipeline.json").exists():
            previous = json.loads((root / "pipeline.json").read_text())
            previous.pop("workers", None)
            if previous != plan:
                raise ValueError("Dataset inputs changed; use a fresh output")
        else:
            temporary = root / "pipeline.json.partial"
            temporary.write_text(json.dumps(plan, indent=2) + "\n")
            temporary.replace(root / "pipeline.json")
        archives = args.archives
        if archives is None:
            archives = root / "downloads"
            download.run(archives)
        if args.databases:
            enriched = root / "archives"
            if not (enriched / "player_ranks.json").exists():
                player_ranks.run(archives, args.databases, enriched)
            archives = enriched
        extra = list(args.extra_archives)
        if args.own_bots:
            teachers = root / "teachers"
            if not (teachers / "player_ranks.json").exists():
                riichilab.run(
                    types.SimpleNamespace(
                        own_bots=args.own_bots,
                        teachers=[
                            f"{bot}:{dan}"
                            for bot, dan in riichilab.TEACHERS.items()
                        ],
                        output=str(teachers),
                    )
                )
            extra.append(str(teachers))
        # Record building is CPU work, including its spawned workers.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from log_dataset import critic_data

        critic_data.run(
            types.SimpleNamespace(
                archives=str(archives),
                extra_archives=extra,
                output=str(root / "data"),
                reserve=args.reserve,
                workers=args.workers,
                limit_games=args.limit_games,
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--archives",
        help="Existing archives; omit to download pinned Houou logs",
    )
    parser.add_argument(
        "--databases",
        help="Historical rank databases; omit for enriched archives",
    )
    parser.add_argument("--extra-archives", action="append", default=[])
    parser.add_argument("--own-bots", nargs="+", type=int)
    parser.add_argument("--reserve", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit-games", type=int)
    args = parser.parse_args()
    if args.archives is None and args.databases is None:
        parser.error(
            "downloading archives requires --databases for rank enrichment"
        )
    if args.workers < 1 or (
        args.limit_games is not None and args.limit_games < 1
    ):
        parser.error("workers and limit-games must be positive")
    run(args)


if __name__ == "__main__":
    main()
