"""Native duplicate matches and paired estimates over full seat rotations."""

import atexit
import concurrent.futures
import gzip
import itertools
import json
import multiprocessing
import pathlib
import secrets
import time

import numpy

WORKER_POLICIES = {}


def completed_games(root, requested):
    root = pathlib.Path(root)
    path = root / "games.jsonl"
    if not path.exists():
        return []
    raw = path.read_bytes()
    complete = raw[: raw.rfind(b"\n") + 1]
    if complete != raw:
        path.write_bytes(complete)
    results = []
    for line in complete.splitlines():
        row = json.loads(line)
        if row.get("format") != "riichienv_duplicate_south":
            raise ValueError("Evaluation format mismatch; use a fresh output")
        if (row["seed"], *row["new_seats"]) not in requested:
            continue
        with gzip.open(root / row["replay"], "rt") as stream:
            events = [json.loads(line) for line in stream]
        if events and events[-1]["type"] == "end_game":
            results.append(row)
    return results


def summarize(results, bootstrap_seed=0):
    rotations = {}
    for row in results:
        if row.get("format") != "riichienv_duplicate_south":
            raise ValueError("Cannot combine different evaluation formats")
        rotation = rotations.setdefault(row["seed"], {})
        seat = tuple(row["new_seats"])
        if seat not in ((0,), (1,), (2,), (3,)) or seat in rotation:
            raise ValueError("Expected one distinct candidate seat per game")
        rotation[seat] = row
    complete = [
        row
        for _, rotation in sorted(rotations.items())
        if len(rotation) == 4
        for _, row in sorted(rotation.items())
    ]
    if not complete:
        return None
    grouped = {}
    for row in complete:
        pair = grouped.setdefault(row["seed"], {"new": [], "old": []})
        for seat in range(4):
            pair["new" if seat in row["new_seats"] else "old"].append(
                [
                    row["ranks"][seat],
                    row["ranks"][seat] == 1,
                    row["ranks"][seat] == 4,
                    row["rank_points"][seat],
                    row["scores"][seat],
                ]
            )
    metrics = ("placement", "first", "fourth", "rank_points", "score")
    clusters = {
        model: numpy.array(
            [numpy.mean(pair[model], 0) for pair in grouped.values()]
        )
        for model in ("new", "old")
    }
    difference = clusters["new"] - clusters["old"]
    generator = numpy.random.default_rng(bootstrap_seed)
    draws = generator.integers(0, len(grouped), (2000, len(grouped)))
    bootstrap = difference[draws].mean(1)
    return {
        "format": "riichienv_duplicate_south",
        "observed_games": len(results),
        "compared_games": len(complete),
        "seed_groups": len(grouped),
        "models": {
            model: dict(zip(metrics, values.mean(0).tolist()))
            for model, values in clusters.items()
        },
        "new_minus_old": dict(zip(metrics, difference.mean(0).tolist())),
        "paired_bounds": dict(
            zip(
                metrics,
                numpy.quantile(bootstrap, [0.025, 0.975], axis=0).T.tolist(),
            )
        ),
    }


def start_worker(root, devices):
    from . import frozen_policy

    device = devices.get()
    for name in ("new", "old"):
        WORKER_POLICIES[name] = frozen_policy.FrozenPolicy(
            model_path=root / name / "saved_model",
            device=device,
            memory_mb=2048,
        )
        atexit.register(WORKER_POLICIES[name].close)


def play_games(tasks):
    import riichienv

    for policy in WORKER_POLICIES.values():
        policy.reset()
    rule = riichienv.GameRule.default_tenhou()
    rule.duplicate = True
    games = [
        riichienv.RiichiEnv(game_mode=2, seed=seed, rule=rule)
        for _, seed, _ in tasks
    ]
    observations = [env.reset(oya=0) for env in games]
    while any(not env.done() for env in games):
        choices = [{} for _ in games]
        requests = [
            (game * 4 + actor, env, observation)
            for game, env in enumerate(games)
            if not env.done()
            for actor, observation in observations[game].items()
        ]
        for name, policy in WORKER_POLICIES.items():
            policy_requests = [
                request
                for request in requests
                if (request[0] % 4 == tasks[request[0] // 4][2])
                == (name == "new")
            ]
            if policy_requests:
                for request, action in zip(
                    policy_requests,
                    policy.select_many(
                        policy_requests,
                        batch_size=4 * 8,
                    ),
                    strict=True,
                ):
                    game, actor = divmod(request[0], 4)
                    choices[game][actor] = action
        for game, env in enumerate(games):
            if not env.done():
                observations[game] = env.step(choices[game])
    results = []
    for env, (root, seed, seat) in zip(games, tasks, strict=True):
        replay_path = f"replays/{seed}-{seat}.jsonl.gz"
        with gzip.open(root / replay_path, "wt") as stream:
            for event in env.mjai_log:
                stream.write(json.dumps(event) + "\n")
        ranks = list(env.ranks())
        results.append(
            {
                "format": "riichienv_duplicate_south",
                "seed": seed,
                "new_seats": [seat],
                "replay": replay_path,
                "ranks": ranks,
                "scores": list(env.scores()),
                "rank_points": [
                    [125, 60, -5, -255][rank - 1] for rank in ranks
                ],
            }
        )
    return results


def publish_summary(root, results, bootstrap_seed):
    comparison = summarize(results, bootstrap_seed)
    if comparison is not None:
        summary = root / "summary.json.partial"
        summary.write_text(json.dumps(comparison, indent=2))
        summary.replace(root / "summary.json")


def run(args):
    from . import frozen_policy, source_snapshot

    if args.games <= 0 or args.games % 4:
        raise ValueError(
            "Duplicate evaluation requires complete four-game groups"
        )
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "replays").mkdir(exist_ok=True)
    requested = {
        (args.seed + game // 4, game % 4) for game in range(args.games)
    }
    manifest = {
        "format": "riichienv_duplicate_south",
        "seed": args.seed,
        "games": args.games,
    }
    manifest_path = root / "evaluation.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text()) != manifest
    ):
        raise ValueError("Evaluation settings changed; use a fresh output")
    results = completed_games(root, requested)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    publish_summary(root, results, args.seed)
    completed = {(row["seed"], *row["new_seats"]) for row in results}
    tasks = [(root, seed, seat) for seed, seat in sorted(requested - completed)]
    if not tasks:
        return
    for name, model_path in (("new", args.new), ("old", args.old)):
        path = root / name
        if not path.exists():
            source = frozen_policy.source_directory(model_path)
            if source.name == "model":
                source_snapshot.freeze(
                    source, path / "source/model", include_native=True
                )
                (path / "source/feature_vector.py").write_bytes(
                    (source.parent / "feature_vector.py").read_bytes()
                )
                source_snapshot.freeze(
                    source.parent.parent / "log_dataset",
                    path / "log_dataset",
                    include_native=True,
                )
            else:
                # The fixed reference release uses the older flat package.
                source_snapshot.freeze(
                    source, path / "source", include_native=True
                )
            source_snapshot.freeze(
                pathlib.Path(model_path), path / "saved_model"
            )
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    devices = context.Queue()
    for worker in range(args.workers):
        devices.put(
            args.devices.split(",")[worker % len(args.devices.split(","))]
        )
    batches = [
        tasks[begin : begin + args.batch_games]
        for begin in range(0, len(tasks), args.batch_games)
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=start_worker,
        initargs=(root, devices),
    ) as executor:
        batches = iter(batches)
        pending = {
            executor.submit(play_games, batch)
            for batch in itertools.islice(batches, 2 * args.workers)
        }
        while pending:
            finished, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in finished:
                completed = future.result()
                with (root / "games.jsonl").open("a") as stream:
                    for row in completed:
                        stream.write(json.dumps(row) + "\n")
                results.extend(completed)
                publish_summary(root, results, args.seed)
                print(
                    f"games={len(results)}/{args.games} "
                    f"elapsed_seconds={time.monotonic() - started:.3f}",
                    flush=True,
                )
            pending.update(
                executor.submit(play_games, batch)
                for batch in itertools.islice(batches, len(finished))
            )
    devices.close()
    devices.join_thread()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", action="store_true")
    arguments = parser.parse_args()
    manifest = pathlib.Path(arguments.output) / "evaluation.json"
    arguments.seed = (
        json.loads(manifest.read_text())["seed"]
        if manifest.exists()
        else secrets.randbits(30)
    )
    arguments.games = 4 if arguments.experiment else 10000
    arguments.workers = 1 if arguments.experiment else 8
    arguments.devices = "1" if arguments.experiment else "1,2,3,4"
    arguments.batch_games = 8
    run(arguments)
