"""Native matches and paired estimates over full seat rotations."""

import argparse
import atexit
import concurrent.futures
import gzip
import hashlib
import importlib.util
import itertools
import json
import multiprocessing
import pathlib
import secrets
import shutil
import time

import numpy
import tensorboard.compat.proto.event_pb2 as event_proto
import tensorboard.compat.proto.summary_pb2 as summary_proto
import tensorboard.summary.writer.event_file_writer as event_writer

from model import outcomes

WORKER_POLICIES = {}
MATCH_FORMATS = ("riichienv_duplicate_south", "riichienv_standard_south")


def completed_games(root, requested, match_format):
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
        if row["format"] != match_format:
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
        if (
            row["format"] not in MATCH_FORMATS
            or row["format"] != results[0]["format"]
        ):
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
        rank_points = outcomes.saint3_jade_south(row["scores"], row["ranks"])
        for seat in range(4):
            pair["new" if seat in row["new_seats"] else "old"].append(
                [
                    row["ranks"][seat],
                    row["ranks"][seat] == 1,
                    row["ranks"][seat] == 4,
                    rank_points[seat],
                    row["scores"][seat],
                    outcomes.SAINT3_JADE_SOUTH_OFFSETS[row["ranks"][seat] - 1],
                ]
            )
    metrics = (
        "placement",
        "first",
        "fourth",
        "rank_points",
        "score",
        "placement_points",
    )
    clusters = {
        model: numpy.array(
            [numpy.mean(pair[model], 0) for pair in grouped.values()]
        )
        for model in ("new", "old")
    }
    difference = clusters["new"] - clusters["old"]
    generator = numpy.random.default_rng(bootstrap_seed)
    bootstrap = numpy.concatenate(
        [
            difference[
                generator.integers(
                    0, len(grouped), (min(64, 2000 - begin), len(grouped))
                )
            ].mean(1)
            for begin in range(0, 2000, 64)
        ]
    )
    return {
        "format": results[0]["format"],
        "rank_point_formula": "ceil((final_score-25000)/1000 + ST3_offset)",
        "rank_point_offsets": list(outcomes.SAINT3_JADE_SOUTH_OFFSETS),
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


def start_worker(root, devices, batch_games):
    from training import frozen_policy

    device = devices.get()
    for name in ("new", "old"):
        if (root / name / "checkpoint").exists():
            from training import mortal_policy

            WORKER_POLICIES[name] = mortal_policy.MortalPolicy(
                root / name, device=device, capacity=batch_games * 3
            )
        else:
            WORKER_POLICIES[name] = frozen_policy.FrozenPolicy(
                model_path=root / name / "saved_model",
                device=device,
                memory_mb=1536,
            )
        atexit.register(WORKER_POLICIES[name].close)


def play_games(tasks):
    import riichienv

    for policy in WORKER_POLICIES.values():
        policy.reset([seed for _, seed, _ in tasks])
    rule = riichienv.GameRule.default_tenhou()
    match_format = json.loads((tasks[0][0] / "evaluation.json").read_text())[
        "format"
    ]
    rule.duplicate = match_format == "riichienv_duplicate_south"
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
        pending = []
        for name, policy in WORKER_POLICIES.items():
            policy_requests = [
                request
                for request in requests
                if (request[0] % 4 == tasks[request[0] // 4][2])
                == (name == "new")
            ]
            if policy_requests:
                policy.send_requests(policy_requests)
                pending.append((policy, policy_requests))
        # Both frozen policies can construct features and infer concurrently.
        for policy, policy_requests in pending:
            for request, action in zip(
                policy_requests,
                policy.receive_actions(policy_requests),
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
                "format": match_format,
                "seed": seed,
                "new_seats": [seat],
                "replay": replay_path,
                "ranks": ranks,
                "scores": list(env.scores()),
                "rank_points": outcomes.saint3_jade_south(
                    env.scores(), ranks
                ).tolist(),
            }
        )
    return results


def publish_summary(root, results, bootstrap_seed, writer):
    comparison = summarize(results, bootstrap_seed)
    if comparison is not None:
        summary = root / "summary.json.partial"
        summary.write_text(json.dumps(comparison, indent=2))
        summary.replace(root / "summary.json")
    values = {
        "progress/completed_games": len(results),
        "progress/percent": 100
        * len(results)
        / json.loads((root / "evaluation.json").read_text())["games"],
    }
    if comparison is not None:
        for side, metrics in comparison["models"].items():
            values.update(
                {f"{side}/{key}": value for key, value in metrics.items()}
            )
        for key, value in comparison["new_minus_old"].items():
            values[f"difference/{key}"] = value
            values[f"difference/{key}_lower"] = comparison["paired_bounds"][
                key
            ][0]
            values[f"difference/{key}_upper"] = comparison["paired_bounds"][
                key
            ][1]
    writer.add_event(
        event_proto.Event(
            wall_time=time.time(),
            step=len(results),
            summary=summary_proto.Summary(
                value=[
                    summary_proto.Summary.Value(tag=key, simple_value=value)
                    for key, value in values.items()
                ]
            ),
        )
    )
    writer.flush()


def merge_parts(parts, output, seed, games, *, writer, complete=False):
    """Publish one match from disjoint host runs with identical policies.

    Live snapshots may contain unfinished JSON lines and missing host
    manifests. Finalization requires every requested seat and replay.
    """
    root = pathlib.Path(output)
    root.mkdir(parents=True, exist_ok=True)
    requested = {(seed + game // 4, game % 4) for game in range(games)}
    manifests = []
    results = {}
    for part in map(pathlib.Path, parts):
        if not (part / "evaluation.json").exists():
            if complete:
                raise ValueError("Missing evaluation host manifest")
            continue
        manifest = json.loads((part / "evaluation.json").read_text())
        manifests.append(manifest)
        if {
            key: value
            for key, value in manifest.items()
            if key not in ("seed", "games")
        } != {
            key: value
            for key, value in manifests[0].items()
            if key not in ("seed", "games")
        }:
            raise ValueError("Evaluation hosts used different policies")
        path = part / "games.jsonl"
        raw = path.read_bytes() if path.exists() else b""
        for line in raw[: raw.rfind(b"\n") + 1].splitlines():
            row = json.loads(line)
            key = (row["seed"], *row["new_seats"])
            if key not in requested or key in results:
                raise ValueError("Overlapping or unexpected evaluation game")
            if row["replay"] != f"replays/{key[0]}-{key[1]}.jsonl.gz":
                raise ValueError("Evaluation replay does not match its game")
            if (
                not manifest["seed"]
                <= row["seed"]
                < (manifest["seed"] + manifest["games"] // 4)
            ):
                raise ValueError("Game is outside its host's seed range")
            if row["format"] != manifest["format"]:
                raise ValueError("Evaluation game format mismatch")
            results[key] = row
            if complete:
                with gzip.open(part / row["replay"], "rt") as stream:
                    events = [json.loads(event) for event in stream]
                if not events or events[-1]["type"] != "end_game":
                    raise ValueError("Incomplete evaluation replay")
                destination = root / row["replay"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(part / row["replay"], destination)
    if complete and results.keys() != requested:
        raise ValueError("Evaluation is missing requested games")
    if not manifests:
        return 0
    (root / "evaluation.json").write_text(
        json.dumps({**manifests[0], "seed": seed, "games": games}, indent=2)
        + "\n"
    )
    rows = [row for _, row in sorted(results.items())]
    pending = root / "games.jsonl.partial"
    pending.write_text("".join(json.dumps(row) + "\n" for row in rows))
    pending.replace(root / "games.jsonl")
    publish_summary(root, rows, seed, writer)
    return len(rows)


def run(args):
    from training import frozen_policy

    if args.games <= 0 or args.games % 4:
        raise ValueError("Evaluation requires complete four-game seat groups")
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "replays").mkdir(exist_ok=True)
    requested = {
        (args.seed + game // 4, game % 4) for game in range(args.games)
    }
    manifest = {
        "format": (
            "riichienv_standard_south"
            if args.mortal_source is not None
            else "riichienv_duplicate_south"
        ),
        "seed": args.seed,
        "games": args.games,
        "frozen_policy_randomness": "wall_actor_event_phase",
        "saved_models": {
            name: {
                str(path.relative_to(model_path)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(pathlib.Path(model_path).rglob("*"))
                if path.is_file()
            }
            for name, model_path in (("new", args.new), ("old", args.old))
            if name == "new" or args.mortal_source is None
        },
    }
    if args.mortal_source is not None:
        mortal_files = {
            "checkpoint": pathlib.Path(args.old),
            "source/libriichi.so": pathlib.Path(
                importlib.util.find_spec("libriichi").origin
            ),
            **{
                f"source/{name}": pathlib.Path(args.mortal_source)
                / "mortal"
                / name
                for name in ("model.py", "engine.py")
            },
            **{
                f"adapter/training/{name}": pathlib.Path(__file__).parent / name
                for name in ("mortal_policy.py", "frozen_policy.py")
            },
        }
        manifest["mortal"] = {}
        for name, path in mortal_files.items():
            with path.open("rb") as stream:
                manifest["mortal"][name] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    manifest_path = root / "evaluation.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text()) != manifest
    ):
        raise ValueError("Evaluation settings changed; use a fresh output")
    results = completed_games(root, requested, manifest["format"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    writer = event_writer.EventFileWriter(
        str(args.tensorboard or root / "tensorboard")
    )
    atexit.register(writer.close)
    publish_summary(root, results, args.seed, writer)
    completed = {(row["seed"], *row["new_seats"]) for row in results}
    tasks = [(root, seed, seat) for seed, seat in sorted(requested - completed)]
    if not tasks:
        writer.close()
        return
    for name, model_path in (("new", args.new), ("old", args.old)):
        path = root / name
        if not path.exists():
            if name == "old" and args.mortal_source is not None:
                for filename, source in mortal_files.items():
                    (path / filename).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, path / filename)
                continue
            source = frozen_policy.source_directory(model_path)
            frozen_policy.freeze_source(source.parent, path / "source")
            # Freeze adjacent packages used by the export's own worker.
            if (source.parent.parent / "log_dataset").is_dir():
                frozen_policy.freeze_source(
                    source.parent.parent / "log_dataset",
                    path / "log_dataset",
                )
            frozen_policy.freeze_source(
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
        initargs=(root, devices, args.batch_games),
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
                publish_summary(root, results, args.seed, writer)
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
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument(
        "--mortal-source",
        help="Mortal policy checkout; --old is its .pth or ZIP",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tensorboard", help="Event directory; default OUTPUT/tensorboard"
    )
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--devices", default="-1", help="Comma-separated GPU UUIDs; -1 uses CPU"
    )
    parser.add_argument("--batch-games", type=int, default=8)
    arguments = parser.parse_args()
    manifest = pathlib.Path(arguments.output) / "evaluation.json"
    if arguments.seed is None:
        arguments.seed = (
            json.loads(manifest.read_text())["seed"]
            if manifest.exists()
            else secrets.randbits(30)
        )
    if arguments.workers < 1 or arguments.batch_games < 1:
        parser.error("workers and batch-games must be positive")
    run(arguments)


if __name__ == "__main__":
    main()
