"""Queue cached review disagreements, then collect bounded one-hand trials."""

import argparse
import collections
import gzip
import importlib.util
import json
import pathlib
import shutil
import tempfile
import time

import numpy
from log_dataset import replay, review, rollout_collection, shanten
from model import actions, features, frozen_policy, outcomes, source_snapshot

import riichienv


def prepare(state):
    """Prepare fixed root constraints once for all sampled walls."""
    start = state["events"][0]
    constraints = [
        tile
        for block in range(3)
        for offset in range(4)
        for tile in start["tehais"][(start["oya"] + offset) % 4][
            block * 4 : block * 4 + 4
        ]
    ] + [
        start["tehais"][(start["oya"] + offset) % 4][12] for offset in range(4)
    ]
    slots = dict(enumerate(constraints))
    slots[131] = start["dora_marker"]
    live, rinshan, indicators = 52, 135, 1
    after_kan = False
    for event in state["events"][1:]:
        if event["type"] in {"ankan", "kakan", "daiminkan"}:
            after_kan = True
        elif event["type"] == "tsumo":
            slots[rinshan if after_kan else live] = event["pai"]
            rinshan -= int(after_kan)
            live += int(not after_kan)
            after_kan = False
        elif event["type"] == "dora":
            slots[131 - 2 * indicators] = event["dora_marker"]
            indicators += 1
    wall = [-1] * 136
    stock = list(range(136))
    for slot, tile in slots.items():
        code = shanten.mjai_tile_code(tile)
        physical = next(
            tile for tile in stock if features.native_tile_code(tile) == code
        )
        wall[slot] = physical
        stock.remove(physical)
    expected = replay.Replay.from_start("rollout", start)
    for event in state["events"][1:]:
        expected.apply(event)
    return {
        "wall": wall,
        "stock": stock,
        "start": start,
        "actor": state["actor"],
        "decisions": [
            event
            for event in state["events"]
            if event["type"]
            in {"dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan"}
        ],
        "draws": [
            event["pai"]
            for event in state["events"]
            if event["type"] == "tsumo"
        ],
        "expected": expected,
        "hands": [
            collections.Counter(map(features.native_tile_code, hand))
            for hand in expected.native_env.hands
        ],
    }


def recover(plan, seed):
    """Sample a wall and replay the prefix against its prepared constraints."""
    wall = plan["wall"].copy()
    stock = plan["stock"].copy()
    numpy.random.default_rng(seed).shuffle(stock)
    for slot in range(136):
        if wall[slot] == -1:
            wall[slot] = stock.pop()
    env = riichienv.RiichiEnv(game_mode=2, seed=seed)
    observations = env.reset(
        oya=plan["start"]["oya"],
        wall=wall,
        round_wind="ESWN".index(plan["start"]["bakaze"]),
        scores=plan["start"]["scores"],
        honba=plan["start"]["honba"],
        kyotaku=plan["start"]["kyotaku"],
        seed=seed,
    )
    decision_ix = 0
    kinds = {
        "dahai": riichienv.ActionType.DISCARD,
        "reach": riichienv.ActionType.RIICHI,
        "chi": riichienv.ActionType.CHI,
        "pon": riichienv.ActionType.PON,
        "daiminkan": riichienv.ActionType.DAIMINKAN,
        "ankan": riichienv.ActionType.ANKAN,
        "kakan": riichienv.ActionType.KAKAN,
    }
    while True:
        draws = [
            event["pai"] for event in env.mjai_log if event["type"] == "tsumo"
        ]
        if decision_ix == len(plan["decisions"]) and draws == plan["draws"]:
            break
        choices = {}
        consumed = False
        for seat, observation in observations.items():
            if (
                decision_ix < len(plan["decisions"])
                and seat == plan["decisions"][decision_ix]["actor"]
                and any(
                    action.action_type
                    == kinds[plan["decisions"][decision_ix]["type"]]
                    for action in observation.legal_actions()
                )
            ):
                if plan["decisions"][decision_ix]["type"] == "kakan":
                    choices[seat] = next(
                        action
                        for action in observation.legal_actions()
                        if action.action_type == riichienv.ActionType.KAKAN
                        and features.native_tile_code(action.tile)
                        == shanten.mjai_tile_code(
                            plan["decisions"][decision_ix]["pai"]
                        )
                    )
                else:
                    choices[seat] = observation.select_action_from_mjai(
                        json.dumps(plan["decisions"][decision_ix])
                    )
                if choices[seat] is None:
                    raise ValueError(
                        "Recorded action is not legal: "
                        f"{plan["decisions"][decision_ix]}"
                    )
                consumed = True
            else:
                choices[seat] = next(
                    action
                    for action in observation.legal_actions()
                    if action.action_type == riichienv.ActionType.PASS
                )
        observations = env.step(choices)
        decision_ix += consumed
        if env.done() or any(
            event["type"] in {"hora", "ryukyoku"} for event in env.mjai_log
        ):
            raise ValueError("Recorded prefix terminated before its decision")
    if plan["actor"] not in observations:
        raise ValueError("Recovered root is not at the marked actor's decision")
    assert env.scores() == plan["expected"].native_env.scores()
    assert len(env.wall) == plan["expected"].live_wall + 14
    for seat in range(4):
        assert (
            collections.Counter(map(features.native_tile_code, env.hands[seat]))
            == plan["hands"][seat]
        )
    assert list(map(features.native_tile_code, env.dora_indicators)) == [
        shanten.mjai_tile_code(tile) for tile in plan["expected"].dora_markers
    ]
    return env, plan["expected"]


def trial(root, seeds, policy):
    """Batch independent walls, with isolated branches for each action."""
    results = {seed: {} for seed in seeds}
    branches = {}
    policy.reset()
    actor = root["state"]["actor"]
    plan = prepare(root["state"])
    for seed in seeds:
        env, state = recover(plan, seed)
        observation = env.get_observation(actor)
        request = {
            "features": root["packed"],
            "actor": actor,
            "legal_actions": observation.legal_actions(),
            "drawn_tile": observation.drawn_tile,
            "phase": ("DISCARD", "RESPONSE", "KAN", "RIICHI")[
                int(root["packed"]["decision_phase"])
            ],
        }
        candidates = {}
        for _, code, action, metadata in actions.policy_candidates(request):
            identifier = (
                code
                + {"DISCARD": 0, "RESPONSE": 111, "KAN": 260, "RIICHI": 37}[
                    request["phase"]
                ]
            )
            if request["phase"] == "RESPONSE" and code == 149:
                identifier = 261 + int(root["packed"]["trigger_tile_id"])
            candidates[identifier] = (action, metadata)
        event_count = len(env.mjai_log)
        for identifier in sorted(set(root["mark"]["nominations"].values())):
            numpy.random.seed(seed)
            branch_env = env.clone()
            observations = {
                seat: branch_env.get_observation(seat)
                for seat in branch_env.active_players
            }
            selected, metadata = candidates[identifier]
            if selected is None:
                continued = actions.continue_request(request)
                prediction = root["mark"]["prediction"]
                outputs = {
                    name.replace("probabilities", "logits"): numpy.log(
                        numpy.maximum([value], 1e-30)
                    )
                    for name, value in prediction.items()
                    if name.endswith("probabilities")
                }
                selected, metadata = actions.select_policy_action(
                    continued, outputs, 0
                )
            pending_cut = (
                metadata["discard_action"]
                if selected.action_type
                in {
                    riichienv.ActionType.RIICHI,
                    riichienv.ActionType.CHI,
                    riichienv.ActionType.PON,
                }
                else None
            )
            branches[len(branches)] = {
                "action": identifier,
                "seed": seed,
                "state": state,
                "env": branch_env,
                "event_count": event_count,
                "observations": observations,
                "choices": {actor: selected},
                "pending_cut": pending_cut,
                "random_state": numpy.random.get_state(),
            }
    while branches:
        # Separate streams keep each branch's replay and pending calls isolated.
        requests = [
            (4 * identifier + seat, branch["env"], observation)
            for identifier, branch in branches.items()
            for seat, observation in branch["observations"].items()
            if seat not in branch["choices"]
        ]
        if requests:
            for (stream, _, _), selected in zip(
                requests,
                # Only unresolved decisions need inference. The worker's
                # batch_size is a padding floor, not a limit on real rows.
                policy.select_many(requests, batch_size=1),
                strict=True,
            ):
                branches[stream // 4]["choices"][stream % 4] = selected
        for identifier, branch in list(branches.items()):
            branch["observations"] = branch["env"].step(branch["choices"])
            events = branch["env"].mjai_log[branch["event_count"] :]
            branch["event_count"] += len(events)
            if any(event["type"] in {"hora", "ryukyoku"} for event in events):
                results[branch["seed"]][branch["action"]] = settlement(
                    branch["env"], branch["state"]
                )
                del branches[identifier]
                continue
            branch["choices"] = {}
            if branch["pending_cut"] is not None:
                if actor in branch["observations"] and any(
                    action.action_type == riichienv.ActionType.DISCARD
                    for action in branch["observations"][actor].legal_actions()
                ):
                    numpy.random.set_state(branch["random_state"])
                    branch["choices"][actor] = actions.select_discard(
                        branch["observations"][actor].legal_actions(),
                        branch["pending_cut"],
                        branch["observations"][actor].drawn_tile,
                    )
                    branch["pending_cut"] = None
                elif any(
                    event["type"] in {"tsumo", "chi", "pon", "daiminkan"}
                    for event in events
                ):
                    branch["pending_cut"] = None
    return results


def settlement(env, state):
    import tensorflow as tf
    from model import joint_settlement

    events = env.mjai_log
    start = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "start_kyoku"
    )
    end = next(
        (
            index
            for index in range(start + 1, len(events))
            if events[index]["type"] == "start_kyoku"
        ),
        len(events),
    )
    events = [dict(event) for event in events[start:end]]
    # Native zero-transfer draws can retain the previous deposit delta.
    final = replay.Replay.from_start("rollout-settlement", events[0])
    for event in events:
        if event["type"] == "ryukyoku" and event["reason"] == "exhaustive_draw":
            ready = numpy.array(
                [
                    bool(
                        riichienv.HandEvaluator(
                            final.native_env.hands[seat], final.melds[seat]
                        ).get_waits()
                    )
                    for seat in range(4)
                ]
            )
            count = int(ready.sum())
            event["deltas"] = (
                numpy.where(ready, 3000 // count, -3000 // (4 - count)).tolist()
                if 0 < count < 4
                else [0] * 4
            )
        elif event["type"] == "ryukyoku":
            # Native special-draw deltas can include a previous riichi deposit.
            # Recover hand transfers from the authoritative boundary scores;
            # deposits and the final-game bank award are separate transfers.
            delta = (
                numpy.asarray(env.scores())
                - numpy.asarray(events[0]["scores"])
                + 1000 * numpy.asarray(final.accepted)
            )
            if env.done():
                delta[numpy.argmin(env.ranks())] -= 1000 * (
                    events[0]["kyotaku"] + sum(final.accepted)
                )
            event["deltas"] = delta.tolist()
        if event["type"] not in {
            "start_kyoku",
            "hora",
            "ryukyoku",
            "end_kyoku",
            "end_game",
        }:
            final.apply(event)
    payments = outcomes.terminal_payments(events)
    payments[3] += numpy.diag(state.accepted) * 1000
    scores = outcomes.terminal_scores(events)
    native_scores = numpy.array(env.scores())
    verified_scores = scores.copy()
    if env.done():
        verified_scores[numpy.argmin(outcomes.score_ranks(scores))] += (
            sum(events[0]["scores"]) + 1000 * events[0]["kyotaku"] - sum(scores)
        )
    numpy.testing.assert_array_equal(verified_scores, native_scores)
    if env.done():
        utility = outcomes.saint3_jade_south(native_scores, env.ranks())
    else:
        remaining = max(8 - (env.round_wind * 4 + env.oya), 0)
        dealerships = numpy.bincount(
            [(env.oya + offset) % 4 for offset in range(remaining)], minlength=4
        )
        order = (numpy.arange(4)[:, None] + numpy.arange(4)) % 4
        utility = joint_settlement.continuation_values(
            tf.constant(native_scores[order], tf.float32),
            tf.constant([remaining] * 4, tf.float32),
            tf.constant(dealerships[order], tf.float32),
        )[0].numpy()
    # Settlement labels precede the final-game award of outstanding sticks.
    return {
        "payments": payments.tolist(),
        "scores": scores.tolist(),
        "native_scores": env.scores(),
        "game_ended": env.done(),
        "projected_st3": utility.tolist(),
        "events": events,
    }


def enqueue(args):
    queue = rollout_collection.Queue(args.store)
    policy = {
        "model": str(pathlib.Path(args.model).resolve()),
        "revision": args.revision,
        "hashes": source_snapshot.hashes(args.model),
    }
    count = 0
    try:
        for path in args.reviews:
            with gzip.open(path, "rt") as stream:
                game = json.load(stream)
            for round_ix, annotations in enumerate(game["annotations"]):
                for cursor, annotation in enumerate(annotations):
                    if annotation is None:
                        continue
                    state, _ = review.reconstruct(
                        game["rounds"][round_ix], cursor
                    )
                    actor = annotation["actor"]
                    packed = features.pack(
                        state=state,
                        actor=actor,
                        phase=annotation["phase"],
                        legal_observation=state.legal_observation(actor),
                    )
                    identity = queue.mark(
                        state={
                            "events": state.history,
                            "actor": actor,
                            "game_id": game["gameId"],
                        },
                        packed=packed,
                        prediction=annotation["predictions"][0],
                        expert=annotation["expert_decisions"],
                        policy=policy,
                    )
                    count += identity is not None
    finally:
        queue.close()
    print(json.dumps({"marked": count}))


def collect(args):
    started = time.monotonic()
    queue = rollout_collection.Queue(args.store)
    policy = None
    runtime = tempfile.TemporaryDirectory(prefix="rollout-worker-")
    completed = 0
    reserve = args.reserve_seconds
    try:
        while time.monotonic() - started + reserve <= args.seconds:
            root = queue.next(args.trials)
            if root is None:
                break
            if policy is None or policy.model_path != pathlib.Path(
                root["policy"]["model"]
            ):
                if policy is not None:
                    policy.close()
                if (
                    source_snapshot.hashes(root["policy"]["model"])
                    != root["policy"]["hashes"]
                ):
                    raise ValueError(
                        "Queued continuation checkpoint has changed"
                    )
                source = frozen_policy.source_directory(root["policy"]["model"])
                target = pathlib.Path(tempfile.mkdtemp(dir=runtime.name))
                shutil.copytree(source.parent, target / "source")
                if (source.parent.parent / "log_dataset").is_dir():
                    shutil.copytree(
                        source.parent.parent / "log_dataset",
                        target / "log_dataset",
                    )
                spec = importlib.util.spec_from_file_location(
                    "rollout_frozen_policy",
                    target / "source/model/frozen_policy.py",
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                policy = module.FrozenPolicy(
                    root["policy"]["model"], source_path=target / "source/model"
                )
            if time.monotonic() - started + reserve > args.seconds:
                break
            trial_start = time.monotonic()
            seeds = range(
                args.seed + root["trial"],
                args.seed
                + min(args.trials, root["trial"] + args.trial_batch_size),
            )
            results = trial(root, seeds, policy)
            for offset, seed in enumerate(seeds):
                queue.complete(
                    {**root, "trial": root["trial"] + offset},
                    seed,
                    results[seed],
                )
            reserve = max(reserve, 2 * (time.monotonic() - trial_start))
            completed += len(seeds)
            print(
                json.dumps(
                    {
                        "root": root["id"],
                        "trial": root["trial"],
                        "trials": len(seeds),
                        "seconds": time.monotonic() - trial_start,
                    }
                ),
                flush=True,
            )
    finally:
        if policy is not None:
            policy.close()
        runtime.cleanup()
        queue.close()
    print(
        json.dumps(
            {
                "completed": completed,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    )


def run():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("mark")
    scan.add_argument("--store", required=True)
    scan.add_argument("--model", required=True)
    scan.add_argument("--revision", required=True)
    scan.add_argument("reviews", nargs="+")
    worker = commands.add_parser("collect")
    worker.add_argument("--store", required=True)
    worker.add_argument("--seconds", type=float, required=True)
    worker.add_argument("--reserve-seconds", type=float, default=30)
    worker.add_argument("--trials", type=int, default=32)
    worker.add_argument("--trial-batch-size", type=int, default=16)
    worker.add_argument("--seed", type=int, default=827100)
    args = parser.parse_args()
    if args.command == "mark":
        enqueue(args)
    else:
        if (
            min(
                args.seconds,
                args.reserve_seconds,
                args.trials,
                args.trial_batch_size,
            )
            <= 0
        ):
            parser.error("collection budgets and trials must be positive")
        collect(args)


if __name__ == "__main__":
    run()
