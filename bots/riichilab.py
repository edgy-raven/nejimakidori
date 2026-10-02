"""Play RiichiLab sessions using a frozen release policy."""

import argparse
import asyncio
import collections
import itertools
import json
import pathlib
import time
import types

import aiohttp
import riichienv

import model.frozen_policy


def choose(policy, events, message):
    observation = riichienv.Observation.deserialize_from_base64(
        message["observation"]
    )
    events.extend(observation.events)
    action = policy.select(types.SimpleNamespace(mjai_log=events), observation)
    if action is None:
        raise ValueError("Policy returned an action outside the observation")
    response = json.loads(action.to_mjai())
    if action.action_type == riichienv.ActionType.DISCARD:
        response["tsumogiri"] = action.tile == observation.drawn_tile
    if not any(
        all(response.get(key) == value for key, value in legal.items())
        for legal in message["possible_actions"]
    ):
        raise ValueError("Policy action does not match possible_actions")
    response["request_id"] = message["request_id"]
    return response


def local_game(policy, seed):
    """Exercise masked observations and full policy decisions before online play."""
    policy.reset()
    env = riichienv.RiichiEnv(
        seed=seed, game_mode=1, rule=riichienv.GameRule.default_tenhou()
    )
    observations = env.reset()
    events = []
    decisions = 0
    while not env.done():
        selected = {}
        for actor, observation in observations.items():
            if actor == 0:
                response = choose(
                    policy,
                    events,
                    {
                        "observation": observation.serialize_to_base64(),
                        "possible_actions": [
                            json.loads(action.to_mjai())
                            for action in observation.legal_actions()
                        ],
                        "request_id": decisions,
                    },
                )
                selected[actor] = observation.select_action_from_mjai(
                    json.dumps(response)
                )
                decisions += 1
            else:
                selected[actor] = next(
                    (
                        action
                        for action in observation.legal_actions()
                        if action.action_type == riichienv.ActionType.DISCARD
                        and action.tile == observation.drawn_tile
                    ),
                    observation.legal_actions()[0],
                )
        observations = env.step(selected)
    return {
        "type": "local_result",
        "decisions": decisions,
        "scores": list(env.scores()),
    }


async def play(session, policy, args):
    policy.reset()
    receipts = collections.Counter()
    events = []
    ended = False
    with (args.output / f"{args.mode}-{time.time_ns()}.jsonl").open("x") as log:
        async with session.ws_connect(
            f"wss://game.riichi.dev/ws/{args.mode}",
            headers={
                "Authorization": f"Bearer {args.token_file.read_text().strip()}"
            },
            heartbeat=20,
        ) as ws:
            async for frame in ws:
                if frame.type == aiohttp.WSMsgType.ERROR:
                    raise ws.exception()
                if frame.type != aiohttp.WSMsgType.TEXT:
                    continue
                message = json.loads(frame.data)
                log.write(json.dumps({"received": message}) + "\n")
                log.flush()
                if "error" in message:
                    raise RuntimeError(message["error"])
                kind = message["type"]
                if kind == "request_action":
                    started = time.monotonic()
                    response = await asyncio.to_thread(
                        choose, policy, events, message
                    )
                    await ws.send_json(response)
                    log.write(
                        json.dumps(
                            {
                                "sent": response,
                                "inference_ms": 1000
                                * (time.monotonic() - started),
                                "deadline_ms": message["time"]["deadline_ms"],
                            }
                        )
                        + "\n"
                    )
                    log.flush()
                elif kind == "action_ack":
                    receipts[message["status"]] += 1
                    if message["status"] != "accepted":
                        print(json.dumps(message), flush=True)
                elif kind == "end_game":
                    ended = True
                    print(
                        json.dumps({**message, "receipts": receipts}),
                        flush=True,
                    )
                    if args.mode == "ranked":
                        break
                elif kind == "validation_result":
                    print(json.dumps(message), flush=True)
                    if not message["passed"]:
                        raise RuntimeError("RiichiLab validation failed")
                    return
                elif kind == "start_game":
                    print(json.dumps(message), flush=True)
    if not ended or args.mode == "validate":
        raise RuntimeError("Connection closed before the final result")
    if set(receipts) - {"accepted"}:
        raise RuntimeError("Stopping after a game with action receipt errors")


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    policy = model.frozen_policy.FrozenPolicy(
        model_path=args.model,
        source_path=args.source,
        device=args.device,
        memory_mb=args.memory_mb,
    )
    try:
        print(
            json.dumps(await asyncio.to_thread(local_game, policy, 20261001)),
            flush=True,
        )
        if args.mode != "local":
            async with aiohttp.ClientSession() as session:
                for _ in (
                    range(args.games) if args.games else itertools.count()
                ):
                    await play(session, policy, args)
    finally:
        policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--memory-mb", type=int, default=4096)
    parser.add_argument("--token-file", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--mode", choices=("local", "validate", "ranked"), required=True
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games; 0 keeps playing ranked games",
    )
    args = parser.parse_args()
    if args.games < 0 or (args.games == 0 and args.mode != "ranked"):
        parser.error("--games must be positive, or 0 for ranked play")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
