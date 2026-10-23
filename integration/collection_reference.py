"""Python collection oracle for native feature and rollout equivalence tests."""

import numpy
import riichienv

from log_dataset import _native_collection, features, replay
from model import actions, outcomes, self_play


def advance(ctx, events, statistics):
    """Update the existing replay and count each completed hand once."""
    for event in events:
        if event["type"] == "start_kyoku":
            ctx["hand"] += 1
            ctx["state"] = replay.Replay.from_start("self-play", event)
            ctx["pending"].clear()
            ctx["hand_stats"] = dict(
                reached=set(),
                called=set(),
                dealt=set(),
                won=set(),
                cuts=[0] * 4,
            )
            statistics["hands"] += 1
        elif ctx["state"] is not None:
            if event["type"] in ("chi", "pon", "daiminkan"):
                ctx["hand_stats"]["called"].add(event["actor"])
                ctx["pending"] = {
                    seat: plan
                    for seat, plan in ctx["pending"].items()
                    if seat == event["actor"] and plan[1] == event["type"]
                }
            elif event["type"] in ("tsumo", "dahai"):
                ctx["pending"].pop(event["actor"], None)
                if event["type"] == "dahai":
                    ctx["hand_stats"]["cuts"][event["actor"]] += 1
            elif event["type"] in ("hora", "ryukyoku", "end_kyoku"):
                ctx["pending"].clear()
            if event["type"] == "reach_accepted":
                ctx["hand_stats"]["reached"].add(event["actor"])
                statistics["riichi_turn_sum"] += ctx["hand_stats"]["cuts"][
                    event["actor"]
                ]
            elif event["type"] == "hora":
                ctx["hand_stats"]["won"].add(event["actor"])
                statistics["win_points"] += event["deltas"][event["actor"]]
                statistics["tsumo_wins"] += event["actor"] == event["target"]
                if event["actor"] != event["target"]:
                    ctx["hand_stats"]["dealt"].add(event["target"])
                    statistics["dealin_points"] -= event["deltas"][
                        event["target"]
                    ]
            elif event["type"] == "ryukyoku":
                statistics["draws"] += 1
                if event["reason"] == "exhaustive_draw":
                    statistics["exhaustive_draws"] += 1
                    statistics["exhaustive_tenpai_seats"] += sum(
                        riichienv.HandEvaluator(hand, melds).is_tenpai()
                        for hand, melds in zip(
                            ctx["state"].native_env.hands, ctx["state"].melds
                        )
                    )
            elif event["type"] == "end_kyoku":
                for name, seats in (
                    ("wins", "won"),
                    ("dealin_hands", "dealt"),
                    ("riichi_hands", "reached"),
                    ("open_hands", "called"),
                ):
                    statistics[name] += len(ctx["hand_stats"][seats])
                statistics["riichi_wins"] += len(
                    ctx["hand_stats"]["reached"] & ctx["hand_stats"]["won"]
                )
                statistics["riichi_dealins"] += len(
                    ctx["hand_stats"]["reached"] & ctx["hand_stats"]["dealt"]
                )
            elif event["type"] == "end_game":
                statistics["games"] += 1
            ctx["state"].apply(event)
            if event["type"] == "tsumo" and "tiles_left" in event:
                ctx["state"].live_wall = event["tiles_left"]
    ctx["cursor"] += len(events)


def collect(predict, games, seed):
    """Complete south games; networks stay fixed until collection finishes."""
    generator = numpy.random.default_rng(seed)
    envs = [
        riichienv.RiichiEnv(
            game_mode=2,
            seed=seed + game,
            rule=riichienv.GameRule.default_tenhou(),
        )
        for game in range(games)
    ]
    observations = [env.reset(oya=0) for env in envs]
    contexts = [dict(cursor=0, state=None, pending={}, hand=-1) for _ in envs]
    samples = []
    statistics = self_play.collection_counts()
    while any(not env.done() for env in envs):
        requests = []
        choices = [{} for _ in envs]
        for game, env in enumerate(envs):
            if env.done():
                continue
            ctx = contexts[game]
            advance(ctx, env.mjai_log_since(ctx["cursor"]), statistics)
            # Native maps do not promise seat order; keep seeded draws stable.
            for seat, observation in sorted(observations[game].items()):
                legal = observation.legal_actions()
                selected = actions.forced_policy_action(
                    ctx["state"], seat, legal
                )
                if seat in ctx["pending"] and any(
                    action.action_type == riichienv.ActionType.DISCARD
                    for action in legal
                ):
                    cut, _ = ctx["pending"].pop(seat)
                    if selected is None:
                        selected = actions.select_discard(
                            legal, cut, observation.drawn_tile
                        )
                if selected is not None:
                    choices[game][seat] = selected
                else:
                    _, request = actions.policy_request(
                        ctx["state"], seat, observation
                    )
                    requests.append((game, seat, request))
        while requests:
            packed = {
                name: numpy.stack([r[2]["features"][name] for r in requests])
                for name in _native_collection.INPUTS
            }
            outputs = {
                name: numpy.asarray(value)
                for name, value in predict(packed).items()
            }
            deferred = []
            for row, (game, seat, request) in enumerate(requests):
                candidates = actions.policy_candidates(request)
                head = actions.TRAIN_POLICY_HEADS.index(candidates[0][0])
                legal = numpy.zeros(150, bool)
                codes = [candidate[1] for candidate in candidates]
                legal[codes] = True
                scores = outputs[candidates[0][0]][row, codes]
                scores = scores - scores.max()
                log_probability = numpy.full(150, -1e9, numpy.float32)
                log_probability[codes] = scores - numpy.log(
                    numpy.exp(scores).sum()
                )
                chosen = int(
                    generator.choice(150, p=numpy.exp(log_probability))
                )
                _, _, selected, metadata = candidates[codes.index(chosen)]
                samples.append(
                    dict(
                        features=request["features"],
                        head=head,
                        legal=legal,
                        chosen=chosen,
                        behavior=log_probability,
                        game=game,
                        hand=contexts[game]["hand"],
                        seat=seat,
                    )
                )
                if selected is None:
                    deferred.append(
                        (game, seat, actions.continue_request(request))
                    )
                    continue
                choices[game][seat] = selected
                if "discard_action" in metadata and selected.action_type in (
                    riichienv.ActionType.RIICHI,
                    riichienv.ActionType.CHI,
                    riichienv.ActionType.PON,
                ):
                    contexts[game]["pending"][seat] = (
                        metadata["discard_action"],
                        features.action_event(selected, request["drawn_tile"])[
                            "type"
                        ],
                    )
            requests = deferred
        for game, env in enumerate(envs):
            if not env.done():
                observations[game] = env.step(choices[game])
    for ctx, env in zip(contexts, envs):
        advance(ctx, env.mjai_log_since(ctx["cursor"]), statistics)
    settlements = [
        outcomes.hand_continuation_states(env.mjai_log) for env in envs
    ]
    batch = {
        "features": {
            name: numpy.stack([sample["features"][name] for sample in samples])
            for name in _native_collection.INPUTS
        },
        **{
            name: numpy.asarray([sample[name] for sample in samples])
            for name in ("head", "legal", "chosen", "behavior")
        },
        "settlement": numpy.asarray(
            [
                settlements[sample["game"]][sample["hand"], sample["seat"]]
                for sample in samples
            ]
        ),
    }
    return batch, statistics
