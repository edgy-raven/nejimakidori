"""Complete-round labels separated from decision-time visible inputs."""

import types

import numpy
import riichienv

from log_dataset import (
    _native_collection,
    features,
    sakigiri_regret,
    scoring,
    shanten,
)
from model import outcomes, payment_scoring


def ron_labels(state, actor, phase):
    """Known ron and abortive-draw payments for one decision."""
    payments = numpy.full((3, 37), -1, numpy.int32)
    legal_ron = numpy.full((3, 37), -1, numpy.int8)
    if phase in ("DISCARD", "RIICHI", "RESPONSE", "KAN"):
        cuts = {
            features.native_tile_code(action.tile): action.tile
            for action in state.legal_observation(actor).legal_actions()
            if action.action_type == riichienv.ActionType.DISCARD
        }
        kans = {}
        if phase == "KAN":
            kans = {
                action.tile // 4: action
                for action in state.legal_observation(actor).legal_actions()
                if action.action_type
                in (riichienv.ActionType.ANKAN, riichienv.ActionType.KAKAN)
            }
            cuts = {code: action.tile for code, action in kans.items()}
        if phase == "RESPONSE":
            for action in features.prefer_red_calls(
                state.legal_observation(actor).legal_actions()
            ):
                if action.action_type not in (
                    riichienv.ActionType.CHI,
                    riichienv.ActionType.PON,
                ):
                    continue
                env = state.native_env.clone()
                env.apply_event(
                    {
                        **features.action_event(action),
                        "actor": actor,
                        "target": state.pending_discard["actor"],
                    }
                )
                cuts.update(
                    {
                        features.native_tile_code(cut.tile): cut.tile
                        for cut in env.get_observation(actor).legal_actions()
                        if cut.action_type == riichienv.ActionType.DISCARD
                    }
                )
        payments[:, list(cuts)] = 0
        legal_ron[:, list(cuts)] = 0
        ura = next(
            (
                event["ura_markers"]
                for event in state.events
                if event["type"] == "hora" and event["ura_markers"]
            ),
            [],
        )
        for relative in range(3):
            opponent = (actor + relative + 1) % 4
            hand = state.legal_observation(opponent).hand
            if len(hand) % 3 != 1:
                continue
            waits = riichienv.HandEvaluator(
                hand, state.melds[opponent]
            ).get_waits()
            if (
                not waits
                or state.missed_agari(opponent)
                or (
                    phase == "RESPONSE"
                    and state.pending_discard["actor"] != opponent
                    and shanten.base_id(state.pending_discard["pai"]) in waits
                )
                or set(waits)
                & {
                    shanten.base_id(row["pai"])
                    for row in state.rivers[opponent]
                }
            ):
                continue
            for code, tile in cuts.items():
                if tile // 4 not in waits:
                    continue
                if (
                    phase == "KAN"
                    and kans[code].action_type == riichienv.ActionType.ANKAN
                    and (
                        state.melds[opponent]
                        or len({t // 4 for t in hand + [tile]}) != 13
                        or any(
                            t // 4 < 27 and t // 4 % 9 not in (0, 8)
                            for t in hand + [tile]
                        )
                    )
                ):
                    continue
                riichi = (
                    state.declared[opponent]
                    if phase == "RESPONSE"
                    else state.accepted[opponent]
                )
                score = _native_collection.score_hand(
                    types.SimpleNamespace(
                        tiles=hand + [tile],
                        melds=state.melds[opponent],
                        agari_tile=tile,
                        dora_indicators=state.legal_observation(
                            opponent
                        ).dora_indicators,
                    ),
                    conditions=riichienv.Conditions(
                        riichi=riichi,
                        double_riichi=(
                            riichi
                            and state.rivers[opponent][0]["riichi"]
                            and not any(
                                event["type"]
                                in {"chi", "pon", "daiminkan", "ankan", "kakan"}
                                for event in state.history[
                                    : state.rivers[opponent][0]["event_index"]
                                ]
                            )
                        ),
                        ippatsu=state.ippatsu[opponent] and phase != "RESPONSE",
                        houtei=state.live_wall == 0 and phase != "KAN",
                        chankan=phase == "KAN",
                        player_wind=(
                            riichienv.Wind.East,
                            riichienv.Wind.South,
                            riichienv.Wind.West,
                            riichienv.Wind.North,
                        )[(opponent - state.start["oya"]) % 4],
                        round_wind="ESWN".index(state.start["bakaze"]),
                    ),
                    ura=(
                        [
                            shanten.base_id(marker) * 4
                            for marker in ura[: len(state.dora_markers)]
                        ]
                        if riichi
                        else []
                    ),
                )
                if score.is_win:
                    legal_ron[relative, code] = 1
                    payments[relative, code] = (
                        -1
                        if (
                            (
                                riichi
                                and not score.yakuman
                                and len(ura) < len(state.dora_markers)
                            )
                            or state.native_env.pao[opponent]
                        )
                        else score.ron_agari + 300 * state.start["honba"]
                    )
        payments[:, (payments > 0).sum(axis=0) == 3] = -1
    return {
        "ron_payments": payments,
        "ron_legal": legal_ron,
    }


def round_targets(state, events):
    length = len(events)
    ready = numpy.zeros((length, 4), bool)
    hand_yaku_focus = numpy.zeros(4, numpy.int8)
    dangers = numpy.zeros((length, 4, 37), numpy.float32)
    completion = numpy.full((length, 4, 2, len(scoring.NAMES)), -1, numpy.int8)
    completion_score = numpy.full((length, 4, 2), -1, numpy.int16)
    for index, event in enumerate(events):
        if index:
            state.apply(event)
        for actor in range(4):
            counts = state.concealed_counts(actor)
            if counts.sum() % 3 != 1:
                continue
            ready[index, actor] = shanten.calculate_shanten(counts) == 0
            if index == 0 or (
                event.get("actor") == actor
                and event["type"] in {"dahai", "ankan", "kakan", "daiminkan"}
            ):
                if ready[index, actor] and not hand_yaku_focus[actor]:
                    hand_yaku_focus[actor] = any(
                        flag > 0 and name in scoring.FOCUSED_NAMES
                        for name, flag in zip(
                            scoring.NAMES,
                            scoring.completion_flags(
                                tiles=state.legal_observation(actor).hand,
                                melds=state.melds[actor],
                                player_wind=(actor - state.start["oya"]) % 4,
                                round_wind="ESWN".index(state.start["bakaze"]),
                                require_all=False,
                            ),
                        )
                    )
            if not ready[index, actor]:
                continue
            observation = state.legal_observation(actor)
            visible = state.public_tiles()
            _, waits = scoring.score_waits(
                observation.hand,
                state.melds[actor],
                {
                    "own_discards": [
                        shanten.base_id(row["pai"])
                        for row in state.rivers[actor]
                    ],
                    "missed_agari": state.missed_agari(actor),
                    "unavailable_counts": counts
                    + numpy.bincount(
                        [shanten.base_id(tile) for tile in visible],
                        minlength=34,
                    ),
                    "unavailable_red": [
                        observation.hand.count(tile) + visible.count(name)
                        for tile, name in zip(
                            (16, 52, 88), ("5mr", "5pr", "5sr")
                        )
                    ],
                    "dora_indicators": observation.dora_indicators,
                    "player_wind": (actor - state.start["oya"]) % 4,
                    "round_wind": "ESWN".index(state.start["bakaze"]),
                },
            )
            # Yaku and value are one scorer witness, selected by points.
            # Ordinary ron/tsumo use the actual declaration and known dora,
            # with no hypothetical ura, ippatsu or special-event bonuses.
            for mode in range(2):
                candidates = [
                    wait["values"][2 * int(state.declared[actor]) + mode]
                    for wait in waits
                    if wait["unseen"] > 0
                    and wait["values"][2 * int(state.declared[actor]) + mode][
                        "legal"
                    ]
                ]
                if candidates:
                    best = max(candidates, key=lambda value: value["points"])
                    completion[index, actor, mode] = best["yaku"]
                    completion_score[index, actor, mode] = (
                        payment_scoring.score_class(
                            best["han"],
                            best["fu"],
                            best["yakuman"],
                        )
                    )
            for wait in waits:
                dangers[index, actor, wait["tile"]] = (
                    wait["values"][2 if state.accepted[actor] else 0]["points"]
                    / 10000
                    if wait["unseen"] > 0
                    else 0
                )
        if event["type"] == "hora":
            ready[index, event["actor"]] = True
    # Forecast the next recorded tenpai shape, never a maximum over future
    # shapes. A tenpai with no legal completion stays unknown for that mode.
    for index in range(length - 2, -1, -1):
        missing = ~ready[index]
        completion[index, missing] = completion[index + 1, missing]
        completion_score[index, missing] = completion_score[index + 1, missing]
    return {
        "completion": completion,
        "completion_score": completion_score,
        "hand_yaku_focus": hand_yaku_focus,
        "sakigiri_pairs": sakigiri_regret.round_pairs(state.game_id, events),
        "tenpai_reached": ready.any(axis=0).astype(numpy.int8),
        "danger": dangers,
        "payments": outcomes.terminal_payments(events),
    }


def candidate_yaku_affinity(packed):
    """Action-specific visible route labels, not realized future yaku."""
    result = numpy.full((96, 3), -1, numpy.float16)
    for hand_ix in numpy.flatnonzero(packed["candidate_hand_valid"]):
        melds = []
        for tile, kind in numpy.argwhere(
            packed["candidate_melds"][hand_ix, :, :5] > 0
        ):
            for _ in range(packed["candidate_melds"][hand_ix, tile, kind]):
                melds.append(
                    riichienv.Meld(
                        (
                            riichienv.MeldType.Chi,
                            riichienv.MeldType.Pon,
                            riichienv.MeldType.Daiminkan,
                            riichienv.MeldType.Ankan,
                            riichienv.MeldType.Kakan,
                        )[kind],
                        (
                            [int(tile + offset) * 4 for offset in range(3)]
                            if kind == 0
                            else [
                                int(tile) * 4 + copy
                                for copy in range(3 if kind == 1 else 4)
                            ]
                        ),
                        bool(kind != 3),
                    )
                )
        removed = numpy.flatnonzero(
            packed["candidate_hand_counts"][0]
            != packed["candidate_hand_counts"][hand_ix]
        )
        shared_discard = len(removed) == 1 and (
            packed["candidate_hand_counts"][0, removed[0]]
            - packed["candidate_hand_counts"][hand_ix, removed[0]]
            == 1
        )
        # Search each discard from the common parent so its first draw/cut
        # nodes reuse the already-computed ukeire neighborhood. Routes and
        # their separate search budgets are identical to the post-cut hand.
        result[hand_ix] = scoring.hand_yaku_fractions(
            counts=packed["candidate_hand_counts"][
                0 if shared_discard else hand_ix
            ],
            unavailable=packed["known_unavailable_counts"],
            melds=melds,
            player_wind=int(packed["seat_wind"][0]),
            round_wind=int(packed["prevailing_wind"]),
            cuts=removed if shared_discard else None,
            names=("tanyao", "honitsu", "chanta"),
        )
    return result


def attach(observation, state, index, action_index):
    actor = observation.actor
    order = [(actor + offset) % 4 for offset in range(4)]
    observation.candidate_yaku_affinity = candidate_yaku_affinity(
        observation.features
    )
    observation.terminal_payment = outcomes.relative_payments(
        state.targets["payments"], actor, state.accepted
    )
    observation.opponent_shanten = numpy.full(3, -1, numpy.int8)
    observation.opponent_ukeire = numpy.full((3, 34), -1, numpy.int8)
    observation.opponent_hand_counts = numpy.zeros((3, 37), numpy.int8)
    for relative, seat in enumerate(order[1:]):
        hand = state.legal_observation(seat).hand
        for tile in hand + [
            tile for meld in state.melds[seat] for tile in meld.tiles
        ]:
            observation.opponent_hand_counts[
                relative, features.native_tile_code(tile)
            ] += 1
        if len(hand) % 3 == 1:
            distance, improving, _ = shanten.improving(
                state.concealed_counts(seat)
            )
            observation.opponent_shanten[relative] = min(3, distance)
            observation.opponent_ukeire[relative] = 0
            observation.opponent_ukeire[relative, improving] = 1
    observation.round_completion_yaku = state.targets["completion"][
        index, order
    ]
    observation.round_completion_score = state.targets["completion_score"][
        index, order
    ]
    observation.hand_yaku_focus = state.targets["hand_yaku_focus"][actor]
    observation.sakigiri_weight = 0.0
    if (
        observation.features["decision_phase"] == 0
        and observation.discard_action >= 0
        and action_index in state.targets["sakigiri_pairs"]
    ):
        pair, weight = state.targets["sakigiri_pairs"][action_index]
        observation.sakigiri_pair = numpy.array(pair, numpy.int8)
        observation.sakigiri_weight = weight
    if (
        state.targets["tenpai_reached"][actor]
        and state.events[action_index]["type"] == "dahai"
        and state.draws[actor] is not None
        and not state.accepted[actor]
        and not any(state.declared[seat] for seat in order[1:])
        and not state.targets["danger"][index, order[1:]].any()
        and not any(high_value(state, seat) for seat in order[1:])
    ):
        counts = observation.features["candidate_hand_counts"][0].astype(
            numpy.int32
        )
        distance, improving, _ = shanten.improving(
            counts - (numpy.arange(34) == shanten.base_id(state.draws[actor])),
            observation.features["known_unavailable_counts"],
        )
        retained_ukeire = shanten.improving(
            counts
            - (
                numpy.arange(34)
                == shanten.base_id(state.events[action_index]["pai"])
            ),
            observation.features["known_unavailable_counts"],
        )[1]
        missed_draws = []
        for event in state.events[action_index + 1 :]:
            if event.get("actor") != actor:
                continue
            if event["type"] in {
                "chi",
                "pon",
                "daiminkan",
                "ankan",
                "kakan",
            } or (event["type"] == "dahai" and not event["tsumogiri"]):
                break
            if event["type"] == "tsumo":
                draw = shanten.base_id(event["pai"])
                if draw in improving:
                    missed_draws.append(draw)
                # Include the draw that changes the retained hand's progress,
                # then end attribution to the old shape. Tsumogiri alone
                # does not end the window; passive visibility is not a change
                # in the hand's structural improving-tile set.
                if draw in retained_ukeire:
                    break
        if missed_draws:
            for cut in numpy.flatnonzero(
                observation.features["discard_action_legal"]
            ):
                after = counts - (numpy.arange(34) == shanten.base_id(cut))
                if shanten.calculate_shanten(after) == distance:
                    observation.discard_regret[cut] = numpy.mean(
                        ~numpy.isin(
                            missed_draws,
                            shanten.improving(
                                after,
                                observation.features[
                                    "known_unavailable_counts"
                                ],
                            )[1],
                        )
                    )
            if (
                numpy.unique(
                    observation.discard_regret[observation.discard_regret >= 0]
                ).size
                < 2
            ):
                observation.discard_regret[:] = -1


def high_value(state, actor):
    observation = state.legal_observation(actor)
    counts = state.concealed_counts(actor)
    if counts.sum() % 3 != 1 or shanten.calculate_shanten(counts) > 1:
        return False
    visible = state.public_tiles()
    unavailable = counts + numpy.bincount(
        [shanten.base_id(tile) for tile in visible], minlength=34
    )
    context = {
        "own_discards": [
            shanten.base_id(row["pai"]) for row in state.rivers[actor]
        ],
        "missed_agari": state.missed_agari(actor),
        "unavailable_counts": unavailable,
        "unavailable_red": [
            observation.hand.count(tile) + visible.count(name)
            for tile, name in zip((16, 52, 88), ("5mr", "5pr", "5sr"))
        ],
        "dora_indicators": observation.dora_indicators,
        "player_wind": (actor - state.start["oya"]) % 4,
        "round_wind": "ESWN".index(state.start["bakaze"]),
    }
    if shanten.calculate_shanten(counts) == 0:
        return any(
            row["unseen"] > 0 and row["values"][0]["points"] >= 7700
            for row in scoring.score_waits(
                observation.hand, state.melds[actor], context
            )[1]
        )
    improving = shanten.improving(counts, unavailable)[1]
    occupied = {
        *observation.hand,
        *(tile for meld in state.melds[actor] for tile in meld.tiles),
    }
    for draw in range(37):
        base = shanten.base_id(draw)
        if base not in improving:
            continue
        unseen = (
            1 - context["unavailable_red"][draw - 34]
            if draw >= 34
            else 4
            - unavailable[base]
            - (
                1 - context["unavailable_red"][base // 9]
                if base in (4, 13, 22)
                else 0
            )
        )
        if unseen <= 0:
            continue
        hand = list(observation.hand) + [
            next(
                tile
                for tile in range(base * 4, base * 4 + 4)
                if tile not in occupied
                and features.native_tile_code(tile) == draw
            )
        ]
        after_draw = counts + (numpy.arange(34) == base)
        for cut in numpy.flatnonzero(after_draw):
            if shanten.calculate_shanten(
                after_draw - (numpy.arange(34) == cut)
            ):
                continue
            retained = list(hand)
            # A retained red copy cannot reduce the best reachable value.
            retained.remove(max(tile for tile in retained if tile // 4 == cut))
            _, waits = scoring.score_waits(
                retained,
                state.melds[actor],
                {
                    **context,
                    "missed_agari": False,
                    "own_discards": context["own_discards"] + [int(cut)],
                    "unavailable_counts": unavailable
                    + (numpy.arange(34) == base),
                    "unavailable_red": numpy.asarray(context["unavailable_red"])
                    + (numpy.arange(34, 37) == draw),
                },
            )
            if any(
                row["unseen"] > 0 and row["values"][0]["points"] >= 7700
                for row in waits
            ):
                return True
    return False
