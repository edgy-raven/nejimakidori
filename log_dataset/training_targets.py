"""Complete-round labels separated from decision-time visible inputs."""

import types

import numpy
import riichienv

from log_dataset import (
    _native_collection,
    features,
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
    return {"ron_payments": payments, "ron_legal": legal_ron}


def round_targets(state, events, indices=None):
    """Forecast first tenpai witnesses, scoring only requested decisions."""
    length = len(events)
    ready = numpy.zeros((length, 4), bool)
    completion = numpy.full((length, 4, 2, len(scoring.NAMES)), -1, numpy.int8)
    completion_score = numpy.full((length, 4, 2), -1, numpy.int16)
    pending = numpy.zeros(4, bool)
    requested = None if indices is None else set(indices)
    for index, event in enumerate(events):
        if index:
            state.apply(event)
        if requested is None or index in requested:
            pending[:] = True
        for actor in numpy.flatnonzero(pending):
            counts = state.concealed_counts(actor)
            if counts.sum() % 3 != 1:
                continue
            ready[index, actor] = shanten.calculate_shanten(counts) == 0
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
        if event["type"] == "hora":
            ready[index, event["actor"]] = True
        pending &= ~ready[index]
        if (
            requested is not None
            and index >= max(requested)
            and not pending.any()
        ):
            break
    # Forecast the next recorded tenpai shape, never a maximum over future
    # shapes. A tenpai with no legal completion stays unknown for that mode.
    following = numpy.minimum.accumulate(
        numpy.where(ready, numpy.arange(length)[:, None], length - 1)[::-1],
        axis=0,
    )[::-1]
    return {
        "completion": completion[following, numpy.arange(4)],
        "completion_score": completion_score[following, numpy.arange(4)],
        "payments": outcomes.terminal_payments(events),
    }


def candidate_yaku_affinity(packed):
    """Action-specific visible route labels, not realized future yaku."""
    result = numpy.full((16, 3), -1, numpy.float16)
    for hand_ix in numpy.flatnonzero(packed["candidate_hand_valid"]):
        melds = []
        for tile, kind in zip(
            packed["candidate_meld_tiles"][hand_ix],
            packed["candidate_meld_types"][hand_ix],
            strict=True,
        ):
            if tile == 34:
                continue
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


def attach_beliefs(observation, state):
    """Decision-time hidden hands are labels, never public model features."""
    observation.opponent_shanten = numpy.full(3, -1, numpy.int8)
    observation.opponent_ukeire = numpy.full((3, 34), -1, numpy.int8)
    observation.opponent_hand_counts = numpy.zeros((3, 37), numpy.int8)
    for relative, seat in enumerate(
        (observation.actor + offset) % 4 for offset in range(1, 4)
    ):
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


def attach(observation, state, index, *, affinity=None):
    actor = observation.actor
    order = [(actor + offset) % 4 for offset in range(4)]
    observation.candidate_yaku_affinity = (
        candidate_yaku_affinity(observation.features)
        if affinity is None
        else affinity
    )
    observation.terminal_payment = outcomes.relative_payments(
        state.targets["payments"], actor, state.accepted
    )
    attach_beliefs(observation, state)
    observation.round_completion_yaku = state.targets["completion"][
        index, order
    ]
    observation.round_completion_score = state.targets["completion_score"][
        index, order
    ]
