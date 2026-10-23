"""Native scoring consumers: completion evidence, waits and bonus-han labels."""

import functools
import types

import numpy
import riichienv

from model import outcomes

from . import _native_collection, shanten

# Bonus counts are exact until ordinary hand value saturates at kazoe.
BONUS_HAN = tuple(range(14))
CELLS = tuple(map(tuple, numpy.argwhere(outcomes.PAYMENT_CELLS)))

NAMES = (
    "pinfu",
    "tanyao",
    "iipeikou",
    "haku",
    "hatsu",
    "chun",
    "seat_wind",
    "round_wind",
    "sanshoku_doujun",
    "ittsuu",
    "chanta",
    "honroutou",
    "toitoi",
    "sanankou",
    "sankantsu",
    "sanshoku_doukou",
    "chiitoitsu",
    "shousangen",
    "honitsu",
    "junchan",
    "ryanpeikou",
    "chinitsu",
    "yakuman",
)
YAKU_IDS = {
    "pinfu": 14,
    "tanyao": 12,
    "iipeikou": 13,
    "haku": 7,
    "hatsu": 8,
    "chun": 9,
    "seat_wind": 10,
    "round_wind": 11,
    "sanshoku_doujun": 17,
    "ittsuu": 16,
    "chanta": 15,
    "honroutou": 24,
    "toitoi": 21,
    "sanankou": 22,
    "sankantsu": 20,
    "sanshoku_doukou": 19,
    "chiitoitsu": 25,
    "shousangen": 23,
    "honitsu": 27,
    "junchan": 26,
    "ryanpeikou": 28,
    "chinitsu": 29,
}

FOCUSED_NAMES = tuple(YAKU_IDS)
YAKU_BITS = {value: 1 << NAMES.index(name) for name, value in YAKU_IDS.items()}


def completion_flags(
    tiles, melds, player_wind, round_wind, require_all=True, family_routes=False
):
    counts = numpy.bincount(numpy.asarray(tiles) // 4, minlength=34)
    distance = shanten.calculate_shanten(counts)
    if distance not in (0, 1):
        return numpy.full(len(NAMES), -1, numpy.int8)
    if distance == 0:
        waits = riichienv.HandEvaluator(tiles, melds).get_waits()
        if not waits:
            return numpy.full(len(NAMES), -1, numpy.int8)
        flags = (1 << len(NAMES)) - 1 if require_all else 0
        for tile in waits:
            physical = tile * 4 + (1 if tile in (4, 13, 22) else 0)
            for tsumo in (False, True):
                score = _native_collection.score_hand(
                    types.SimpleNamespace(
                        tiles=list(tiles) + [physical],
                        melds=melds,
                        agari_tile=physical,
                        dora_indicators=[],
                    ),
                    riichienv.Conditions(
                        tsumo=tsumo,
                        player_wind=(
                            riichienv.Wind.East,
                            riichienv.Wind.South,
                            riichienv.Wind.West,
                            riichienv.Wind.North,
                        )[player_wind],
                        round_wind=round_wind,
                    ),
                    [],
                )
                candidate = int(bool(score.yakuman)) << NAMES.index("yakuman")
                for yaku in score.yaku:
                    candidate |= YAKU_BITS.get(yaku, 0)
                if family_routes:
                    for name, distance in zip(
                        ("tanyao", "honitsu", "chanta"),
                        shanten.calculate_yaku_shanten(
                            counts + (numpy.arange(34) == tile),
                            [[t // 4 for t in meld.tiles] for meld in melds],
                        ),
                    ):
                        candidate &= ~(1 << NAMES.index(name))
                        candidate |= int(distance == -1) << NAMES.index(name)
                flags = flags & candidate if require_all else flags | candidate
                if require_all and not flags:
                    return numpy.zeros(len(NAMES), numpy.int8)
        return numpy.array(
            [(flags >> index) & 1 for index in range(len(NAMES))], numpy.int8
        )
    first_draws = shanten.improving(counts)[1]
    if not first_draws:
        return numpy.full(len(NAMES), -1, numpy.int8)
    result = numpy.full(len(NAMES), int(require_all), numpy.int8)
    for draw in first_draws:
        added = draw * 4 + (1 if draw in (4, 13, 22) else 0)
        hand = list(tiles) + [added]
        next_counts = counts + (numpy.arange(34) == draw)
        flags = numpy.zeros(len(NAMES), numpy.int8)
        for cut in numpy.flatnonzero(next_counts):
            if (
                shanten.calculate_shanten(
                    next_counts - (numpy.arange(34) == cut)
                )
                != 0
            ):
                continue
            retained = list(hand)
            retained.remove(next(tile for tile in retained if tile // 4 == cut))
            candidate = completion_flags(
                tiles=retained,
                melds=melds,
                player_wind=player_wind,
                round_wind=round_wind,
                require_all=require_all,
                family_routes=family_routes,
            )
            flags |= numpy.maximum(candidate, 0)
        result = result & flags if require_all else result | flags
    return result


@functools.lru_cache(maxsize=65536)
def path_completion_flags(counts, melds, player_wind, round_wind):
    """Share immutable scoring evidence across overlapping shortest paths."""
    return tuple(
        completion_flags(
            tiles=[
                tile * 4 + copy
                for tile, count in enumerate(counts)
                for copy in range(count)
            ],
            melds=[
                riichienv.Meld(
                    (
                        riichienv.MeldType.Chi,
                        riichienv.MeldType.Pon,
                        riichienv.MeldType.Daiminkan,
                        riichienv.MeldType.Ankan,
                        riichienv.MeldType.Kakan,
                    )[kind],
                    list(hand),
                    opened,
                )
                for kind, hand, opened in melds
            ],
            player_wind=player_wind,
            round_wind=round_wind,
            family_routes=True,
        )
        > 0
    )


def hand_yaku_fractions(
    counts, unavailable, melds, player_wind, round_wind, cuts=None, names=NAMES
):
    """Copy-weighted shortest routes; truncated searches are unknown."""
    paths = shanten.tenpai_paths(counts, unavailable, cuts)
    if paths is None:
        return numpy.full(len(names), -1.0)
    if set(names) <= {"tanyao", "honitsu", "chanta"}:
        # Necessary shape conditions only: rejected routes cannot carry any
        # requested family, regardless of winning tile or decomposition.
        # Honitsu/chanta include their full-flush/pure-outside counterparts.
        whole = paths[0] + numpy.bincount(
            [tile // 4 for meld in melds for tile in meld.tiles], minlength=34
        )
        possible = {
            "tanyao": ~whole[:, [0, 8, 9, 17, 18, 26, *range(27, 34)]].any(
                axis=1
            ),
            "honitsu": (
                whole[:, :27].reshape(-1, 3, 9).any(axis=2).sum(axis=1) <= 1
            ),
            "chanta": ~whole[:, [3, 4, 5, 12, 13, 14, 21, 22, 23]].any(axis=1),
        }
        possible = numpy.any([possible[name] for name in names], axis=0)
    else:
        possible = numpy.ones(len(paths[0]), numpy.bool_)
    melds = tuple(
        (int(meld.meld_type), tuple(meld.tiles), meld.opened) for meld in melds
    )
    supported = numpy.array(
        [
            (
                path_completion_flags(
                    counts=hand.tobytes(),
                    melds=melds,
                    player_wind=player_wind,
                    round_wind=round_wind,
                )
                if viable
                else (False,) * len(NAMES)
            )
            for hand, viable in zip(paths[0], possible)
        ],
        numpy.bool_,
    )
    return (paths[1] @ supported[:, [NAMES.index(name) for name in names]]) / (
        paths[1].sum()
    )


def score_waits(tiles, melds, context):
    evaluator = riichienv.HandEvaluator(list(tiles), list(melds))
    waits = evaluator.get_waits()
    furiten = context["missed_agari"] or bool(
        set(context["own_discards"]) & set(waits)
    )
    rows = []
    closed = all(not meld.opened for meld in melds)
    for base in waits:
        for red in (False, True) if base in (4, 13, 22) else (False,):
            suit = base // 9
            unseen = (
                1 - context["unavailable_red"][suit]
                if red
                else 4
                - context["unavailable_counts"][base]
                - (
                    1 - context["unavailable_red"][suit]
                    if base in (4, 13, 22)
                    else 0
                )
            )
            tile = base * 4 + (0 if red or base not in (4, 13, 22) else 1)
            while tile in tiles and tile < base * 4 + 3:
                tile += 1
            values = []
            for riichi, tsumo in (
                (False, False),
                (False, True),
                (True, False),
                (True, True),
            ):
                score = _native_collection.score_hand(
                    types.SimpleNamespace(
                        tiles=list(tiles) + [tile],
                        melds=melds,
                        agari_tile=tile,
                        dora_indicators=context["dora_indicators"],
                    ),
                    riichienv.Conditions(
                        riichi=riichi,
                        tsumo=tsumo,
                        player_wind=(
                            riichienv.Wind.East,
                            riichienv.Wind.South,
                            riichienv.Wind.West,
                            riichienv.Wind.North,
                        )[context["player_wind"]],
                        round_wind=context["round_wind"],
                    ),
                    [],
                )
                legal = (
                    score.is_win
                    and (not riichi or closed)
                    and (tsumo or not furiten)
                )
                points = (
                    score.tsumo_agari_ko
                    * (3 if context["player_wind"] == 0 else 2)
                    + (
                        0
                        if context["player_wind"] == 0
                        else score.tsumo_agari_oya
                    )
                    if tsumo
                    else score.ron_agari
                )
                yaku_ids = set(score.yaku)
                values.append(
                    {
                        "points": int(points) if legal else 0,
                        "han": score.han if legal else 0,
                        "fu": score.fu if legal else 0,
                        "legal": bool(legal),
                        "yakuman": bool(score.yakuman),
                        "yaku": [
                            int(
                                bool(score.yakuman)
                                if name == "yakuman"
                                else YAKU_IDS[name] in yaku_ids
                            )
                            for name in NAMES
                        ],
                    }
                )
            rows.append(
                {
                    "tile": 34 + suit if red else base,
                    "unseen": int(max(0, unseen)),
                    "values": values,
                }
            )
    summary = numpy.zeros((4, 8), numpy.float32)
    for scenario in range(4):
        legal = [
            row
            for row in rows
            if row["unseen"] > 0 and row["values"][scenario]["legal"]
        ]
        if legal:
            weights = numpy.array([row["unseen"] for row in legal])
            points = [row["values"][scenario]["points"] for row in legal]
            summary[scenario, :4] = [
                len({shanten.base_id(row["tile"]) for row in legal}),
                weights.sum(),
                min(points),
                max(points),
            ]
            for column, key in ((4, "points"), (5, "han"), (6, "fu")):
                summary[scenario, column] = numpy.average(
                    [row["values"][scenario][key] for row in legal],
                    weights=weights,
                )
        summary[scenario, 7] = int(furiten and scenario % 2 == 0)
    return summary, rows


def win_contexts(events):
    """Use the training replay's physical tiles, including concealed kans."""
    from log_dataset import replay

    state = replay.Replay("payment-bonus", [])
    for event in events:
        if event["type"] in ("start_game", "end_game"):
            continue
        if event["type"] in {"ankan", "kakan"}:
            # A robbed kan never completes and does not cancel ippatsu.
            kan_ippatsu = tuple(state.ippatsu)
        if event["type"] == "hora":
            actor = event["actor"]
            tsumo = actor == event["target"]
            last = next(
                row
                for row in reversed(state.history)
                if row["type"] in ("tsumo", "dahai", "kakan", "ankan")
            )
            tile = shanten.base_id(last["pai"]) * 4
            tile += int(
                tile in (16, 52, 88) and not shanten.is_red(last["pai"])
            )
            observation = state.legal_observation(actor)
            hand = observation.hand if tsumo else [*observation.hand, tile]
            chankan = not tsumo and last["type"] in {"kakan", "ankan"}
            ctx = types.SimpleNamespace(
                tiles=hand,
                melds=state.melds[actor],
                seat=actor,
                agari_tile=tile,
                dora_indicators=observation.dora_indicators,
                ura_indicators=[
                    shanten.base_id(marker) * 4
                    for marker in event["ura_markers"]
                ],
                conditions=riichienv.Conditions(
                    tsumo=tsumo,
                    riichi=state.accepted[actor],
                    double_riichi=(
                        state.accepted[actor]
                        and state.rivers[actor][0]["riichi"]
                        and not any(
                            row["type"]
                            in replay.CALL_TYPES | {"ankan", "kakan"}
                            for row in state.history[
                                : state.rivers[actor][0]["event_index"]
                            ]
                        )
                    ),
                    ippatsu=(
                        kan_ippatsu[actor] if chankan else state.ippatsu[actor]
                    ),
                    haitei=tsumo
                    and state.live_wall == 0
                    and state.rinshan_actor != actor,
                    houtei=not tsumo and state.live_wall == 0,
                    rinshan=tsumo and state.rinshan_actor == actor,
                    chankan=chankan,
                    tsumo_first_turn=tsumo
                    and not state.rivers[actor]
                    and not any(state.melds),
                    player_wind=(
                        riichienv.Wind.East,
                        riichienv.Wind.South,
                        riichienv.Wind.West,
                        riichienv.Wind.North,
                    )[(actor - state.start["oya"]) % 4],
                    round_wind="ESWN".index(state.start["bakaze"]),
                ),
            )
            if state.accepted[actor]:
                assert len(ctx.ura_indicators) == len(ctx.dora_indicators), (
                    state.kyoku_id,
                    actor,
                    "Missing hidden bonus indicators",
                )
            ctx.actual = _native_collection.score_hand(
                ctx, ctx.conditions, ctx.ura_indicators
            )
            if not ctx.actual.is_win:
                raise ValueError((state.kyoku_id, event, "Unscorable win"))
            yield state.kyoku_id, ctx
        state.apply(event)


def bonus_han_scores(ctx):
    """One base hand, rescored by total situational/hidden bonus han.

    Riichi (including double riichi), ippatsu, last tile, rinshan, chankan
    and ura contribute only their combined han. Public dora/red and menzen
    tsumo stay in the base hand. The last bucket is 13+, already kazoe.
    True yakuman retain their actual multiplicity, independently of bonuses.
    """
    base = _native_collection.score_hand(
        ctx,
        riichienv.Conditions(
            tsumo=ctx.conditions.tsumo,
            tsumo_first_turn=ctx.conditions.tsumo_first_turn,
            player_wind=ctx.conditions.player_wind,
            round_wind=ctx.conditions.round_wind,
        ),
        [],
    )
    observed = 0 if base.yakuman else min(ctx.actual.han - base.han, 13)
    result = numpy.full((len(BONUS_HAN), 3), -1, numpy.int32)
    details = numpy.full((len(BONUS_HAN), 3), -1, numpy.int32)
    for bonus in (
        BONUS_HAN[:2] if any(meld.opened for meld in ctx.melds) else BONUS_HAN
    ):
        if not base.is_win and bonus == 0:
            continue
        han = base.han if base.yakuman else base.han + bonus
        points = (
            8000 * (han // 13)
            if base.yakuman
            else outcomes.basic_points(han, base.fu)
        )
        dealer = ctx.conditions.player_wind == riichienv.Wind.East
        result[bonus] = [
            (
                0
                if ctx.conditions.tsumo
                else 100 * ((points * (6 if dealer else 4) + 99) // 100)
            ),
            (
                100 * ((points * (2 if dealer else 1) + 99) // 100)
                if ctx.conditions.tsumo
                else 0
            ),
            (
                100 * ((points * 2 + 99) // 100)
                if ctx.conditions.tsumo and not dealer
                else 0
            ),
        ]
        details[bonus] = (han, base.fu, int(base.yakuman))
    numpy.testing.assert_array_equal(
        result[observed],
        [
            ctx.actual.ron_agari,
            ctx.actual.tsumo_agari_ko,
            ctx.actual.tsumo_agari_oya,
        ],
    )
    return result, observed, details


def round_labels(events):
    """Absolute-seat labels, checked against the canonical payment tensor."""
    from model import payment_scoring

    rounds = {}
    for event in events:
        if event["type"] == "start_kyoku":
            key = f"{event['bakaze']}{event['kyoku']}-{event['honba']}"
            rounds[key] = []
        if event["type"] not in ("start_game", "end_game"):
            rounds[key].append(event)
    result = {
        key: {
            "bonus": numpy.full((2, 4, 4), -1, numpy.int16),
            "score": numpy.full((2, 4, 4, len(BONUS_HAN)), -1, numpy.int16),
            "payment": outcomes.terminal_payments(rows),
        }
        for key, rows in rounds.items()
    }
    for key, ctx in win_contexts(events):
        amounts, observed, details = bonus_han_scores(ctx)
        kind = int(ctx.conditions.tsumo)
        payments = result[key]["payment"][kind, :, ctx.seat]
        dealer = ctx.conditions.player_wind == riichienv.Wind.East
        total = (
            amounts[:, 1] * (3 if dealer else 2)
            + (0 if dealer else amounts[:, 2])
            if kind
            else amounts[:, 0]
        )
        assert total[observed] == payments.sum(), (
            key,
            ctx.seat,
            total[observed],
            payments.tolist(),
        )
        for payer in numpy.flatnonzero(payments):
            if kind:
                dealer_seat = (
                    ctx.seat
                    - (
                        riichienv.Wind.East,
                        riichienv.Wind.South,
                        riichienv.Wind.West,
                        riichienv.Wind.North,
                    ).index(ctx.conditions.player_wind)
                ) % 4
                shares = amounts[:, 2 if payer == dealer_seat else 1]
            else:
                shares = amounts[:, 0]
            # Pao changes which seat pays the same scored total. The observed
            # allocation is label-only; ordinary transfers use scorer shares.
            if shares[observed] != payments[payer]:
                # Replayed native state can omit the pao flag. The recorded
                # allocation and independently scored yakuman total establish
                # this label; ordinary hands must match their scorer shares.
                if not ctx.actual.yakuman:
                    raise ValueError((key, "Unexpected payment allocation"))
                shares = (
                    total.astype(numpy.int64)
                    * payments[payer]
                    // total[observed]
                )
            valid = numpy.all(amounts >= 0, axis=1)
            result[key]["score"][kind, payer, ctx.seat, valid] = [
                payment_scoring.score_class(int(han), int(fu), bool(yakuman))
                for han, fu, yakuman in details[valid]
            ]
            result[key]["bonus"][kind, payer, ctx.seat] = (
                -1 if ctx.actual.yakuman else observed
            )
            assert shares[observed] == payments[payer]
    return result
