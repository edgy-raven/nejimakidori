"""Scorer-based completion evidence and decision-time wait values."""

import numpy
import riichienv

from . import shanten

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


def completion_flags(tiles, melds, player_wind, round_wind, require_all=True):
    counts = numpy.bincount(numpy.asarray(tiles) // 4, minlength=34)
    distance = shanten.calculate_shanten(counts)
    if distance not in (0, 1):
        return numpy.full(len(NAMES), -1, numpy.int8)
    if distance == 0:
        waits = riichienv.HandEvaluator(tiles, melds).get_waits()
        if not waits:
            return numpy.full(len(NAMES), -1, numpy.int8)
        flags = numpy.full(len(NAMES), int(require_all), numpy.int8)
        for tile in waits:
            physical = tile * 4 + (1 if tile in (4, 13, 22) else 0)
            for tsumo in (False, True):
                score = riichienv.HandEvaluator(
                    list(tiles) + [physical], melds
                ).calc(
                    physical,
                    [],
                    riichienv.Conditions(
                        tsumo=tsumo,
                        player_wind=player_wind,
                        round_wind=round_wind,
                    ),
                )
                labels = {yaku.id for yaku in score.yaku_list()}
                candidate = numpy.array(
                    [
                        (
                            bool(score.yakuman)
                            if name == "yakuman"
                            else YAKU_IDS[name] in labels
                        )
                        for name in NAMES
                    ],
                    numpy.int8,
                )
                flags = flags & candidate if require_all else flags | candidate
        return flags
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
            )
            flags |= numpy.maximum(candidate, 0)
        result = result & flags if require_all else result | flags
    return result


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
                score = riichienv.HandEvaluator(
                    list(tiles) + [tile], melds
                ).calc(
                    tile,
                    context["dora_indicators"],
                    riichienv.Conditions(
                        riichi=riichi,
                        tsumo=tsumo,
                        player_wind=context["player_wind"],
                        round_wind=context["round_wind"],
                    ),
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
                values.append(
                    {
                        "points": int(points) if legal else 0,
                        "han": score.han if legal else 0,
                        "fu": score.fu if legal else 0,
                        "legal": bool(legal),
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
        legal = [row for row in rows if row["values"][scenario]["legal"]]
        if legal:
            weights = numpy.array([row["unseen"] for row in legal])
            points = [row["values"][scenario]["points"] for row in legal]
            summary[scenario, :4] = [
                len({shanten.base_id(row["tile"]) for row in legal}),
                weights.sum(),
                min(points),
                max(points),
            ]
            if weights.sum():
                for column, key in ((4, "points"), (5, "han"), (6, "fu")):
                    summary[scenario, column] = numpy.average(
                        [row["values"][scenario][key] for row in legal],
                        weights=weights,
                    )
        summary[scenario, 7] = int(furiten and scenario % 2 == 0)
    return summary, rows
