"""Decision-time scored waits using external RiichiEnv hand evaluation."""

import numpy
import riichienv

from log_dataset import shanten


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
