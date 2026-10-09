"""Same-hand scoring scenarios; hidden bonuses are targets, never inputs."""

import itertools
import types

import numpy
import riichienv
import riichienv._riichienv

from log_dataset import shanten
from model import outcomes, payment_scoring

MODES = ("open", "dama", "riichi", "double_riichi")
SPECIALS = ("ordinary", "last_tile", "rinshan", "chankan", "first_turn")
# Ura 5 is a distributional tail bucket, not an assumption of exactly 5 han.
SCENARIOS = tuple(
    (mode, ura, ippatsu, special)
    for mode, special in itertools.product(
        range(len(MODES)), range(len(SPECIALS))
    )
    for ura, ippatsu in (
        itertools.product(range(6), range(2)) if mode >= 2 else [(0, 0)]
    )
)
# Scenarios with equal added han share one score decoder and target class.
SCORE_GROUPS = numpy.unique(
    [
        (
            mode == 0,
            (2 if mode == 3 else int(mode == 2))
            + ura
            + ippatsu
            + int(special in (1, 2, 3)),
            special == 4,
            ura == 5,
        )
        for mode, ura, ippatsu, special in SCENARIOS
    ],
    axis=0,
    return_inverse=True,
)[1].astype(numpy.int32)
SCORE_SCENARIOS = tuple(
    SCENARIOS[numpy.flatnonzero(SCORE_GROUPS == group)[0]]
    for group in range(SCORE_GROUPS.max() + 1)
)

CELLS = tuple(map(tuple, numpy.argwhere(outcomes.PAYMENT_CELLS)))


def score_hand(ctx, conditions, ura):
    """Rust scoring with Tenhou's single-value named yakuman rules.

    Call the core directly: the Python evaluator counts kan tiles when
    deciding whether to append the winning tile, corrupting complete hands.
    Replay already supplies the complete concealed hand and separate melds.
    """
    score = riichienv._riichienv.HandEvaluator(ctx.tiles, ctx.melds).calc(
        ctx.agari_tile,
        ctx.dora_indicators,
        ura if conditions.riichi or conditions.double_riichi else [],
        riichienv._riichienv.Conditions(
            tsumo=conditions.tsumo,
            riichi=conditions.riichi,
            double_riichi=conditions.double_riichi,
            ippatsu=conditions.ippatsu,
            haitei=conditions.haitei,
            houtei=conditions.houtei,
            rinshan=conditions.rinshan,
            chankan=conditions.chankan,
            tsumo_first_turn=conditions.tsumo_first_turn,
            player_wind=conditions.player_wind,
            round_wind=(
                riichienv.Wind.East,
                riichienv.Wind.South,
                riichienv.Wind.West,
                riichienv.Wind.North,
            )[conditions.round_wind],
        ),
    )
    if not score.yakuman:
        return score
    # Named double yakuman count once; distinct yakuman still stack.
    han = 13 * len(score.yaku_list())
    payments = riichienv.calculate_score(
        han=han,
        fu=0,
        is_oya=conditions.player_wind == riichienv.Wind.East,
        is_tsumo=conditions.tsumo,
        honba=0,
        num_players=4,
    )
    return types.SimpleNamespace(
        is_win=score.is_win,
        han=han,
        fu=0,
        yakuman=True,
        yaku_list=score.yaku_list,
        ron_agari=payments.pay_ron,
        tsumo_agari_ko=payments.pay_tsumo_ko,
        tsumo_agari_oya=payments.pay_tsumo_oya,
    )


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
            ctx.actual = score_hand(ctx, ctx.conditions, ctx.ura_indicators)
            if not ctx.actual.is_win:
                raise ValueError((state.kyoku_id, event, "Unscorable win"))
            yield state.kyoku_id, ctx
        state.apply(event)


def scored_amounts(score, extra_han, dealer, tsumo):
    """Ron, nondealer tsumo share, dealer tsumo share; honba excluded."""
    if not score.is_win:
        return None
    if not extra_han or score.yakuman:
        return numpy.array(
            [score.ron_agari, score.tsumo_agari_ko, score.tsumo_agari_oya],
            numpy.int32,
        )
    base = outcomes.basic_points(score.han + extra_han, score.fu)
    return numpy.array(
        [
            0 if tsumo else 100 * ((base * (6 if dealer else 4) + 99) // 100),
            100 * ((base * (2 if dealer else 1) + 99) // 100) if tsumo else 0,
            100 * ((base * 2 + 99) // 100) if tsumo and not dealer else 0,
        ],
        numpy.int32,
    )


def winning_scenarios(ctx):
    """Rescore one completed hand, preserving its tiles and meld structure.

    Returns amounts, observed scenario, and joint score details. Tail
    supervision requires observed hidden indicators revealing >=5 ura han.
    """
    scores = {}
    closed = all(not meld.opened for meld in ctx.melds)
    conditions = ctx.conditions
    observed_mode = (
        3
        if conditions.double_riichi
        else 2 if conditions.riichi else int(closed)
    )
    specials = (
        conditions.haitei or conditions.houtei,
        conditions.rinshan,
        conditions.chankan,
        conditions.tsumo_first_turn,
    )
    if sum(specials) > 1:
        raise ValueError("Overlapping special win conditions")
    special = next((i + 1 for i, flag in enumerate(specials) if flag), 0)
    # Count physical bonus tiles, including concealed kan, rather than
    # subtracting capped han totals or counting the hidden indicators.
    counts = numpy.bincount(
        [tile // 4 for tile in ctx.tiles]
        + [tile // 4 for meld in ctx.melds for tile in meld.tiles],
        minlength=34,
    )
    ura = 0
    if observed_mode >= 2:
        for indicator in ctx.ura_indicators:
            tile = indicator // 4
            successor = (
                tile // 9 * 9 + (tile + 1) % 9
                if tile < 27
                else 27 + (tile - 26) % 4 if tile < 31 else 31 + (tile - 30) % 3
            )
            ura += int(counts[successor])
    observed = SCENARIOS.index(
        (observed_mode, min(ura, 5), int(conditions.ippatsu), special)
    )
    result = numpy.full((len(SCENARIOS), 3), -1, numpy.int32)
    details = numpy.full((len(SCENARIOS), 3), -1, numpy.int32)
    for scenario, (mode, bonus, ippatsu, extra) in enumerate(SCENARIOS):
        if (mode == 0) == closed or (bonus == 5 and ura < 5):
            continue
        if (
            (extra == 2 and not conditions.tsumo)
            or (extra == 3 and conditions.tsumo)
            or (extra == 4 and (not conditions.tsumo or mode != 1 or ctx.melds))
        ):
            continue
        if (mode, ippatsu, extra) not in scores:
            scores[mode, ippatsu, extra] = score_hand(
                ctx,
                riichienv.Conditions(
                    tsumo=conditions.tsumo,
                    riichi=mode >= 2,
                    double_riichi=mode == 3,
                    ippatsu=bool(ippatsu),
                    haitei=extra == 1 and conditions.tsumo,
                    houtei=extra == 1 and not conditions.tsumo,
                    rinshan=extra == 2,
                    chankan=extra == 3,
                    tsumo_first_turn=extra == 4,
                    player_wind=conditions.player_wind,
                    round_wind=conditions.round_wind,
                ),
                [],
            )
        amounts = scored_amounts(
            scores[mode, ippatsu, extra],
            ura if bonus == 5 else bonus,
            conditions.player_wind == riichienv.Wind.East,
            conditions.tsumo,
        )
        if amounts is not None:
            result[scenario] = amounts
            details[scenario] = (
                scores[mode, ippatsu, extra].han
                + (
                    0
                    if scores[mode, ippatsu, extra].yakuman
                    else ura if bonus == 5 else bonus
                ),
                scores[mode, ippatsu, extra].fu,
                int(scores[mode, ippatsu, extra].yakuman),
            )
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
            "scenario": numpy.full((2, 4, 4, len(SCENARIOS)), -1, numpy.int16),
            "score": numpy.full((2, 4, 4, len(SCENARIOS)), -1, numpy.int16),
            "payment": outcomes.terminal_payments(rows),
        }
        for key, rows in rounds.items()
    }
    for key, ctx in win_contexts(events):
        amounts, observed, details = winning_scenarios(ctx)
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
            classes = outcomes.payment_classes(shares[valid])
            result[key]["scenario"][kind, payer, ctx.seat, valid] = classes
            result[key]["score"][kind, payer, ctx.seat, valid] = [
                payment_scoring.score_class(int(han), int(fu), bool(yakuman))
                for han, fu, yakuman in details[valid]
            ]
            result[key]["bonus"][kind, payer, ctx.seat] = observed
            assert shares[observed] == payments[payer]
    return result
