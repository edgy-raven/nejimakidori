"""Stable hand payments, score conservation and initial-dealer tie priority."""

import numpy

PAYMENT_TYPES = ("ron", "tsumo", "draw", "bank")
SAINT3_JADE_SOUTH_OFFSETS = (125, 60, -5, -255)
PAYMENT_CELLS = numpy.array(
    [
        [
            [
                payer == recipient if kind == "bank" else payer != recipient
                for recipient in range(4)
            ]
            for payer in range(4)
        ]
        for kind in PAYMENT_TYPES
    ]
)


# Scoring states are hand values, never payer-specific point amounts.
# 20/25 fu require at least two han; sub-mangan hands stop at 110 fu.
# Fu-capped mangan (3 han 70+ fu, 4 han 40+ fu) shares the limit tier.
# Limit hands use representative han: mangan, haneman, baiman, sanbaiman,
# kazoe. Named yakuman count once; at most four distinct families can stack
# in the native four-player rules (double-value variants are normalized).
SCORE_STATES = (
    tuple(
        (han, fu)
        for han in range(1, 5)
        for fu in (20, 25, *range(30, 111, 10))
        if (han >= 2 or fu >= 30) and fu * 2 ** (han + 2) < 2000
    )
    + tuple((han, 0) for han in (5, 6, 8, 11, 13))
    + tuple((0, count) for count in range(1, 5))
)


def basic_points(han, fu):
    if han >= 13:
        return 8000
    if han >= 11:
        return 6000
    if han >= 8:
        return 4000
    if han >= 6:
        return 3000
    if han >= 5:
        return 2000
    return min(2000, fu * 2 ** (han + 2))


# Derived monetary alphabet, including exhaustive draws and pao shares.
# Bank awards also have four public-state atoms; records store raw amounts.
PAYMENT_VALUES = numpy.array(
    sorted(
        {-1000, 0, 750, 1000, 1500, 3000}
        | {
            4000 * count * factor
            for count in range(1, 5)
            for factor in (1, 2, 3, 4, 6, 8, 12)
        }
        | {
            100
            * (
                (
                    factor * (8000 * fu if han == 0 else basic_points(han, fu))
                    + 99
                )
                // 100
            )
            for han, fu in SCORE_STATES
            for factor in (1, 2, 4, 6)
        }
    ),
    dtype=numpy.int32,
)


def saint3_jade_south(scores, ranks):
    """Final ST3 return, with one-based ranks and scores in table points."""
    return numpy.ceil(
        (numpy.asarray(scores) - 25000) / 1000
        + numpy.asarray(SAINT3_JADE_SOUTH_OFFSETS)[numpy.asarray(ranks) - 1]
    ).astype(numpy.int32)


def score_ranks(scores, first_dealer=0):
    order = sorted(
        range(4), key=lambda seat: (-scores[seat], (seat - first_dealer) % 4)
    )
    return tuple(order.index(seat) + 1 for seat in range(4))


def normalize_draw_payments(events):
    """Remove native draw deltas' already-paid riichi deposits.

    Nagashi and zero-transfer exhaustive draws can retain the per-seat
    deposit delta. MJAI reach_accepted already accounts for that payment.
    Accept only the observed, conservation-checked native defect.
    """
    for event in events:
        if event["type"] == "start_kyoku":
            deposits = [0] * 4
        elif event["type"] == "reach_accepted":
            deposits[event["actor"]] += 1000
        elif event["type"] == "ryukyoku" and sum(event["deltas"]):
            corrected = [
                delta + deposit
                for delta, deposit in zip(event["deltas"], deposits)
            ]
            if sum(corrected) or not (
                event["reason"] == "nagashimangan"
                or (
                    event["reason"] == "exhaustive_draw"
                    and corrected == [0] * 4
                )
            ):
                raise ValueError(f"Invalid native draw payment: {event}")
            event["deltas"] = corrected


def terminal_payments(events):
    """Typed ron/tsumo/draw/bank transfers, without honba, including pao."""
    result = numpy.zeros((len(PAYMENT_TYPES), 4, 4), dtype=numpy.int32)
    honba = 0
    for event in events:
        if event["type"] == "start_kyoku":
            honba = event.get("honba", 0)
        elif event["type"] == "reach_accepted":
            result[3, event["actor"], event["actor"]] -= 1000
        elif event["type"] == "hora":
            winner = event["actor"]
            delta = event["deltas"]
            payers = [seat for seat in range(4) if delta[seat] < 0]
            for payer in payers:
                if event["target"] == winner:
                    # Tenhou pao tsumo charges one seat all three shares.
                    bonus = 300 * honba // len(payers)
                else:
                    # Split pao ron charges honba only to the liable seat,
                    # which is the payer other than the discarder.
                    bonus = (
                        300
                        * honba
                        * (len(payers) == 1 or payer != event["target"])
                    )
                result[int(event["target"] == winner), payer, winner] += (
                    -delta[payer] - bonus
                )
            result[3, winner, winner] += sum(delta)
            honba = 0
        elif event["type"] == "ryukyoku":
            delta = numpy.array(event["deltas"])
            losers = numpy.flatnonzero(delta < 0)
            winners = numpy.flatnonzero(delta > 0)
            for loser in losers:
                for winner in winners:
                    result[2, loser, winner] += -delta[loser] // len(winners)
    return result


def settlement(payments):
    payments = numpy.asarray(payments).sum(axis=0)
    diagonal = numpy.diag(payments)
    return numpy.r_[
        payments.sum(0) - payments.sum(1) + diagonal, -diagonal.sum()
    ]


def relative_payments(payments, actor, accepted_riichi):
    order = [(actor + offset) % 4 for offset in range(4)]
    result = numpy.asarray(payments)[:, order][:, :, order].copy()
    result[3] += numpy.diag(numpy.asarray(accepted_riichi)[order]) * 1000
    return result


def terminal_scores(events):
    return (
        numpy.asarray(events[0]["scores"])
        + sum(
            (
                numpy.asarray(event["deltas"])
                for event in events
                if "deltas" in event
            ),
            numpy.zeros(4, dtype=int),
        )
        - 1000
        * numpy.bincount(
            [
                event["actor"]
                for event in events
                if event["type"] == "reach_accepted"
            ],
            minlength=4,
        )
    )


def hand_continuation_states(events):
    """One observed settlement per hand and actor in a completed game.

    Columns: four scores, four zero-based ranks, next relative dealer,
    next zero-based hand number, terminal flag, outstanding deposit points.
    Future hands supply only the immediately following round/dealer identity.
    """
    assert events[-1]["type"] == "end_game"
    starts = [
        i for i, event in enumerate(events) if event["type"] == "start_kyoku"
    ]
    first_dealer = (
        events[starts[0]]["oya"] - events[starts[0]]["kyoku"] + 1
    ) % 4
    states = []
    for hand, start in enumerate(starts):
        terminal = hand + 1 == len(starts)
        end = len(events) if terminal else starts[hand + 1]
        scores = terminal_scores(events[start:end])
        ranks = numpy.asarray(score_ranks(scores, first_dealer)) - 1
        following = events[start] if terminal else events[end]
        next_hand = (
            4 * "ESWN".index(following["bakaze"]) + following["kyoku"] - 1
        )
        pot = (
            sum(events[start]["scores"])
            + 1000 * events[start]["kyotaku"]
            - scores.sum()
        )
        states.append(
            [
                numpy.r_[
                    scores[(seat + numpy.arange(4)) % 4],
                    ranks[(seat + numpy.arange(4)) % 4],
                    (following["oya"] - seat) % 4,
                    next_hand,
                    int(terminal),
                    pot,
                ]
                for seat in range(4)
            ]
        )
    return numpy.asarray(states, numpy.int32)


def payment_classes(payments):
    values = numpy.asarray(payments)
    matches = values[..., None] == PAYMENT_VALUES
    if not matches.any(-1).all():
        raise ValueError(
            "payment is outside the declared classes: "
            f"{numpy.unique(values[~matches.any(-1)]).tolist()}"
        )
    return matches.argmax(-1)
