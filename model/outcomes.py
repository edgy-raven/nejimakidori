"""Stable hand payments, score conservation and initial-dealer tie priority."""

import numpy

PAYMENT_TYPES = ("ron", "tsumo", "draw", "bank")
PAYMENT_VALUES = numpy.array(
    [
        -1000,
        0,
        300,
        400,
        500,
        600,
        700,
        750,
        800,
        1000,
        1200,
        1300,
        1500,
        1600,
        2000,
        2300,
        2400,
        2600,
        2900,
        3000,
        3200,
        3400,
        3900,
        4000,
        4500,
        4800,
        5000,
        5200,
        5800,
        6000,
        6400,
        6800,
        7000,
        7700,
        8000,
        8700,
        9000,
        9600,
        11600,
        12000,
        14800,
        15700,
        16000,
        18000,
        24000,
        32000,
        36000,
        48000,
        64000,
        96000,
    ]
)
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
NUM_OUTCOMES = len(PAYMENT_VALUES)
DEALERSHIP_TARGET = "root_remaining_points_saint3_future_dealerships"


def dealerships_remaining(start, actor):
    hand = 4 * "ESWN".index(start["bakaze"]) + start["kyoku"] - 1
    return sum(
        (start["oya"] + future - hand) % 4 == actor
        for future in range(hand + 1, 8)
    )


def dealership_utility(hand_points, rank, dealerships):
    if dealerships >= 2:
        return hand_points / 10000
    placement = 2 * ((125, 60, -5, -255)[rank - 1] + 255) / 380 - 1
    return (
        0.5 * (hand_points / 10000 + placement)
        if dealerships == 1
        else placement
    )


def score_ranks(scores, first_dealer=0):
    order = sorted(
        range(4), key=lambda seat: (-scores[seat], (seat - first_dealer) % 4)
    )
    return tuple(order.index(seat) + 1 for seat in range(4))


def hand_outcomes(events):
    """Win receipts, ron deal-in incidence and ron loss, in 10,000 points.

    Draw transfers, tsumo payments and riichi deposits are not deal-ins.
    Multiple ron winners contribute to the same losing player's total loss.
    """
    result = numpy.zeros((4, 3), numpy.float32)
    for event in events:
        if event["type"] != "hora":
            continue
        winner, payer = event["actor"], event["target"]
        result[winner, 0] += event["deltas"][winner] / 10000
        if payer != winner:
            result[payer, 1] = 1
            result[payer, 2] -= event["deltas"][payer] / 10000
    return result


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


def payment_classes(payments):
    values = numpy.asarray(payments)
    matches = values[..., None] == PAYMENT_VALUES
    if not matches.any(-1).all():
        raise ValueError(
            "payment is outside the declared classes: "
            f"{numpy.unique(values[~matches.any(-1)]).tolist()}"
        )
    return matches.argmax(-1)


def expected_payments(probabilities):
    return numpy.asarray(probabilities) @ PAYMENT_VALUES
