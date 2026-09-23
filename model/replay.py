"""Incremental MJAI state and ordered decision-time training observations."""

import dataclasses
import inspect
import json

import numpy
import riichienv

from log_dataset import shanten

from . import actions, outcomes, records

CALL_TYPES = {"chi", "pon", "daiminkan"}
OBSERVATION_FIELDS = tuple(
    name
    for name in inspect.signature(riichienv.Observation).parameters
    if name not in {"legal_actions", "hands", "drawn_tile"}
)

Observation = dataclasses.make_dataclass(
    "Observation",
    [
        (name, object, dataclasses.field(default=None))
        for name in (
            "game_id",
            "kyoku_id",
            "decision_index",
            "actor",
            "features",
            "opponent_high_value",
            "hand_yaku_focus",
            *records.LABELS,
        )
    ],
)


def hanchan_targets(events):
    starts = [event for event in events if event["type"] == "start_kyoku"]
    scores = outcomes.terminal_scores(events[events.index(starts[-1]) :])
    first_dealer = (starts[0]["oya"] - (starts[0]["kyoku"] - 1)) % 4
    return {
        "ranks": outcomes.score_ranks(scores, first_dealer),
        "first_dealer": first_dealer,
    }


def recorded_win_choices(state, index, next_index):
    event = state.events[index]
    if event["type"] not in {"tsumo", "dahai", "kakan", "ankan"}:
        return {}
    while next_index < len(state.events) and state.events[next_index][
        "type"
    ] in {"dora", "reach_accepted"}:
        next_index += 1
    if next_index == len(state.events):
        return {}
    winners = {}
    end = next_index
    while end < len(state.events) and state.events[end]["type"] == "hora":
        winners[state.events[end]["actor"]] = end
        end += 1
    if not winners and state.events[next_index]["type"] not in {
        "tsumo",
        "dahai",
        "reach",
        "chi",
        "pon",
        "daiminkan",
        "ankan",
        "kakan",
    }:
        return {}
    choices = {}
    for actor in ((event["actor"],) if event["type"] == "tsumo" else range(4)):
        if not any(
            action.action_type in actions.WIN_ACTIONS
            for action in state.legal_observation(actor).legal_actions()
        ):
            continue
        if actor in winners:
            choices[actor] = (1, winners[actor])
        elif not winners:
            choices[actor] = (0, index)
    return choices


class Replay:
    def __init__(self, game_id, events, hanchan=None):
        self.game_id = game_id
        self.events = events
        self.hanchan = hanchan
        self.native_env = riichienv.RiichiEnv(
            rule=riichienv.GameRule.default_tenhou()
        )
        self.history = []
        self.event_index = -1
        self.decision_index = 0
        self.pending_discard = None

    @classmethod
    def from_start(cls, game_id, event):
        state = cls(game_id, [])
        state.apply(event)
        return state

    @property
    def melds(self):
        return self.native_env.melds

    @property
    def accepted_riichi(self):
        return self.accepted

    def public_tiles(self):
        return (
            list(self.dora_markers)
            + [row["pai"] for river in self.rivers for row in river]
            + [
                tile
                for melds in self.meld_events
                for meld in melds
                for tile in meld["consumed"]
            ]
            + [
                meld["upgrade_tile"]
                for melds in self.meld_events
                for meld in melds
                if meld["type"] == "kakan"
            ]
        )

    def concealed_counts(self, actor):
        return numpy.bincount(
            [tile // 4 for tile in self.native_env.hands[actor] if tile < 136],
            minlength=34,
        )

    def missed_agari(self, actor):
        if self.native_env.missed_agari_doujun[actor]:
            return True
        waits = riichienv.HandEvaluator(
            self.native_env.hands[actor], self.melds[actor]
        ).get_waits()
        for index in range(len(self.history) - 1, -1, -1):
            event = self.history[index]
            if event.get("actor") == actor and (
                event["type"] == "reach_accepted"
                or (
                    not self.accepted[actor]
                    and event["type"]
                    in {"tsumo", "chi", "pon", "daiminkan", "ankan", "kakan"}
                )
            ):
                break
            if (
                event["type"] == "dahai"
                and event["actor"] != actor
                and (
                    self.pending_discard is None
                    or index != self.pending_discard["event_index"]
                )
                and shanten.base_id(event["pai"]) in waits
            ):
                return True
        return False

    def legal_observation(self, actor):
        from . import features

        original = self.native_env.get_observation(actor)
        hand = []
        for tile in original.hand:
            physical = tile // 4 * 4
            if tile not in (16, 52, 88) and physical in (16, 52, 88):
                physical += 1
            while physical in hand:
                physical += 1
            hand.append(physical)
        drawn = None
        if self.draws[actor] is not None and self.draws[actor] != "?":
            code = shanten.mjai_tile_code(self.draws[actor])
            drawn = next(
                tile
                for tile in reversed(hand)
                if features.native_tile_code(tile) == code
            )
        legal = []
        for action in original.legal_actions():
            if action.action_type == riichienv.ActionType.DISCARD:
                for tile in hand:
                    if features.native_tile_code(
                        tile
                    ) == features.native_tile_code(action.tile):
                        if self.accepted[actor] and tile != drawn:
                            continue
                        if not any(
                            item.action_type == action.action_type
                            and item.tile == tile
                            for item in legal
                        ):
                            legal.append(
                                riichienv.Action(
                                    type=action.action_type,
                                    tile=tile,
                                    actor=actor,
                                )
                            )
            else:
                legal.append(action)
        values = {name: getattr(original, name) for name in OBSERVATION_FIELDS}
        values.update(
            hands=[
                hand if seat == actor else original.hands[seat]
                for seat in range(4)
            ],
            legal_actions=legal,
            drawn_tile=drawn,
        )
        values["events"] = [json.dumps(event) for event in original.events]
        return riichienv.Observation(**values)

    def apply(self, event):
        self.event_index += 1
        self.history.append(event)
        kind = event["type"]
        if kind == "start_kyoku":
            self.start = event
            self.kyoku_id = (
                f"{event['bakaze']}{event['kyoku']}-{event['honba']}"
            )
            self.history = [event]
            self.rivers = [[] for _ in range(4)]
            self.meld_events = [[] for _ in range(4)]
            self.declared = [False] * 4
            self.accepted = [False] * 4
            self.ippatsu = [False] * 4
            self.draws = [None] * 4
            self.live_wall = 70
            self.pending_discard = None
            self.last_origin = [0] * 4
            self.rinshan_actor = None
            self.dora_markers = [event["dora_marker"]]
        elif kind == "tsumo":
            self.live_wall -= 1
            self.draws[event["actor"]] = event["pai"]
            self.last_origin[event["actor"]] = 1
            if self.pending_discard:
                self.pending_discard["resolution"] = 1
            self.pending_discard = None
        elif kind == "dahai":
            actor = event["actor"]
            if self.pending_discard:
                self.pending_discard["resolution"] = 1
            record = {
                **event,
                "event_index": len(self.history) - 1,
                "live_wall": self.live_wall,
                "own_turns": [len(river) for river in self.rivers],
                "origin": self.last_origin[actor],
                "riichi": self.declared[actor] and not self.accepted[actor],
                "resolution": 0,
                "caller": None,
                "dora_markers": list(self.dora_markers),
            }
            self.rivers[actor].append(record)
            self.pending_discard = record
            self.draws[actor] = None
            self.rinshan_actor = None
            if self.accepted[actor]:
                self.ippatsu[actor] = False
        elif kind == "reach":
            self.declared[event["actor"]] = True
        elif kind == "reach_accepted":
            self.accepted[event["actor"]] = True
            self.ippatsu[event["actor"]] = True
        elif kind in CALL_TYPES | {"ankan", "kakan"}:
            self.ippatsu = [False] * 4
            if self.pending_discard and kind in CALL_TYPES:
                self.pending_discard["resolution"] = {
                    "chi": 2,
                    "pon": 3,
                    "daiminkan": 4,
                }[kind]
                self.pending_discard["caller"] = event["actor"]
            self.pending_discard = None
            self.draws[event["actor"]] = None
            self.last_origin[event["actor"]] = kind
            if kind in {"ankan", "kakan", "daiminkan"}:
                self.rinshan_actor = event["actor"]
            if kind == "kakan":
                for meld in self.meld_events[event["actor"]]:
                    if meld["type"] == "pon" and shanten.base_id(
                        meld["pai"]
                    ) == shanten.base_id(event["pai"]):
                        meld["upgrade_index"] = len(self.history) - 1
                        meld["upgrade_tile"] = event["pai"]
                        meld["type"] = "kakan"
            else:
                self.meld_events[event["actor"]].append(
                    {**event, "event_index": len(self.history) - 1}
                )
        elif kind == "dora":
            self.dora_markers.append(event["dora_marker"])
        self.native_env.apply_event(event)

    def replay_events(self):
        index = 0
        while index < len(self.events):
            event = self.events[index]
            self.apply(event)
            if event["type"] == "tsumo":
                while self.events[index + 1]["type"] == "dora":
                    index += 1
                    self.apply(self.events[index])
            self.event_index = index
            yield index, event
            index += 1

    def decisions(self):
        from . import features

        for index, event in self.replay_events():
            requests = []
            next_index = index + 1
            while (
                next_index < len(self.events)
                and self.events[next_index]["type"] == "reach_accepted"
            ):
                next_index += 1
            win_choices = (
                recorded_win_choices(self, index, next_index)
                if next_index < len(self.events)
                else {}
            )
            requests.extend((actor, "WIN") for actor in win_choices)
            if event["type"] in {
                "tsumo",
                "chi",
                "pon",
                "reach",
            } and self.events[next_index]["type"] not in {"hora", "ryukyoku"}:
                actor = event["actor"]
                legal = self.legal_observation(actor).legal_actions()
                types = {action.action_type for action in legal}
                if (
                    riichienv.ActionType.ANKAN in types
                    or riichienv.ActionType.KAKAN in types
                ):
                    requests.append((actor, "KAN"))
                    if self.events[next_index]["type"] in {"ankan", "kakan"}:
                        types = set()
                if riichienv.ActionType.RIICHI in types:
                    requests.append((actor, "RIICHI"))
                    if self.events[next_index]["type"] != "reach":
                        requests.append((actor, "DISCARD"))
                elif riichienv.ActionType.DISCARD in types:
                    requests.append((actor, "DISCARD"))
            elif (
                event["type"] == "dahai"
                and self.events[next_index]["type"] != "hora"
            ):
                # Other seats' intentions are censored by an accepted
                # call; a losing claim must not be labeled as a pass.
                for actor in (
                    (self.events[next_index]["actor"],)
                    if self.events[next_index]["type"] in CALL_TYPES
                    else range(4)
                ):
                    legal = self.legal_observation(actor).legal_actions()
                    if any(
                        action.action_type
                        in {
                            riichienv.ActionType.CHI,
                            riichienv.ActionType.PON,
                            riichienv.ActionType.DAIMINKAN,
                        }
                        for action in legal
                    ):
                        requests.append((actor, "RESPONSE"))
            for actor, phase in requests:
                next_event = self.events[next_index]
                if (
                    phase == "RESPONSE"
                    and next_event["type"] in {"chi", "pon"}
                    and next_event["actor"] == actor
                    and not features.preferred_recorded_call(
                        next_event["type"],
                        next_event["consumed"],
                        self.legal_observation(actor).legal_actions(),
                    )
                ):
                    continue
                yield index, actor, phase, next_index, win_choices
                self.decision_index += 1

    def observations(self):
        from . import actions, features, training_targets

        self.targets = training_targets.round_targets(
            Replay.from_start(self.game_id, self.events[0]), self.events
        )
        for index, actor, phase, next_index, win_choices in self.decisions():
            legal_observation = self.legal_observation(actor)
            if (
                actions.forced_policy_action(
                    self, actor, legal_observation.legal_actions()
                )
                is not None
            ):
                continue
            if (
                phase == "DISCARD"
                and features.single_legal_action(
                    [
                        action
                        for action in legal_observation.legal_actions()
                        if action.action_type == riichienv.ActionType.DISCARD
                    ],
                    legal_observation.drawn_tile,
                )
                is not None
            ):
                continue
            packed = features.pack(self, actor, phase, legal_observation)
            observation = Observation(
                game_id=self.game_id,
                kyoku_id=self.kyoku_id,
                decision_index=self.decision_index,
                actor=actor,
                features=packed,
            )
            for name, definition in records.LABELS.items():
                setattr(
                    observation,
                    name,
                    numpy.full(definition["shape"], -1, definition["dtype"]),
                )
            observation.opponent_high_value = numpy.zeros(3, numpy.int8)
            next_event = self.events[next_index]
            action_index = index
            if phase == "DISCARD" and next_event["type"] == "dahai":
                action_index = next_index
                if (
                    not self.accepted[actor]
                    and features.discard_legal(packed).sum() > 1
                ):
                    observation.discard_action = (
                        37
                        if next_event["tsumogiri"]
                        else shanten.mjai_tile_code(next_event["pai"])
                    )
            elif phase == "RIICHI":
                declaration = next_event["type"] == "reach"
                action_index = next_index + int(declaration)
                cut = self.events[next_index + 1] if declaration else next_event
                observation.riichi_action = int(declaration)
                observation.continuation_discard = (
                    37
                    if cut["tsumogiri"]
                    else shanten.mjai_tile_code(cut["pai"])
                )
            elif phase == "RESPONSE":
                observation.response_action = 0
                if (
                    next_event["type"] in CALL_TYPES
                    and next_event["actor"] == actor
                ):
                    action_index = next_index + int(
                        next_event["type"] != "daiminkan"
                    )
                    observation.response_action = {
                        "chi": 1,
                        "pon": 2,
                        "daiminkan": 3,
                    }[next_event["type"]]
                    if next_event["type"] == "chi":
                        observation.chi_pattern = features.chi_pattern(
                            next_event["pai"], next_event["consumed"]
                        )
                    if next_event["type"] != "daiminkan":
                        observation.continuation_discard = (
                            shanten.mjai_tile_code(
                                self.events[next_index + 1]["pai"]
                            )
                        )
            elif phase == "KAN":
                observation.kan_action = 0
                if next_event["type"] in {"ankan", "kakan"}:
                    action_index = next_index
                    observation.kan_action = 1 + shanten.base_id(
                        next_event["consumed"][0]
                        if next_event["type"] == "ankan"
                        else next_event["pai"]
                    )
            training_targets.attach(observation, self, index, action_index)
            for name, value in training_targets.ron_labels(
                self, actor, phase
            ).items():
                setattr(observation, name, value)
            yield observation
