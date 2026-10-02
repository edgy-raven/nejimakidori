"""Rebuild a queryable training corpus from enriched MJAI archives."""

import concurrent.futures
import contextlib
import dataclasses
import gzip
import hashlib
import json
import multiprocessing
import pathlib
import sqlite3
import time
import zipfile

import numpy

from model import features, records, replay

from . import afk, cache, materialize, player_ranks, scoring, shanten


@dataclasses.dataclass
class Game:
    game_id: str
    date: object
    archive: object
    member: str


def retain_discard_threat(packed, opponent_high_value):
    """Threat and late-wall discards retain either physical-copy origin."""
    return bool(
        packed["riichi_state"][1:].any()
        or opponent_high_value.any()
        or packed["live_wall_count"] <= 16
    )


def retain_focused(observation, participation=None):
    """Keep critical/threat choices; broader yaku/round discards use tedashi."""
    packed = observation.features
    phase = int(packed["decision_phase"])
    distance = (
        min(
            packed[name + "_self_shanten"]
            for name in ("normal", "chiitoitsu", "kokushi")
        )
        - 1
    )
    call_ready = phase == 1 and any(
        numpy.any(
            (packed[kind + "_discard_legal"] > 0)
            & (packed[kind + "_discard_shanten"] == 1)
            & (
                (packed[kind + "_call_tenpai_values"][..., 0, 4] > 3900)
                | (
                    bool(packed["closed_hand"])
                    & (
                        packed[kind + "_call_closed_riichi_values"][..., 0, 4]
                        > 0
                    )
                    & (
                        packed[kind + "_call_closed_riichi_values"][..., 0, 4]
                        < 7700
                    )
                )
            )
        )
        for kind in ("chi", "pon")
    )
    if (
        phase == 3
        or distance <= 0
        or (phase == 2 and int(observation.kan_action) > 0)
        or call_ready
        or (
            phase in (0, 3)
            and numpy.any(
                (packed["discard_action_legal"] > 0)
                & (packed["discard_action_all_shanten"] == 1)
            )
        )
        or (packed["live_wall_count"] <= 16 and distance <= 1)
    ):
        return True
    opponent_pattern = False
    for seat in range(1, 4):
        opened = (packed["meld_valid"][seat] > 0) & (
            packed["meld_type"][seat] != 4
        )
        if opened.sum() >= 2:
            tiles = packed["meld_tile_ids"][seat][opened][
                packed["meld_tile_valid"][seat][opened] > 0
            ]
            opponent_pattern |= len(numpy.unique(tiles[tiles < 27] // 9)) == 1
            opponent_pattern |= bool(
                numpy.all(packed["meld_type"][seat][opened] != 1)
            )
    if phase != 0:
        return bool(observation.hand_yaku_focus or opponent_pattern)
    if retain_discard_threat(packed, observation.opponent_high_value):
        return True
    if int(observation.discard_origin) == 1:
        return False
    legal_yaku = packed["discard_action_yaku_possibility"][
        packed["discard_action_legal"] > 0
    ]
    return bool(
        observation.hand_yaku_focus
        or opponent_pattern
        or numpy.any(
            (legal_yaku.min(axis=0) >= 0)
            & (legal_yaku.max(axis=0) > legal_yaku.min(axis=0))
        )
        or packed["prevailing_wind"] >= 2
        or (packed["prevailing_wind"] == 1 and packed["kyoku_number"] >= 2)
        or floating_dora_tradeoff(packed, participation)
    )


def floating_dora_tradeoff(packed, participation=None):
    """Low-participation bonus retention that costs shanten or ukeire.

    Compare the best efficiency cuts with cuts retaining strictly more bonus.
    Red and ordinary fives share structure but have separate bonus costs.
    Unresolved participation never qualifies as low participation.
    """
    cuts = numpy.flatnonzero(packed["discard_action_legal"])
    bonus = packed["dora_multiplicity"][shanten.BASE_IDS[:37]] + (
        numpy.arange(37) >= 34
    )
    best = min(
        cuts,
        key=lambda cut: (
            packed["discard_action_all_shanten"][cut],
            -packed["discard_action_all_ukeire_count"][cut],
            bonus[cut],
        ),
    )
    if bonus[cuts].min() >= bonus[best]:
        return False
    if participation is None:
        participation = shanten.tenpai_participation(
            packed["candidate_hand_counts"][0],
            packed["known_unavailable_counts"],
            cuts,
        )
    if (participation < 0).any():
        return False
    best_cuts = cuts[
        (
            packed["discard_action_all_shanten"][cuts]
            == packed["discard_action_all_shanten"][best]
        )
        & (
            packed["discard_action_all_ukeire_count"][cuts]
            == packed["discard_action_all_ukeire_count"][best]
        )
        & (bonus[cuts] == bonus[best])
    ]
    return bool(
        numpy.any(
            participation[shanten.BASE_IDS[best_cuts]]
            < records.FOCUSED_SAMPLING["dora_participation_below"]
        )
    )


def archive_games(archives, extra_archives=()):
    """Validate enriched archives and enumerate all eligible game members."""
    import datetime
    import re

    rank_summary = json.loads(
        (pathlib.Path(archives) / "player_ranks.json").read_text()
    )
    if rank_summary["sampling"] != player_ranks.SAMPLING:
        raise ValueError("player rank sampling contract mismatch")
    games = []
    for archive_path in sorted(pathlib.Path(archives).glob("*.zip")):
        with archive_path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != rank_summary["years"][archive_path.stem]["merged_sha256"]:
            raise ValueError(f"rank archive mismatch: {archive_path.name}")
        with zipfile.ZipFile(archive_path) as archive:
            for member in sorted(archive.namelist()):
                match = re.search(r"(20\d{2})(\d{2})(\d{2})", member)
                if match and member.endswith((".mjson", ".jsonl")):
                    date = datetime.date(*map(int, match.groups()))
                    games.append(
                        Game(
                            game_id=pathlib.Path(member).stem,
                            date=date,
                            archive=archive_path,
                            member=member,
                        )
                    )
    if extra_archives:
        rank_summary["supplemental"] = []
        for path in extra_archives:
            extra_games, extra_summary = archive_games(path)
            games.extend(extra_games)
            rank_summary["supplemental"].append(
                {
                    "archives": str(pathlib.Path(path).resolve()),
                    "player_ranks": extra_summary,
                }
            )
    if len({game.game_id for game in games}) != len(games):
        raise ValueError("duplicate game IDs across training archives")
    return games, rank_summary


def import_games(connection, args):
    games, rank_summary = archive_games(args.archives)
    if args.limit_games and len(games) > args.limit_games:
        games = [
            games[index]
            for index in numpy.linspace(
                0, len(games) - 1, args.limit_games, dtype=int
            )
        ]
    if not games:
        raise ValueError("no eligible dated games")
    with contextlib.ExitStack() as stack:
        archives = {
            path: stack.enter_context(zipfile.ZipFile(path))
            for path in {game.archive for game in games}
        }
        for number, game in enumerate(
            sorted(games, key=lambda game: game.game_id)
        ):
            raw = archives[game.archive].read(game.member)
            if raw.startswith(b"\x1f\x8b"):
                raw = gzip.decompress(raw)
            events = [json.loads(line) for line in raw.splitlines()]
            rank = events[0]["player_ranks"]
            game_id = connection.execute(
                "INSERT INTO games (game_id, date, archive, member, shard, "
                "hanchan) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    game.game_id,
                    game.date.isoformat(),
                    str(game.archive),
                    game.member,
                    number // args.games_per_shard,
                    json.dumps(
                        replay.hanchan_targets(events),
                        default=lambda value: value.tolist(),
                    ),
                ),
            ).lastrowid
            connection.executemany(
                "INSERT INTO game_players VALUES (?, ?, ?, ?, ?)",
                (
                    (
                        game_id,
                        actor,
                        rank["dan"][actor] if rank["dan"] else None,
                        rank["rating"][actor] if rank["rating"] else None,
                        int(actor in rank["retained_seats"]),
                    )
                    for actor in range(4)
                ),
            )
            hands = []
            for event in events:
                if event["type"] == "start_kyoku":
                    hands.append([])
                if hands and event["type"] != "end_game":
                    hands[-1].append(event)
            connection.executemany(
                "INSERT INTO hands (game_id, kyoku_id, events) "
                "VALUES (?, ?, ?)",
                (
                    (
                        game_id,
                        f"{hand[0]['bakaze']}{hand[0]['kyoku']}-"
                        f"{hand[0]['honba']}",
                        json.dumps(hand),
                    )
                    for hand in hands
                ),
            )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('player_ranks', ?)",
        (json.dumps(rank_summary),),
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS games_export ON games(shard, game_id)"
    )


def hand_yaku_eligibility(state):
    """Collect retrospective sampling evidence from recorded tenpai states."""
    eligible = numpy.zeros((4, len(scoring.NAMES)), numpy.int8)
    for _, event in state.replay_events():
        if event["type"] not in {
            "start_kyoku",
            "dahai",
            "ankan",
            "kakan",
            "daiminkan",
        }:
            continue
        for actor in (
            range(4) if event["type"] == "start_kyoku" else (event["actor"],)
        ):
            counts = state.concealed_counts(actor)
            if counts.sum() % 3 != 1 or shanten.calculate_shanten(counts) != 0:
                continue
            eligible[actor] = numpy.maximum(
                eligible[actor],
                scoring.completion_flags(
                    tiles=state.legal_observation(actor).hand,
                    melds=state.melds[actor],
                    player_wind=(actor - state.start["oya"]) % 4,
                    round_wind="ESWN".index(state.start["bakaze"]),
                    require_all=False,
                ),
            )
    return eligible


def select_game(connection, game_id, hands):
    """Select decisions directly from cached features and sampling evidence."""
    rows = {table: [] for table in cache.TABLES["selection"]}
    excluded = afk.suspected_afk_kyokus(
        [json.loads(events) for _, events in hands]
    )
    retained = {
        actor
        for actor, keep in connection.execute(
            "SELECT actor, retained FROM game_players WHERE game_id = ?",
            (game_id,),
        )
        if keep
    }
    for number, (hand_id, _) in enumerate(hands):
        rows["hand_selection"].append((hand_id, int(number not in excluded)))
        hand_yaku = {
            actor: int(
                any(
                    value > 0 and name in scoring.FOCUSED_NAMES
                    for name, value in zip(
                        scoring.NAMES,
                        numpy.frombuffer(raw, numpy.int8),
                        strict=True,
                    )
                )
            )
            for actor, raw in connection.execute(
                "SELECT actor, eligible FROM hand_yaku WHERE hand_id = ?",
                (hand_id,),
            )
        }
        rows["hand_focus"].extend(
            (hand_id, actor, value) for actor, value in hand_yaku.items()
        )
        packed = cache.feature_rows(connection, hand_id, cache.FEATURE_TABLES)
        for (
            decision,
            actor,
            labels,
            context,
            participation,
        ) in connection.execute(
            "SELECT decision_index, actor, labels.fields, "
            "label_context.opponent_high_value, "
            "tile_participation.probabilities FROM labels "
            "JOIN label_context USING (hand_id, decision_index, actor) "
            "JOIN tile_participation USING (hand_id, decision_index, actor) "
            "WHERE hand_id = ? ORDER BY decision_index, actor",
            (hand_id,),
        ):
            observation = replay.Observation(
                features=packed[(decision, actor)],
                hand_yaku_focus=hand_yaku[actor],
                opponent_high_value=numpy.frombuffer(context, numpy.int8),
                **cache.decode_fields(labels, records.LABELS, "label"),
            )
            rows["selection"].append(
                (
                    hand_id,
                    decision,
                    actor,
                    int(
                        retain_focused(
                            observation,
                            numpy.frombuffer(participation, numpy.float32),
                        )
                    ),
                    int(number not in excluded and actor in retained),
                )
            )
    return rows


def process_game(task):
    database, game_id, stage = task
    with cache.connect(database) as connection:
        hands = connection.execute(
            "SELECT id, events FROM hands WHERE game_id = ? ORDER BY id",
            (game_id,),
        ).fetchall()
        if stage == "selection":
            return select_game(connection, game_id, hands)
        rows = {table: [] for table in cache.TABLES[stage]}
        game_name, hanchan = connection.execute(
            "SELECT game_id, hanchan FROM games WHERE id = ?", (game_id,)
        ).fetchone()
        hanchan = json.loads(hanchan)
        if stage == "replay":
            tables = ()
        elif stage in {"analysis", "yaku", "participation"}:
            tables = ("state",)
        elif stage == "scoring":
            tables = ("state", "analysis")
        else:
            tables = tuple(cache.FEATURE_TABLES)
        table = "state" if stage == "replay" else stage
        for hand_id, raw_events in hands:
            packed = cache.feature_rows(connection, hand_id, tables)
            if stage == "participation":
                for (decision, actor), values in packed.items():
                    cuts = numpy.flatnonzero(values["discard_action_legal"])
                    probabilities = (
                        shanten.tenpai_participation(
                            values["candidate_hand_counts"][0],
                            values["known_unavailable_counts"],
                            cuts,
                        )
                        if len(cuts)
                        else numpy.zeros(34, numpy.float32)
                    )
                    rows["tile_participation"].append(
                        (
                            hand_id,
                            decision,
                            actor,
                            probabilities.astype(numpy.float32).tobytes(),
                        )
                    )
                continue
            state = replay.Replay(game_name, json.loads(raw_events), hanchan)
            if stage == "labels":

                def cached_features(state, actor, phase, legal_observation):
                    return packed[(state.decision_index, actor)]

                for observation in state.observations(cached_features):
                    key = (
                        hand_id,
                        observation.decision_index,
                        observation.actor,
                    )
                    rows["labels"].append(
                        (
                            *key,
                            cache.encode_fields(
                                vars(observation), records.LABELS, "label"
                            ),
                        )
                    )
                    rows["label_context"].append(
                        (*key, observation.opponent_high_value.tobytes())
                    )
                continue
            if stage == "yaku":
                rows["hand_yaku"].extend(
                    (hand_id, actor, flags.tobytes())
                    for actor, flags in enumerate(hand_yaku_eligibility(state))
                )
                state = replay.Replay(
                    game_name, json.loads(raw_events), hanchan
                )
            for event, actor, phase, _, _ in state.policy_decisions():
                values = features.pack(
                    state=state,
                    actor=actor,
                    phase=phase,
                    legal_observation=state.legal_observation(actor),
                    stage=table,
                    packed=(
                        None
                        if stage == "replay"
                        else packed[(state.decision_index, actor)]
                    ),
                )
                key = (hand_id, state.decision_index, actor)
                if stage == "replay":
                    rows["decisions"].append(
                        (*key, event, features.PHASES[phase])
                    )
                rows[table].append(
                    (
                        *key,
                        cache.encode_fields(
                            values, cache.FEATURE_TABLES[table], "feature"
                        ),
                    )
                )
    return rows


def run_stage(database, stage, args, executor):
    started = time.monotonic()
    with contextlib.closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        cache.require(connection, cache.DEPENDENCIES[stage])
        with connection:
            cache.invalidate(connection, stage)
            if stage == "import":
                import_games(connection, args)
            else:
                games = connection.execute("SELECT id FROM games ORDER BY id")
                # Workers read the committed prerequisites. Only this process
                # writes; completed stages become visible together at commit.
                while batch := games.fetchmany(args.workers * 2):
                    tasks = (
                        (str(database), game_id, stage) for (game_id,) in batch
                    )
                    results = (
                        map(process_game, tasks)
                        if executor is None
                        else executor.map(process_game, tasks)
                    )
                    for rows in results:
                        for table, values in rows.items():
                            if values:
                                connection.executemany(
                                    f'INSERT INTO "{table}" VALUES ('
                                    + ",".join("?" for _ in values[0])
                                    + ")",
                                    values,
                                )
            connection.execute("INSERT INTO stages VALUES (?)", (stage,))
    print(
        f"cache stage={stage} elapsed_seconds={time.monotonic() - started:.3f}",
        flush=True,
    )


def run(args):
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    database = root / "decisions.sqlite"
    if args.stage in {"all", "import"}:
        if not args.archives:
            raise ValueError("--archives is required for import")
        if not database.exists():
            with contextlib.closing(sqlite3.connect(database)) as connection:
                cache.create(connection)
    stages = tuple(cache.DEPENDENCIES) if args.stage == "all" else (args.stage,)
    with (
        concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        if args.workers > 1
        else contextlib.nullcontext()
    ) as executor:
        for stage in stages:
            run_stage(database, stage, args, executor)
            (root / "dataset_summary.json").unlink(missing_ok=True)
    if args.stage in {"all", "selection"}:
        with cache.connect(database) as connection:
            (root / "player_rank_selection.json").write_text(
                connection.execute(
                    "SELECT value FROM metadata WHERE name = 'player_ranks'"
                ).fetchone()[0]
            )
        if args.stage == "all":
            materialize.write(database, root, workers=args.workers)
