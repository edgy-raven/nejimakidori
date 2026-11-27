"""Validate and enumerate the rank-enriched training archives."""

import dataclasses
import functools
import hashlib
import json
import pathlib
import re
import zipfile

from . import player_ranks


@dataclasses.dataclass
class Game:
    game_id: str
    archive: pathlib.Path
    member: str


@functools.lru_cache(maxsize=4)
def open_archive(path):
    """Worker-owned readers for immutable inputs; eviction closes readers."""
    return zipfile.ZipFile(path)


def games(archives, extra_archives=()):
    """Validate enriched archives and enumerate all eligible game members."""
    rank_summary = json.loads(
        (pathlib.Path(archives) / "player_ranks.json").read_text()
    )
    if rank_summary["sampling"] != player_ranks.SAMPLING:
        raise ValueError("player rank sampling contract mismatch")
    game_list = []
    for archive_path in sorted(pathlib.Path(archives).glob("*.zip")):
        with archive_path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != rank_summary["years"][archive_path.stem]["merged_sha256"]:
            raise ValueError(f"rank archive mismatch: {archive_path.name}")
        with zipfile.ZipFile(archive_path) as archive:
            for member in sorted(archive.namelist()):
                match = re.search(r"(20\d{2})(\d{2})(\d{2})", member)
                if match and member.endswith((".mjson", ".jsonl")):
                    game_list.append(
                        Game(
                            game_id=pathlib.Path(member).stem,
                            archive=archive_path,
                            member=member,
                        )
                    )
    if extra_archives:
        rank_summary["supplemental"] = []
        for path in extra_archives:
            extra_games, extra_summary = games(path)
            game_list.extend(extra_games)
            rank_summary["supplemental"].append(
                {
                    "archives": str(pathlib.Path(path).resolve()),
                    "player_ranks": extra_summary,
                }
            )
    if len({game.game_id for game in game_list}) != len(game_list):
        raise ValueError("duplicate game IDs across training archives")
    return game_list, rank_summary
