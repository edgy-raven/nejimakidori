"""Public matches supply context; only approved teacher seats supply targets."""

import gzip
import json
import types
import zipfile

import pytest

from integration.rewrite import test_corpus_rebuild
from log_dataset import rebuild, riichilab


def test_import_deduplicates_matches_and_samples_only_assigned_teachers(
    tmp_path, monkeypatch
):
    games = {}
    histories = {bot: [] for bot in (120, 299, 351, 407, 408)}
    events = [
        {"type": "start_game"},
        {"type": "tsumo", "actor": 0, "pai": "5mr"},
        {"type": "end_game"},
    ]
    for number in range(6):
        bot_ids = [407, (120, 299, 351)[number % 3], 999, 408]
        if number == 0:
            bot_ids[2] = 299
        game = {
            "id": f"game-{number}",
            "played_at": f"2026-10-01 12:00:0{number}",
            "game_type": "ranked",
            "player_count": 4,
            "players": [
                {
                    "seat": seat,
                    "bot_id": bot,
                    "bot_name": str(bot),
                    "is_disconnected": number == 5 and seat == 1,
                    "is_penalized": False,
                }
                for seat, bot in enumerate(bot_ids)
            ],
        }
        games[game["id"]] = game
        for bot in bot_ids:
            if bot in histories:
                histories[bot].append(
                    {"game_id": game["id"], "played_at": game["played_at"]}
                )

    def api(path):
        if path.startswith("games/"):
            return games[path.split("/")[1]]
        rows = sorted(
            histories[int(path.split("/")[1])],
            key=lambda row: row["played_at"],
            reverse=True,
        )
        offset = int(path.split("offset=")[1])
        return {"total": len(rows), "games": rows[offset : offset + 2]}

    monkeypatch.setattr(riichilab, "api", api)
    monkeypatch.setattr(
        riichilab,
        "fetch",
        lambda url: gzip.compress("\n".join(map(json.dumps, events)).encode()),
    )
    output = tmp_path / "teachers"
    riichilab.run(
        types.SimpleNamespace(
            own_bots=[407, 408],
            teachers=["120:8", "299:8", "351:8"],
            output=output,
        )
    )
    imported, summary = rebuild.archive_games(output)
    assert len(imported) == 6
    assert summary["retained_rank_counts"] == {"8": 4}
    assert summary["rank_source"] == "user_assigned_training_dan"
    retained = 0
    with zipfile.ZipFile(output / "riichilab.zip") as archive:
        for member in archive.namelist():
            actual = [
                json.loads(line) for line in archive.read(member).splitlines()
            ]
            assert actual[1:] == events[1:]
            rank = actual[0]["player_ranks"]
            assert rank["dan"][0] is None and rank["dan"][3] is None
            for seat in rank["retained_seats"]:
                assert actual[0]["riichilab"]["bot_ids"][seat] in (
                    120,
                    299,
                    351,
                )
                assert rank["dan"][seat] == 8
                retained += 1
            if "game-5" in member:
                assert rank["dan"][1] is None
                assert not rank["retained_seats"]
    assert retained == 4

    primary = tmp_path / "tenhou"
    primary.mkdir()
    test_corpus_rebuild.write_enriched_archive(primary)
    combined, metadata = rebuild.archive_games(primary, [output])
    assert len(combined) == 7
    assert metadata["supplemental"][0]["player_ranks"] == summary
    with pytest.raises(ValueError, match="duplicate game IDs"):
        rebuild.archive_games(output, [output])
    with pytest.raises(ValueError, match="our bots"):
        riichilab.run(
            types.SimpleNamespace(
                own_bots=[407], teachers=["407:8"], output=tmp_path / "invalid"
            )
        )
    with pytest.raises(ValueError, match="incomplete game"):
        riichilab.enrich(games["game-0"], events[:-1], {120: 8})
