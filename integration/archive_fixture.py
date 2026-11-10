"""Generated rank-enriched archive shared by dataset and importer workflows."""

import hashlib
import json
import zipfile

import riichienv

from log_dataset import player_ranks


def write_enriched_archive(root):
    """Write one valid, rank-enriched 2024 MJAI game."""
    environment = riichienv.RiichiEnv(
        seed=14, rule=riichienv.GameRule.default_tenhou()
    )
    observations = environment.reset()
    while not environment.done():
        observations = environment.step(
            {
                seat: observation.legal_actions()[0]
                for seat, observation in observations.items()
            }
        )
    events = environment.mjai_log
    events[0]["player_ranks"] = {
        "dan": [9, 9, 9, 9],
        "rating": [2500.0] * 4,
        "retained_seats": [0, 1, 2, 3],
    }
    archive = root / "2024.zip"
    member = "2024010100gm-00a9-0000-rewrite.mjson"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr(
            member,
            "".join(json.dumps(event) + "\n" for event in events),
        )
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    (root / "player_ranks.json").write_text(
        json.dumps(
            {
                "sampling": player_ranks.SAMPLING,
                "years": {"2024": {"merged_sha256": checksum}},
            }
        )
    )
