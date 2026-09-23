"""Download and verify the fixed Houou MJAI corpus."""

import argparse
import hashlib
import json
import pathlib
import shutil
import urllib.request
import zipfile

HOUOU_MJAI_RELEASE = "https://github.com/NikkeTryHard/tenhou-to-mjai/"
HOUOU_MJAI_ARCHIVES = {
    2024: (
        HOUOU_MJAI_RELEASE + "releases/download/v2.0.0/2024.zip",
        "4c0bb20ebdc263ab5819d848b620f6bf36304ccb70914722fd89b524be3afa1f",
    ),
    2025: (
        HOUOU_MJAI_RELEASE + "releases/download/v2.0.0/2025.zip",
        "0274e3f2a3b4ed32cd9af6d0f9895c061c5141e63bb6c90e561f349043b7a771",
    ),
}


def file_checksum(path):
    with open(path, "rb") as input_file:
        return hashlib.file_digest(input_file, "sha256").hexdigest()


def validate_archive(path, expected_checksum):
    summary_path = path.parent / "player_ranks.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())["years"][path.stem]
        if summary["archive_sha256"] != expected_checksum:
            raise ValueError(f"source SHA256 mismatch: {path}")
        expected_checksum = summary["merged_sha256"]
    if file_checksum(path) != expected_checksum:
        raise ValueError(f"SHA256 mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not names or any(not name.endswith(".mjson") for name in names):
            raise ValueError(f"Expected a nonempty MJAI archive: {path}")


def fetch_archive(output, year):
    path = output / f"{year}.zip"
    part_path = output / f".{year}.zip.part"
    url, expected_checksum = HOUOU_MJAI_ARCHIVES[year]
    if path.exists():
        validate_archive(path, expected_checksum)
        return path
    if part_path.exists() and file_checksum(part_path) == expected_checksum:
        validate_archive(part_path, expected_checksum)
        part_path.replace(path)
        return path
    offset = part_path.stat().st_size if part_path.exists() else 0
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "nejimakidori-model/1",
            **({"Range": f"bytes={offset}-"} if offset else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        append = offset > 0 and response.status == 206
        if response.status == 206 and not response.headers[
            "Content-Range"
        ].startswith(f"bytes {offset}-"):
            raise ValueError(f"Incorrect download resume offset: {url}")
        with part_path.open("ab" if append else "wb") as output_file:
            shutil.copyfileobj(response, output_file, length=1024 * 1024)
    validate_archive(part_path, expected_checksum)
    part_path.replace(path)
    return path


def run(output):
    """Return verified archive paths, resuming interrupted downloads."""
    output = pathlib.Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "player_ranks.json").exists():
        paths = [output / f"{year}.zip" for year in HOUOU_MJAI_ARCHIVES]
        for path in paths:
            validate_archive(path, HOUOU_MJAI_ARCHIVES[int(path.stem)][1])
        return paths
    return [fetch_archive(output, year) for year in HOUOU_MJAI_ARCHIVES]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    for path in run(parser.parse_args().output):
        print(path)
