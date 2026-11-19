"""Checkpoint matches consume verified policy exports."""

import hashlib
import json

import pytest


def test_phase_evaluation_passes_loadable_export_to_matches(
    tmp_path, monkeypatch
):
    import pathlib
    import types

    from training import checkpoint_match as evaluation

    calls = []
    weights = tmp_path / "weights.h5"
    weights.write_bytes(b"saved checkpoint")

    def run(argv, check):
        assert check
        calls.append(argv)
        if argv[2] == "model.export":
            exported = (
                pathlib.Path(argv[argv.index("--output") + 1]) / "saved_model"
            )
            exported.mkdir(parents=True)
            (exported / "saved_model.pb").write_bytes(b"fixture")
            (exported.parent / "export-verification.json").write_text(
                json.dumps(
                    {
                        "weight_sha256": hashlib.sha256(
                            weights.read_bytes()
                        ).hexdigest()
                    }
                )
            )
        else:
            exported = pathlib.Path(argv[argv.index("--new") + 1])
            assert (exported / "saved_model.pb").is_file()
            assert argv[argv.index("--old") + 1] == args.opponent
            assert argv[argv.index("--games") + 1] == str(args.games)
            assert argv[argv.index("--workers") + 1] == str(args.workers)
            assert argv[argv.index("--seed") + 1] == "941"
            assert argv[argv.index("--batch-games") + 1] == "32"
            assert argv[argv.index("--devices") + 1] == "GPU-a,GPU-b"

    monkeypatch.setattr(evaluation.subprocess, "run", run)
    args = types.SimpleNamespace(
        output=str(tmp_path),
        weights=str(weights),
        opponent="incumbent/saved_model",
        games=512,
        seed=941,
        workers=4,
        devices="GPU-a,GPU-b",
        batch_games=32,
        mortal_source=None,
    )
    args.games = 3
    with pytest.raises(ValueError, match="complete four-game"):
        evaluation.run(args)
    assert calls == []
    args.games = 512
    evaluation.run(args)
    assert len(calls) == 2
    args.opponent = "mortal/checkpoint.pth"
    args.mortal_source = "mortal/source"
    evaluation.run(args)
    assert calls[-1][-2:] == ["--mortal-source", "mortal/source"]
    assert len(calls) == 3
    assert calls[-1][2] == "training.evaluate"
    weights.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="different weights"):
        evaluation.run(args)
    assert len(calls) == 3
