"""Offline vision workflows using repository image fixtures."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy

ROOT = pathlib.Path(__file__).resolve().parents[1]


class VisionWorkflows(unittest.TestCase):
    def test_vision_worker_preserves_capture_and_tracks_real_offers(self):
        captures = ROOT / "integration/fixtures/vision/live_capture"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            screen = root / "room.png"
            original = (captures / "00_room_three_normal_ai.png").read_bytes()
            screen.write_bytes(original)
            requests = [
                {"mode": "room", "image": str(screen)} for _ in range(2)
            ]
            for name in (
                "lobby_home_debug1",
                "lobby_friendly_menu_current",
                "lobby_join_dialog_current",
            ):
                requests.append(
                    {
                        "mode": "lobby",
                        "image": str(captures / f"{name}.png"),
                    }
                )
            for name, action in (
                ("01_call_pon_skip.png", "pon"),
                ("10_call_chii_skip_opponent_open_meld.png", "chii"),
                ("15_tsumo_offer.png", "hora"),
            ):
                image = cv2.imread(str(captures / name))
                shifted = root / name
                cv2.imwrite(
                    str(shifted),
                    cv2.warpAffine(
                        image,
                        numpy.float32([[1, 0, -35], [0, 1, 8]]),
                        (1280, 720),
                    ),
                )
                for path, buttons in (
                    (captures / name, [action, "skip"]),
                    (shifted, [action, "skip"]),
                    (captures / name, ["discard"]),
                ):
                    requests.append(
                        {
                            "mode": "control",
                            "image": str(path),
                            "context": {
                                "buttons": buttons,
                                "hand": {"meldCount": 0},
                            },
                        }
                    )
            requests.extend(
                {
                    "mode": "lobby",
                    "image": str(
                        ROOT
                        / f"vision/static/live_capture/lobby_home_debug{number}.png"
                    ),
                }
                for number in range(1, 5)
            )
            result = subprocess.run(
                args=[
                    sys.executable,
                    str(ROOT / "mahjongsoul.py"),
                    "--serve",
                ],
                input="".join(json.dumps(item) + "\n" for item in requests),
                text=True,
                capture_output=True,
                check=True,
                timeout=90,
                env={**os.environ, "MAHJONG_SOUL_BUTTON_CAPTURES": directory},
            )
            outputs = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(outputs), len(requests))
            self.assertEqual(outputs[0], outputs[1])
            self.assertTrue(outputs[0]["in_room"])
            self.assertEqual(outputs[0]["ready"], [True] * 4)
            self.assertTrue(outputs[0]["start_enabled"])
            self.assertEqual(screen.read_bytes(), original)
            self.assertEqual(
                [item["screen"] for item in outputs[2:5]],
                ["home", "friendly_menu", "join_dialog"],
            )
            for offset, action in zip((5, 8, 11), ("pon", "chii", "hora")):
                before = outputs[offset]["actions"][action]["center"]
                after = outputs[offset + 1]["actions"][action]["center"]
                self.assertEqual(after, [before[0] - 35, before[1] + 8])
                self.assertNotIn(action, outputs[offset + 2]["actions"])
                self.assertEqual(outputs[offset]["rivers"], {})
            self.assertEqual(
                outputs[11]["actions"]["hora"]["center"], [673, 567]
            )
            self.assertEqual(
                [item["screen"] for item in outputs[-4:]], ["home"] * 4
            )

    def test_live_hand_targets_and_protocol_meld_count(self):
        import vision.mahjongsoul as mahjongsoul

        captures = ROOT / "integration/fixtures/vision/live_capture/targeting"
        for name, tiles, expected in (
            ("incoming_draw", ["1m", "8s"], [1022, 179, 935]),
            ("chii_dora", ["5s", "4p"], [683, 746, 368]),
        ):
            fixture = json.loads((captures / f"{name}.json").read_text())
            image = cv2.imread(str(captures / f"{name}.png"))
            fixture["tiles"] = tiles
            fixture["control"] = {"hand": mahjongsoul.parse_hand(image)}
            result = subprocess.run(
                args=[
                    "node",
                    "--input-type=module",
                    "-e",
                    """
    import {discardTarget} from './live/web/mahjongsoul_targeting.mjs';
let text = '';
for await (const chunk of process.stdin) text += chunk;
const fixture = JSON.parse(text);
console.log(JSON.stringify([fixture.action.pai, ...fixture.tiles].map(pai =>
    discardTarget({pai}, fixture.position, fixture.control).x)));
""",
                ],
                cwd=ROOT,
                input=json.dumps(fixture),
                text=True,
                capture_output=True,
                check=True,
                timeout=15,
            )
            self.assertEqual(json.loads(result.stdout), expected)
            if name == "chii_dora":
                image[620:720, 1040:1280] = 40
                hand = mahjongsoul.parse_hand(image, fixture["position"])
                self.assertEqual(hand["meld_count"], 1)
                self.assertEqual(len(hand["tiles"]), 11)
                self.assertTrue(hand["decision"])
