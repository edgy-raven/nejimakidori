import base64
import json
import pathlib
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import cv2
import numpy as np

import vision.mahjongsoul as mahjongsoul

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CAPTURE_ROOT = FIXTURES / "live_capture"


class ActionsTest(unittest.TestCase):
    def test_riichi_is_identified_when_kan_is_also_legal(self):
        for name in ["friendly-1", "friendly-2", "friendly-3"]:
            with self.subTest(account=name):
                image = mahjongsoul.normalize_image(
                    CAPTURE_ROOT / f"riichi_{name}.png"
                )
                detected = mahjongsoul.parse_actions(
                    image, {"buttons": ["discard", "reach", "kan", "skip"]}
                )
                self.assertEqual(detected["reach"]["verified_by"], "template")
                self.assertTrue(620 <= detected["reach"]["center"][0] <= 740)
                self.assertNotIn("kan", detected)
                self.assertNotIn(
                    "reach",
                    mahjongsoul.parse_actions(
                        image, {"buttons": ["discard", "kan", "skip"]}
                    ),
                )

    def test_call_templates_follow_buttons_and_reject_stale_offers(self):
        for name, action in [
            ("01_call_pon_skip.png", "pon"),
            ("10_call_chii_skip_opponent_open_meld.png", "chii"),
        ]:
            image = mahjongsoul.normalize_image(CAPTURE_ROOT / name)
            context = {"buttons": [action, "skip"]}
            original = mahjongsoul.parse_actions(image, context)
            shifted = mahjongsoul.parse_actions(
                cv2.warpAffine(
                    image, np.float32([[1, 0, -35], [0, 1, 8]]), (1280, 720)
                ),
                context,
            )
            self.assertEqual(set(shifted), {action, "skip"})
            self.assertEqual(
                shifted[action]["center"],
                [
                    original[action]["center"][0] - 35,
                    original[action]["center"][1] + 8,
                ],
            )
            self.assertNotIn(
                "kan",
                mahjongsoul.parse_actions(image, {"buttons": ["kan", "skip"]}),
            )

    def test_red_win_button_tracks_the_offer_and_requires_legal_win(self):
        image = mahjongsoul.normalize_image(CAPTURE_ROOT / "15_tsumo_offer.png")
        context = {"buttons": ["hora", "discard"]}
        original = mahjongsoul.parse_actions(image, context)["hora"]
        self.assertEqual(original["center"], [673, 567])
        shifted = mahjongsoul.parse_actions(
            cv2.warpAffine(
                image, np.float32([[1, 0, -35], [0, 1, 8]]), (1280, 720)
            ),
            context,
        )["hora"]
        self.assertEqual(shifted["center"], [638, 575])
        self.assertNotIn(
            "hora", mahjongsoul.parse_actions(image, {"buttons": ["discard"]})
        )
        for path in CAPTURE_ROOT.glob("*.png"):
            if path.name == "15_tsumo_offer.png":
                continue
            with self.subTest(image=path.name):
                self.assertNotIn(
                    "hora",
                    mahjongsoul.parse_actions(
                        mahjongsoul.normalize_image(path), context
                    ),
                )

    def test_unique_button_can_be_saved_then_recognized_with_other_offers(self):
        image = mahjongsoul.normalize_image(
            CAPTURE_ROOT / "01_call_pon_skip.png"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "button_anchor.png").write_bytes(
                (mahjongsoul.BUTTON_ROOT / "button_anchor.png").read_bytes()
            )
            (root / "skip.json").write_bytes(
                (mahjongsoul.BUTTON_ROOT / "skip.json").read_bytes()
            )
            with unittest.mock.patch.object(
                mahjongsoul, "BUTTON_ROOT", root
            ), unittest.mock.patch.object(
                mahjongsoul, "BUTTON_CAPTURE_ROOT", root
            ):
                detected = mahjongsoul.parse_actions(
                    image, {"buttons": ["pon", "skip"]}
                )
                self.assertEqual(
                    detected["pon"]["verified_by"], "unique-legal-button"
                )
                self.assertNotIn(
                    "kan",
                    mahjongsoul.parse_actions(
                        image, {"buttons": ["kan", "reach", "skip"]}
                    ),
                )
                (root / "pon.json").write_text(
                    json.dumps(
                        {
                            "box": detected["pon"]["box"],
                            "image": base64.b64encode(
                                cv2.imencode(".png", image)[1]
                            ).decode(),
                            "verifiedBy": {"type": "pon"},
                        }
                    )
                )
                learned = mahjongsoul.parse_actions(
                    image, {"buttons": ["pon", "kan", "skip"]}
                )
                self.assertEqual(learned["pon"]["verified_by"], "template")
                self.assertNotIn("kan", learned)
