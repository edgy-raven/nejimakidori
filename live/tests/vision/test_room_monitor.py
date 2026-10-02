import json
import pathlib
import subprocess
import sys
import unittest

import cv2

import vision.mahjongsoul as mahjongsoul

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CAPTURE_ROOT = FIXTURES / "live_capture"


class RoomMonitorTest(unittest.TestCase):
    def test_room_mode_through_controller_vision_worker(self):
        result = subprocess.run(
            args=[
                sys.executable,
                str(mahjongsoul.ROOT.parent / "mahjongsoul.py"),
                "--serve",
            ],
            input=json.dumps(
                {
                    "mode": "room",
                    "image": str(CAPTURE_ROOT / "00_room_three_normal_ai.png"),
                }
            )
            + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        room = json.loads(result.stdout)
        self.assertTrue(room["in_room"])
        self.assertEqual(room["ready"], [True] * 4)
        self.assertTrue(room["start_enabled"])

    def test_room_readiness_and_non_room_screens(self):
        image = mahjongsoul.room_reference_image().copy()
        room = mahjongsoul.parse_room(image)
        self.assertTrue(room["in_room"])
        self.assertEqual(room["ready"], [True] * 4)
        self.assertTrue(room["start_enabled"])
        self.assertFalse(room["match_end_confirm"])

        # A player has not readied up, even if Start is still animating.
        image[430:550, 645:810] = 0
        self.assertEqual(
            mahjongsoul.parse_room(image)["ready"],
            [True, True, False, True],
        )

        # The same Start label on a disabled grey button must not qualify.
        image = mahjongsoul.room_reference_image().copy()
        image[642:686, 695:828] = cv2.cvtColor(
            cv2.cvtColor(image[642:686, 695:828], cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR,
        )
        self.assertFalse(mahjongsoul.parse_room(image)["start_enabled"])

        for path in CAPTURE_ROOT.glob("*.png"):
            if path.name == "00_room_three_normal_ai.png":
                continue
            with self.subTest(screen=path.name):
                room = mahjongsoul.parse_room(mahjongsoul.normalize_image(path))
                self.assertFalse(room["in_room"])

        room = mahjongsoul.parse_room(
            mahjongsoul.normalize_image(
                CAPTURE_ROOT / "00_room_three_normal_ai.png"
            )
        )
        self.assertTrue(room["in_room"])
        self.assertEqual(room["ready"], [True] * 4)
        self.assertTrue(room["start_enabled"])

    def test_match_end_requires_heading_and_confirm_button(self):
        image = mahjongsoul.room_reference_image(
            "13_match_end_normal_ai.png"
        ).copy()
        self.assertTrue(mahjongsoul.parse_room(image)["match_end_confirm"])
        image[639:681, 1085:1239] = 0
        self.assertFalse(mahjongsoul.parse_room(image)["match_end_confirm"])


if __name__ == "__main__":
    unittest.main()
