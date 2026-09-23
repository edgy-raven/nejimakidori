import json
import pathlib
import subprocess
import unittest

import cv2

import vision.mahjongsoul as mahjongsoul

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CAPTURE_ROOT = FIXTURES / "live_capture"


class TargetingTest(unittest.TestCase):
    def test_chii_greyed_first_tile_does_not_hide_a_legal_discard(self):
        image = mahjongsoul.normalize_image(
            CAPTURE_ROOT / "targeting/chii-grey-first.png"
        )
        control = mahjongsoul.parse_image(
            image,
            include_rivers=False,
            context={"hand": {"meldCount": 2}, "buttons": ["discard", "skip"]},
        )
        self.assertEqual(control["screen"], "table")
        self.assertTrue(control["control_ready"])
        self.assertEqual(len(control["hand"]["tiles"]), 8)
        self.assertEqual(control["hand"]["tiles"][4]["label"], "7s")
        self.assertTrue(control["hand"]["tiles"][4]["trusted"])

    def test_chii_greyed_last_tile_keeps_the_protocol_hand_size(self):
        fixture = CAPTURE_ROOT / "targeting/chii-grey-last"
        control = mahjongsoul.parse_image(
            mahjongsoul.normalize_image(fixture.with_suffix(".png")),
            include_rivers=False,
            context=json.loads(fixture.with_suffix(".json").read_text())[
                "position"
            ],
        )
        self.assertTrue(control["control_ready"])
        self.assertEqual(len(control["hand"]["tiles"]), 11)
        self.assertTrue(control["hand"]["decision"])

    def test_protocol_meld_count_survives_a_hidden_meld_animation(self):
        fixture_path = CAPTURE_ROOT / "targeting/chii_dora"
        fixture = json.loads(fixture_path.with_suffix(".json").read_text())
        image = cv2.imread(str(fixture_path.with_suffix(".png")))
        image[620:720, 1040:1280] = 40
        hand = mahjongsoul.parse_hand(image, fixture["position"])
        self.assertEqual(hand["meld_count"], 1)
        self.assertEqual(len(hand["tiles"]), 11)
        self.assertTrue(hand["decision"])

    def test_live_draw_and_chii_keep_exact_targets_despite_tile_effects(self):
        for name, tiles, expected in [
            ("incoming_draw", ["1m", "8s"], [1022, 179, 935]),
            ("chii_dora", ["5s", "4p"], [683, 746, 368]),
            ("chii-grey-last", ["3m", "7p"], [620, 179, 557]),
        ]:
            fixture_path = CAPTURE_ROOT / "targeting" / name
            fixture = json.loads(fixture_path.with_suffix(".json").read_text())
            fixture["tiles"] = tiles
            fixture["control"] = {
                "hand": mahjongsoul.parse_hand(
                    cv2.imread(str(fixture_path.with_suffix(".png"))),
                    fixture["position"],
                )
            }
            result = subprocess.run(
                args=[
                    "node",
                    "--input-type=module",
                    "-e",
                    """
    import {discardTarget} from './live/web/mahjongsoul_targeting.mjs';
    let input = '';
    for await (const chunk of process.stdin) input += chunk;
    const fixture = JSON.parse(input);
    console.log(JSON.stringify([fixture.action.pai, ...fixture.tiles].map(pai =>
      discardTarget({pai}, fixture.position, fixture.control).x)));
    """,
                ],
                cwd=mahjongsoul.ROOT.parent,
                input=json.dumps(fixture),
                text=True,
                capture_output=True,
                check=True,
                timeout=10,
            )
            self.assertEqual(json.loads(result.stdout), expected)
