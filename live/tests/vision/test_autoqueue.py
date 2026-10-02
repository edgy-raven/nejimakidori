import pathlib
import types
import unittest
import unittest.mock

import numpy as np

import vision.mahjongsoul as mahjongsoul
import vision.postgame as postgame

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CAPTURE_ROOT = FIXTURES / "live_capture"


class AutoqueueTest(unittest.TestCase):
    def test_ranked_navigation_recognizes_rooms_and_only_four_player_modes(
        self,
    ):
        for name, screen, controls in (
            ("ranked-home", "home", ["ranked match"]),
            (
                "ranked-rooms",
                "ranked rooms",
                ["bronze room", "silver room", "gold room"],
            ),
            (
                "ranked-higher-rooms",
                "ranked rooms",
                ["gold room", "jade room", "throne room"],
            ),
            (
                "ranked-bronze",
                "ranked modes",
                ["back", "4 player east", "4 player south"],
            ),
        ):
            with self.subTest(name=name):
                image = mahjongsoul.normalize_image(
                    FIXTURES / f"autoqueue/{name}.png"
                )
                result = postgame.detect_controls(image, mode="ranked")
                self.assertEqual(result["screen"], screen)
                self.assertEqual(
                    [target["control"] for target in result["targets"]],
                    controls,
                )
                if screen == "ranked modes":
                    self.assertEqual(result["room"], "Bronze")
                    dimmed = postgame.detect_controls(
                        (image * 0.5).astype(np.uint8), mode="ranked"
                    )
                    self.assertFalse(
                        any(
                            target["control"].startswith("4 player")
                            for target in dimmed["targets"]
                        )
                    )

    def test_insufficient_coins_stops_before_confirm_can_be_clicked(self):
        result = postgame.detect_controls(
            mahjongsoul.normalize_image(
                FIXTURES / "autoqueue/insufficient-coins.png"
            )
        )
        self.assertEqual(result["targets"], [])
        self.assertEqual(result["error"]["code"], 1304)
        self.assertEqual(
            result["error"]["message"],
            "Not enough coins to enter this room.",
        )

    def test_ranked_copper_error_exposes_only_safe_exit(self):
        result = postgame.detect_controls(
            mahjongsoul.normalize_image(
                FIXTURES / "autoqueue/insufficient-coins.png"
            ),
            mode="ranked",
        )
        self.assertEqual(result["error"]["code"], 1304)
        self.assertEqual(
            [target["control"] for target in result["targets"]],
            ["confirm"],
        )

    def test_result_button_and_non_result_screens(self):
        image = mahjongsoul.normalize_image(
            CAPTURE_ROOT / "13_match_end_normal_ai.png"
        )
        target = postgame.detect_controls(image)["targets"][0]
        self.assertEqual(target["control"], "confirm")
        self.assertTrue(1080 < target["center"][0] < 1245)
        self.assertTrue(635 < target["center"][1] < 690)
        image[630:700, 1070:1255] = 0
        self.assertEqual(postgame.detect_controls(image)["targets"], [])
        for name in (
            "lobby_home_debug1.png",
            "05_closed_14_tile_discard_decision.png",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    postgame.detect_controls(
                        mahjongsoul.normalize_image(CAPTURE_ROOT / name)
                    )["targets"],
                    [],
                )

    def test_rematch_takes_priority_over_exit_confirmation(self):
        target = postgame.detect_controls(
            mahjongsoul.normalize_image(FIXTURES / "autoqueue/0246.png")
        )["targets"][0]
        self.assertEqual(target["control"], "one more match")
        self.assertTrue(890 < target["center"][0] < 1055)

    def test_item_reward_overlay_can_be_dismissed(self):
        for name in ["reward-received.png", "reward-cookie.png"]:
            with self.subTest(name=name):
                target = postgame.detect_controls(
                    mahjongsoul.normalize_image(FIXTURES / "autoqueue" / name)
                )["targets"][0]
                self.assertEqual(target["control"], "dismiss reward")
                self.assertEqual(target["center"], [640, 560])

    def test_character_reward_scene_skip(self):
        target = postgame.detect_controls(
            mahjongsoul.normalize_image(
                FIXTURES / "autoqueue/character-reward.png"
            )
        )["targets"][0]
        self.assertEqual(target["control"], "skip")
        self.assertGreater(target["center"][0], 1050)
        self.assertLess(target["center"][1], 110)

    def test_both_buttons_survive_detection_on_live_reward_screen(self):
        targets = postgame.detect_controls(
            mahjongsoul.normalize_image(
                CAPTURE_ROOT / "match_end_confirm_autoqueue_off.png"
            )
        )["targets"]
        self.assertEqual(
            [target["control"] for target in targets],
            ["one more match", "confirm"],
        )


class ReadyTest(unittest.TestCase):
    def test_ready_requires_lit_button_text_in_room_control_area(self):
        targets = postgame.detect_controls(
            mahjongsoul.normalize_image(CAPTURE_ROOT / "room_ready_guest.jpg"),
            mode="ready",
        )["targets"]
        self.assertEqual([target["control"] for target in targets], ["ready"])
        self.assertTrue(700 < targets[0]["center"][0] < 820)
        self.assertTrue(630 < targets[0]["center"][1] < 700)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[630:670, 700:800] = (0, 180, 240)
        result = types.SimpleNamespace(
            txts=["Ready", "Ready", "Cancel", "Ready"],
            scores=[0.99, 0.99, 0.99, 0.99],
            boxes=np.array(
                [
                    [[700, 630], [800, 630], [800, 670], [700, 670]],
                    [[700, 200], [800, 200], [800, 240], [700, 240]],
                    [[700, 630], [800, 630], [800, 670], [700, 670]],
                    [[850, 630], [950, 630], [950, 670], [850, 670]],
                ]
            ),
        )
        with unittest.mock.patch.object(postgame, "button_ocr") as ocr:
            ocr.return_value.return_value = result
            targets = postgame.detect_controls(image, mode="ready")
            self.assertEqual(len(targets["targets"]), 1)
            self.assertEqual(targets["targets"][0]["center"], [750, 650])
            self.assertEqual(postgame.detect_controls(image)["targets"], [])
            result.txts[0] = "Error code: 1304"
            self.assertEqual(
                postgame.detect_controls(image, mode="ready")["error"]["code"],
                1304,
            )


class CancelVoteTest(unittest.TestCase):
    def test_only_lit_affirmative_vote_labels_become_targets(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[400:440, 500:600] = (0, 180, 240)
        result = types.SimpleNamespace(
            txts=["Yes", "Agree", "No", "Disagree", "Confirm", "Yes"],
            scores=[0.99] * 6,
            boxes=np.array(
                [[[500, 400], [600, 400], [600, 440], [500, 440]]] * 5
                + [[[700, 400], [800, 400], [800, 440], [700, 440]]]
            ),
        )
        with unittest.mock.patch.object(postgame, "button_ocr") as ocr:
            ocr.return_value.return_value = result
            targets = postgame.detect_controls(image, mode="cancel_vote")
        self.assertEqual(
            [target["control"] for target in targets["targets"]],
            ["yes", "agree"],
        )
        self.assertEqual(targets["targets"][0]["center"], [550, 420])


class RankedPopupTest(unittest.TestCase):
    def test_ranked_popup_allowlist_and_rank_up_priority(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[550:590, 590:690] = (0, 180, 240)
        result = types.SimpleNamespace(
            txts=["Star Up", "Confirm"],
            scores=[0.99, 0.99],
            boxes=np.array(
                [
                    [[500, 100], [780, 100], [780, 150], [500, 150]],
                    [[590, 550], [690, 550], [690, 590], [590, 590]],
                ]
            ),
        )
        with unittest.mock.patch.object(postgame, "button_ocr") as ocr:
            ocr.return_value.return_value = result
            for label, screen in [
                ("Star Up", "star up"),
                ("Daily Quest", "daily quest"),
                ("Rank Up", "rank up"),
                ("Special Offer", None),
            ]:
                result.txts[0] = label
                detected = postgame.detect_controls(image, mode="ranked")
                self.assertEqual(detected.get("screen"), screen)
                self.assertEqual(
                    [item["control"] for item in detected["targets"]],
                    ["confirm"] if screen in {"star up", "daily quest"} else [],
                )

    def test_ranked_chest_fixtures_are_allowed_character_scene_is_not(self):
        for name in ["reward-received", "reward-cookie"]:
            detected = postgame.detect_controls(
                mahjongsoul.normalize_image(FIXTURES / f"autoqueue/{name}.png"),
                mode="ranked",
            )
            self.assertEqual(detected["screen"], "chest reward")
            self.assertEqual(
                detected["targets"][0]["control"], "dismiss reward"
            )
        detected = postgame.detect_controls(
            mahjongsoul.normalize_image(
                FIXTURES / "autoqueue/character-reward.png"
            ),
            mode="ranked",
        )
        self.assertEqual(detected["targets"], [])


class RankedResultFlowTest(unittest.TestCase):
    def test_result_stages_and_hand_settlement_are_distinct_known_screens(self):
        for name, screen, stage in [
            ("autoqueue/0234.png", "match end", "results"),
            ("autoqueue/bajirisuku-match-end.png", "match end", "results"),
            ("autoqueue/bajirisuku-matchend.png", "match end", "results"),
            ("autoqueue/0240.png", "match end", "rank progress"),
            (
                "autoqueue/bajirisuku-split-rank.png",
                "match end",
                "rank progress",
            ),
            ("autoqueue/0246.png", "match end", "rematch"),
            (
                "live_capture/14_ai_riichi_ippatsu_ron_dora.png",
                "hand result",
                None,
            ),
            (
                "live_capture/09_match_result_ranking.png",
                "result waiting",
                None,
            ),
        ]:
            detected = postgame.detect_controls(
                mahjongsoul.normalize_image(FIXTURES / name),
                mode="ranked",
            )
            self.assertEqual(detected["screen"], screen)
            self.assertEqual(detected.get("stage"), stage)
            if stage:
                self.assertIn(
                    "confirm",
                    [target["control"] for target in detected["targets"]],
                )

    def test_verified_result_buttons_do_not_require_a_title(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[640:680, 1090:1230] = (0, 180, 240)
        result = types.SimpleNamespace(
            txts=["Confirm"],
            scores=[0.99],
            boxes=np.array(
                [[[1090, 640], [1230, 640], [1230, 680], [1090, 680]]]
            ),
        )
        with unittest.mock.patch.object(postgame, "button_ocr") as ocr:
            ocr.return_value.return_value = result
            for label in ["Confirm", "One More Match"]:
                result.txts[0] = label
                detected = postgame.detect_controls(image, mode="ranked")
                self.assertEqual(detected["screen"], "match end")
                self.assertEqual(
                    detected["targets"][0]["control"], label.lower()
                )
            image[:] = 0
            self.assertEqual(
                postgame.detect_controls(image, mode="ranked")["targets"], []
            )

    def test_known_hand_result_continue_is_allowed_but_offer_is_not(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[640:680, 1090:1230] = (0, 180, 240)
        result = types.SimpleNamespace(
            txts=["Ron", "Dora", "Continue"],
            scores=[0.99] * 3,
            boxes=np.array(
                [
                    [[1000, 100], [1100, 100], [1100, 140], [1000, 140]],
                    [[500, 100], [600, 100], [600, 140], [500, 140]],
                    [[1090, 640], [1230, 640], [1230, 680], [1090, 680]],
                ]
            ),
        )
        with unittest.mock.patch.object(postgame, "button_ocr") as ocr:
            ocr.return_value.return_value = result
            detected = postgame.detect_controls(image, mode="ranked")
            self.assertEqual(detected["targets"][0]["control"], "continue")
            result.txts[0] = "Special Offer"
            self.assertEqual(
                postgame.detect_controls(image, mode="ranked")["targets"], []
            )


class RematchConfirmationTest(unittest.TestCase):
    def test_official_rematch_prompt_exposes_only_modal_confirm_and_cancel(
        self,
    ):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[515:545, 480:570] = (0, 180, 240)
        image[515:545, 710:800] = (180, 80, 0)
        result = types.SimpleNamespace(
            txts=[
                "Match End",
                "One More Match",
                "How about another match of",
                "Confirm",
                "Cancel",
            ],
            scores=[0.99] * 5,
            boxes=np.array(
                [
                    [[10, 20], [200, 20], [200, 60], [10, 60]],
                    [[500, 160], [780, 160], [780, 200], [500, 200]],
                    [[400, 290], [880, 290], [880, 320], [400, 320]],
                    [[480, 515], [570, 515], [570, 545], [480, 545]],
                    [[710, 515], [800, 515], [800, 545], [710, 545]],
                ]
            ),
        )
        with unittest.mock.patch.object(postgame, "button_ocr") as ocr:
            ocr.return_value.return_value = result
            detected = postgame.detect_controls(image, mode="ranked")
        self.assertEqual(detected["screen"], "rematch confirmation")
        self.assertEqual(
            [target["control"] for target in detected["targets"]],
            ["confirm", "cancel"],
        )
