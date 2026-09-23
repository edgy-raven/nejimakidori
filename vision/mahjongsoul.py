#!/usr/bin/env python3

import argparse
import base64
import functools
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import vision.postgame as postgame
import vision.river_geometry as river_geometry
import vision.tile_matcher as tile_matcher

ROOT = Path(__file__).resolve().parent
CAPTURE_ROOT = ROOT / "static" / "live_capture"
CANVAS_SIZE = (1280, 720)
BOARD_SCALE = 4.0 / 3.0
HAND_Y = 614
HAND_HEIGHT = 101
HAND_X = 148
HAND_STEP = 63
HAND_WIDTH = 62
BUTTON_ROOT = ROOT / "static" / "action_buttons"
BUTTON_CAPTURE_ROOT = Path(
    os.environ.get(
        "MAHJONG_SOUL_BUTTON_CAPTURES",
        str(Path.home() / ".local/state/nejimakidori/button-captures"),
    )
)
ACTION_SEARCH_BOX = (350, 480, 680, 120)
OVERLAY_SEARCH_BOX = (300, 350, 680, 250)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse a canonical Mahjong Soul live screenshot."
    )
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--lobby-only", action="store_true")
    parser.add_argument("--overlays-only", action="store_true")
    args = parser.parse_args()
    if args.serve:
        if (
            args.image
            or args.overlay
            or args.control_only
            or args.lobby_only
            or args.overlays_only
        ):
            parser.error("--serve reads image and mode from JSON lines")
    elif args.image is None:
        parser.error("image is required without --serve")
    return args


def normalize_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    if width * CANVAS_SIZE[1] != height * CANVAS_SIZE[0]:
        raise ValueError(f"expected a 16:9 screenshot, got {width}x{height}")
    if (width, height) != CANVAS_SIZE:
        image = cv2.resize(image, CANVAS_SIZE, interpolation=cv2.INTER_AREA)
    return image


def tile_face_ratio(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 80)).mean())


def crop_box(image, box):
    x, y, width, height = box
    return image[y : y + height, x : x + width]


@functools.lru_cache(maxsize=1)
def lobby_reference_images():
    return {
        screen: tuple(
            (
                box,
                crop_box(normalize_image(path), box),
            )
            for path in CAPTURE_ROOT.glob(f"lobby_{screen}*.png")
        )
        for screen, box in {
            "join_dialog": (440, 148, 310, 45),
            "home": (845, 468, 205, 65),
            "friendly_menu": (845, 350, 205, 70),
        }.items()
    }


def parse_lobby(image):
    scores = {}
    for screen, templates in lobby_reference_images().items():
        scores[screen] = max(
            float(
                cv2.matchTemplate(
                    crop_box(image, box),
                    template,
                    cv2.TM_CCOEFF_NORMED,
                )[0, 0]
            )
            for box, template in templates
        )
    return {
        "screen": next(
            (screen for screen in scores if scores[screen] >= 0.95),
            "unknown",
        ),
        "scores": scores,
    }


@functools.lru_cache(maxsize=2)
def room_reference_image(name="00_room_three_normal_ai.png"):
    return normalize_image(CAPTURE_ROOT / name)


def parse_room(image):
    room_score, start_score = (
        float(
            cv2.matchTemplate(
                crop_box(image, box),
                crop_box(room_reference_image(), box),
                cv2.TM_CCOEFF_NORMED,
            )[0, 0]
        )
        for box in [(140, 37, 100, 43), (695, 642, 133, 44)]
    )
    ready_template = cv2.inRange(
        cv2.cvtColor(
            crop_box(room_reference_image(), (145, 440, 145, 100)),
            cv2.COLOR_BGR2HSV,
        ),
        (35, 120, 140),
        (90, 255, 255),
    )
    ready_scores = [
        float(
            cv2.matchTemplate(
                cv2.inRange(
                    cv2.cvtColor(
                        crop_box(image, (x, 430, 165, 120)),
                        cv2.COLOR_BGR2HSV,
                    ),
                    (35, 120, 140),
                    (90, 255, 255),
                ),
                ready_template,
                cv2.TM_CCOEFF_NORMED,
            ).max()
        )
        for x in [135, 390, 645, 900]
    ]
    return {
        "in_room": room_score >= 0.95,
        "match_end_confirm": all(
            cv2.matchTemplate(
                crop_box(image, box),
                crop_box(
                    room_reference_image("13_match_end_normal_ai.png"), box
                ),
                cv2.TM_CCOEFF_NORMED,
            )[0, 0]
            >= 0.9
            for box in [(537, 42, 181, 32), (1085, 639, 154, 42)]
        ),
        "ready": [score >= 0.8 for score in ready_scores],
        "start_enabled": start_score >= 0.95,
        "scores": {
            "room": room_score,
            "start": start_score,
            "ready": ready_scores,
        },
    }


def recognize_hand_tile(image, x, drawn=False):
    box = (x, HAND_Y, HAND_WIDTH, HAND_HEIGHT)
    result = tile_matcher.score_candidate(
        crop_box(image, box),
        ["none"],
        references=tile_matcher.atlas_reference_tiles(),
        include_scores=True,
    )
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "margin": result["margin"],
        "trusted": result["confidence"] >= 0.78
        and result["margin"] >= 0.012
        and (
            result["label"] != "white"
            or float(
                np.mean(
                    cv2.cvtColor(
                        image[HAND_Y + 18 : HAND_Y + 89, x + 8 : x + 54],
                        cv2.COLOR_BGR2GRAY,
                    )
                    < 150
                )
            )
            < 0.02
        ),
        "drawn": drawn,
        "glare": float(
            np.mean(
                np.all(
                    image[HAND_Y + 18 : HAND_Y + 89, x + 8 : x + 54] > 245,
                    axis=2,
                )
            )
        )
        > 0.08,
        "box": box,
        "top3": result["top3"],
        "scores": result["scores"],
    }


def bottom_meld_count(image):
    count = 0
    for right in (1210, 1040, 880, 720):
        if tile_face_ratio(image[625:710, right - 170 : right]) < 0.25:
            break
        count += 1
    return count


def parse_hand(image, context=None):
    meld_count = (
        context["hand"]["meldCount"] if context else bottom_meld_count(image)
    )
    main_count = 13 - 3 * meld_count
    # Chi can grey out the leftmost tile under the discard restriction.
    # Check the hand's slots instead of treating that one tile as the table.
    if all(
        tile_face_ratio(
            crop_box(
                image, (HAND_X + HAND_STEP * index, HAND_Y, HAND_WIDTH, 90)
            )
        )
        < 0.4
        for index in range(main_count)
    ):
        return {"meld_count": meld_count, "tiles": [], "decision": False}

    tiles = [
        recognize_hand_tile(image, HAND_X + HAND_STEP * index)
        for index in range(main_count)
    ]
    drawn_x = 991 - 190 * meld_count
    draw_present = (
        tile_face_ratio(
            crop_box(image, (drawn_x, HAND_Y, HAND_WIDTH, HAND_HEIGHT))
        )
        >= 0.4
    )
    gap_ratio = tile_face_ratio(
        crop_box(image, (drawn_x - 20, HAND_Y + 8, 18, HAND_HEIGHT - 16))
    )
    if draw_present and gap_ratio < 0.2:
        tiles.append(recognize_hand_tile(image, drawn_x, drawn=True))
    elif (
        context
        and "concealed" in context["hand"]
        and len(context["hand"]["concealed"]) == main_count + 1
    ) or (
        tile_face_ratio(
            crop_box(
                image,
                (
                    HAND_X + HAND_STEP * main_count,
                    HAND_Y,
                    HAND_WIDTH,
                    HAND_HEIGHT,
                ),
            )
        )
        >= 0.4
    ):
        tiles.append(
            recognize_hand_tile(image, HAND_X + HAND_STEP * main_count)
        )
    return {
        "meld_count": meld_count,
        "tiles": tiles,
        "decision": len(tiles) == 14 - 3 * meld_count,
        "trusted": all(tile["trusted"] for tile in tiles),
    }


@functools.lru_cache(maxsize=1)
def river_projections():
    explicit, generic = river_geometry.load_river_annotations()
    annotations = explicit + river_geometry.assign_generic_players(
        explicit, generic
    )
    return {
        player: river_geometry.fit_player_projection(
            player, player_annotations
        )["all"]
        for player, player_annotations in river_geometry.group_by_player(
            annotations
        ).items()
    }


def parse_rivers(image):
    rivers = {}
    for player, projection in river_projections().items():
        tiles = []
        for row in range(3):
            for col in range(1, 7):
                points = (
                    river_geometry.project_cell_quad(
                        projection["homography"], row, col
                    )
                    * BOARD_SCALE
                )
                tile = river_geometry.warp_from_points(
                    image,
                    points.astype(np.float32),
                    river_geometry.PORTRAIT_SIZE,
                )
                occupancy = tile_face_ratio(tile)
                if occupancy < 0.35:
                    continue
                result = tile_matcher.score_candidate(tile, ["none", "180"])
                tiles.append(
                    {
                        "row": row,
                        "col": col,
                        "occupancy": round(occupancy, 4),
                        "label": result["label"],
                        "confidence": result["confidence"],
                        "margin": result["margin"],
                        "trusted": result["confidence"] >= 0.78
                        and result["margin"] >= 0.03,
                        "points": points.round().astype(int).tolist(),
                        "top3": result["top3"],
                    }
                )
        rivers[player] = tiles
    return rivers


@functools.lru_cache(maxsize=None)
def action_reference_image(path, modified_ns):
    reference = json.loads(path.read_text())
    return (
        reference.get("verifiedBy", {}).get("type", path.stem),
        crop_box(
            cv2.imdecode(
                np.frombuffer(
                    base64.b64decode(reference["image"]), dtype=np.uint8
                ),
                cv2.IMREAD_COLOR,
            ),
            reference["box"],
        ),
    )


def action_reference_images():
    references = {}
    for root in (BUTTON_ROOT, BUTTON_CAPTURE_ROOT):
        for path in root.glob("*.json"):
            action, image = action_reference_image(
                path, path.stat().st_mtime_ns
            )
            references.setdefault(action, []).append(image)
    return references


def parse_actions(image, context=None):
    actions = {}
    legal = context["buttons"] if context else None
    for action, references in action_reference_images().items():
        for reference in references:
            scores = cv2.matchTemplate(
                crop_box(image, ACTION_SEARCH_BOX),
                reference,
                cv2.TM_CCOEFF_NORMED,
            )
            _, score, _, location = cv2.minMaxLoc(scores)
            if score >= 0.8 and (
                action not in actions or score > actions[action]["score"]
            ):
                actions[action] = {
                    "score": round(float(score), 4),
                    "center": [
                        ACTION_SEARCH_BOX[0]
                        + location[0]
                        + reference.shape[1] // 2,
                        ACTION_SEARCH_BOX[1]
                        + location[1]
                        + reference.shape[0] // 2,
                    ],
                    "box": [
                        ACTION_SEARCH_BOX[0] + location[0],
                        ACTION_SEARCH_BOX[1] + location[1],
                        reference.shape[1],
                        reference.shape[0],
                    ],
                    "verified_by": "template",
                }
    if legal is not None:
        unknown = set(legal) - {"discard", "skip"} - actions.keys()
        if len(unknown) == 1:
            scores = cv2.matchTemplate(
                cv2.cvtColor(
                    crop_box(image, ACTION_SEARCH_BOX), cv2.COLOR_BGR2GRAY
                ),
                cv2.imread(str(BUTTON_ROOT / "button_anchor.png"), 0),
                cv2.TM_CCOEFF_NORMED,
            )
            candidates = []
            while scores.max() >= 0.8:
                _, score, _, (x, y) = cv2.minMaxLoc(scores)
                center = [
                    ACTION_SEARCH_BOX[0] + x + 67,
                    ACTION_SEARCH_BOX[1] + y - 13,
                ]
                if all(
                    abs(center[0] - value["center"][0]) > 90
                    for value in actions.values()
                ):
                    candidates.append(
                        {
                            "score": round(float(score), 4),
                            "center": center,
                            "box": [center[0] - 80, center[1] - 40, 160, 80],
                            "verified_by": "unique-legal-button",
                        }
                    )
                scores[max(0, y - 25) : y + 26, max(0, x - 90) : x + 91] = -1
            if len(candidates) == 1:
                actions[unknown.pop()] = candidates[0]
        actions = {
            name: value for name, value in actions.items() if name in legal
        }
    return actions


def parse_overlays(image):
    x, y, width, height = OVERLAY_SEARCH_BOX
    hsv = cv2.cvtColor(crop_box(image, OVERLAY_SEARCH_BOX), cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= 10)
        & (hsv[:, :, 0] <= 40)
        & (hsv[:, :, 1] >= 80)
        & (hsv[:, :, 2] >= 150)
    ).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    gold_buttons = []
    for component in range(1, count):
        left, top, component_width, component_height, area = stats[component]
        if component_width < 80 or component_height < 25 or area < 1000:
            continue
        gold_buttons.append(
            {
                "center": [
                    round(x + centers[component][0]),
                    round(y + centers[component][1]),
                ],
                "area": int(area),
            }
        )
    gold_buttons.sort(key=lambda button: button["area"], reverse=True)
    afk = next(
        (
            button
            for button in gold_buttons
            if 540 <= button["center"][0] <= 745
            and 435 <= button["center"][1] <= 490
        ),
        None,
    )
    return {"afk": afk, "gold_buttons": gold_buttons}


def parse_image(image, include_rivers=True, context=None):
    hand = parse_hand(image, context)
    actions = parse_actions(image, context)
    overlays = parse_overlays(image)
    if hand["tiles"] or actions:
        screen = "table"
    else:
        screen = "unknown"
    rivers = parse_rivers(image) if screen == "table" and include_rivers else {}
    return {
        "screen": screen,
        "canvas": {"width": CANVAS_SIZE[0], "height": CANVAS_SIZE[1]},
        "hand": hand,
        "actions": actions,
        "overlays": overlays,
        "rivers": rivers,
        "control_ready": screen == "table"
        and (hand.get("decision", False) or bool(actions)),
        "safe_to_act": screen == "table"
        and hand.get("decision", False)
        and hand.get("trusted", False),
    }


def draw_overlay(image, result):
    overlay = image.copy()
    for tile in result["hand"]["tiles"]:
        x, y, width, height = tile["box"]
        color = (60, 220, 90) if tile["trusted"] else (0, 180, 255)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            overlay,
            tile["label"],
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    for tiles in result["rivers"].values():
        for tile in tiles:
            points = np.array(tile["points"], dtype=np.int32)
            color = (60, 220, 90) if tile["trusted"] else (0, 180, 255)
            cv2.polylines(overlay, [points], True, color, 1, cv2.LINE_AA)
    return overlay


def main():
    args = parse_args()
    if args.serve:
        for line in sys.stdin:
            request = json.loads(line)
            image = normalize_image(request["image"])
            if request["mode"] == "lobby":
                result = parse_lobby(image)
            elif request["mode"] == "autoqueue":
                result = postgame.detect_controls(image)
            elif request["mode"] in {"ready", "cancel_vote", "ranked"}:
                result = postgame.detect_controls(image, mode=request["mode"])
            elif request["mode"] == "room":
                result = parse_room(image)
                if not result["in_room"]:
                    result["screen"] = parse_lobby(image)["screen"]
            elif request["mode"] == "overlays":
                result = {"overlays": parse_overlays(image)}
            elif request["mode"] == "control":
                result = parse_image(
                    image, include_rivers=False, context=request["context"]
                )
            else:
                raise ValueError(f"unknown vision mode: {request['mode']}")
            print(json.dumps(result), flush=True)
        return
    image = normalize_image(args.image)
    result = (
        parse_lobby(image)
        if args.lobby_only
        else (
            {"overlays": parse_overlays(image)}
            if args.overlays_only
            else parse_image(image, include_rivers=not args.control_only)
        )
    )
    print(json.dumps(result, indent=2))
    if args.overlay:
        if not cv2.imwrite(str(args.overlay), draw_overlay(image, result)):
            raise OSError(f"could not write {args.overlay}")


if __name__ == "__main__":
    main()
