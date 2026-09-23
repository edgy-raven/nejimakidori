"""Recognize rematch and reward controls from canonical game screenshots."""

import functools
import re

import cv2
import rapidocr

REMATCH_LABELS = (
    "one more game",
    "one more match",
    "another game",
    "play again",
)


@functools.lru_cache(maxsize=1)
def button_ocr():
    return rapidocr.RapidOCR(
        params={
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )


def ranked_overlay(result):
    """Classify allowed popup text before considering generic button labels."""
    labels = {
        " ".join(re.findall(r"[a-z0-9]+", text.lower()))
        for text, score in zip(result.txts, result.scores)
        if score >= 0.95
    }
    if labels & {"rank up", "rank promotion"}:
        return "rank up"
    if "one more match" in labels and any(
        "how about another match of" in label for label in labels
    ):
        return "rematch confirmation"
    if labels & {"star up", "star promotion"}:
        return "star up"
    if any("daily quest" in label for label in labels):
        return "daily quest"
    if "reward received" in labels:
        if any("chest" in label for label in labels):
            return "chest reward"
        return None
    if "waiting for other players to confirm" in labels:
        return "result waiting"
    if any(re.fullmatch(r"match\s*end", label) for label in labels):
        return "match end"
    if labels & {"ron", "tsumo"} and "dora" in labels:
        return "hand result"
    if any(
        re.fullmatch(
            r"(?:east|south|west|north) ?[1-4] [0-9]+ ?repeat counter", label
        )
        for label in labels
    ):
        return "hand result"
    return None


def detect_controls(image, mode="postgame"):
    result = button_ocr()(image)
    targets = []
    if result.txts is None:
        return {"targets": []}
    error = None
    for text, score in zip(result.txts, result.scores):
        match = re.search(r"\berror\s+code\s*:\s*(\d+)\b", text, re.I)
        if score >= 0.95 and match:
            error = {
                "code": int(match[1]),
                "message": (
                    "Not enough coins to enter this room."
                    if int(match[1]) == 1304
                    else "Mahjong Soul reported a game error."
                ),
            }
            if mode != "ranked":
                return {"targets": [], "error": error}
            break
    navigation = {}
    overlay = ranked_overlay(result) if mode == "ranked" else None
    if overlay == "rank up":
        return {"targets": [], "screen": overlay}
    if mode == "ranked" and not error:
        header = " ".join(
            text.lower()
            for points, text, score in zip(
                result.boxes, result.txts, result.scores
            )
            if score >= 0.95
            and points[:, 0].min() >= 750
            and 140 <= points[:, 1].min() <= 190
        )
        if "back ranked" in header:
            navigation = {"screen": "ranked rooms"}
        else:
            for room in ("Bronze", "Silver", "Gold", "Jade", "Throne"):
                if f"back {room.lower()}" in header and "room" in header:
                    navigation = {"screen": "ranked modes", "room": room}
                    targets.append({"control": "back", "center": [816, 177]})
        if not navigation and "ranked match" in header:
            navigation = {"screen": "home"}
    for points, text, score in zip(result.boxes, result.txts, result.scores):
        label = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
        if (
            mode in {"postgame", "ranked"}
            and score >= 0.95
            and label == "reward received"
            and (mode != "ranked" or overlay in {"daily quest", "chest reward"})
        ):
            # This item overlay dismisses on a background click.
            return {
                **({"screen": overlay} if mode == "ranked" else {}),
                "targets": [
                    {
                        "control": "dismiss reward",
                        "center": [640, 560],
                        "box": [600, 540, 80, 40],
                        "score": float(score),
                        "verified_by": "reward-overlay-text",
                    }
                ],
            }
        x, y = points.min(axis=0).astype(int)
        right, bottom = points.max(axis=0).astype(int)
        if mode == "ranked" and score >= 0.95 and x >= 750:
            room = {
                "novice adept": "Bronze",
                "adept expert": "Silver",
                "expert master": "Gold",
                "master saint": "Jade",
                "saint above": "Throne",
            }.get(label)
            if (
                navigation.get("screen") == "ranked rooms"
                and room
                and 270 <= y <= 570
                and (image[y:bottom, x:right].max(axis=2) >= 200).mean() >= 0.02
            ):
                targets.append(
                    {
                        "control": f"{room.lower()} room",
                        "room": room,
                        "center": [930, int((y + bottom) // 2) - 45],
                    }
                )
                continue
            if (
                navigation.get("screen") == "home" and label == "ranked match"
            ) or (
                navigation.get("screen") == "ranked modes"
                and label in {"4 player east", "4 player south"}
                and 220 <= y <= 420
            ):
                # Modal overlays dim the buttons behind them.
                if (image[y:bottom, x:right].max(axis=2) >= 200).mean() >= 0.02:
                    targets.append(
                        {
                            "control": label,
                            "center": [
                                int((x + right) // 2),
                                int((y + bottom) // 2),
                            ],
                        }
                    )
                continue
        if mode == "ready" and not (
            label == "ready" and x >= 500 and right <= 1100 and y >= 550
        ):
            continue
        skip_scene = (
            mode in {"postgame", "ranked"}
            and label == "skip"
            and x >= 1050
            and bottom <= 110
        )
        if score < 0.95 or (
            not skip_scene
            and label
            not in {
                "ready": ("ready",),
                "cancel_vote": ("yes", "agree"),
                "postgame": REMATCH_LABELS + ("confirm", "next", "continue"),
                "ranked": REMATCH_LABELS
                + ("confirm", "next", "continue", "cancel"),
            }[mode]
        ):
            continue
        if mode == "ranked" and not error:
            if (
                (
                    not overlay
                    and not (
                        label in ("confirm", *REMATCH_LABELS)
                        and x >= 850
                        and y >= 600
                    )
                )
                or label == "skip"
                or (label == "cancel" and overlay != "rematch confirmation")
                or (
                    overlay == "rematch confirmation"
                    and (
                        label not in {"confirm", "cancel"}
                        or not (400 <= x <= 850 and 480 <= y <= 570)
                    )
                )
            ):
                continue
            if overlay in {"match end", "hand result"} and not (
                x >= 850 and y >= 600
            ):
                continue
        # Ready is green; postgame controls use gold or blue buttons.
        hsv = cv2.cvtColor(image[y:bottom, x:right], cv2.COLOR_BGR2HSV)
        gold = (
            (hsv[:, :, 0] >= 10)
            & (hsv[:, :, 0] <= (40 if mode in {"postgame", "ranked"} else 90))
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 150)
        )
        blue = (
            (hsv[:, :, 0] >= 100)
            & (hsv[:, :, 0] <= 130)
            & (hsv[:, :, 1] >= 80)
            & (hsv[:, :, 2] >= 100)
        )
        if not skip_scene and (gold | blue).mean() < 0.35:
            continue
        if mode == "ranked" and not overlay and not error:
            overlay = "match end"
        targets.append(
            {
                "control": label,
                "center": [int((x + right) // 2), int((y + bottom) // 2)],
                "box": [int(x), int(y), int(right - x), int(bottom - y)],
                "score": float(score),
                "verified_by": "button-text",
            }
        )
    if overlay == "match end":
        text = " ".join(result.txts).lower()
        navigation["stage"] = (
            "rematch"
            if any(label in text for label in REMATCH_LABELS)
            else (
                "rank progress"
                if re.search(r"\d+(?:\s*/)+\s*\d+", text)
                else "results"
            )
        )
    targets.sort(key=lambda target: target["control"] not in REMATCH_LABELS)
    return {
        "targets": targets,
        **navigation,
        **({"screen": overlay} if overlay else {}),
        **({"error": error} if error else {}),
    }
