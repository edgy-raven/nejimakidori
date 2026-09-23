#!/usr/bin/env python3

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "datasets" / "tile_classify"
MANIFEST_PATH = DATASET_ROOT / "manifest.json"
PORTRAIT_SIZE = (80, 126)
ROTATIONS = ("none", "cw90", "ccw90", "180")


@dataclass
class ReferenceTile:
    label: str
    source: str
    path: str
    image: np.ndarray
    gray: np.ndarray
    edge: np.ndarray
    mask: np.ndarray
    color: np.ndarray


def rotate_image(image_bgr, rotation):
    if rotation == "none":
        return image_bgr
    if rotation == "cw90":
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw90":
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(image_bgr, cv2.ROTATE_180)
    raise ValueError(f"unknown rotation: {rotation}")


def fit_to_canvas(image_bgr, size=PORTRAIT_SIZE, fill=255):
    target_w, target_h = size
    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), fill, dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(
        image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC
    )
    canvas = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def foreground_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.where((hsv[:, :, 2] < 238) | (hsv[:, :, 1] > 28), 255, 0)
    mask = mask.astype(np.uint8)
    mask = cv2.medianBlur(mask, 3)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def color_descriptor(image_bgr, mask):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    keep = mask > 0
    if not keep.any():
        return np.zeros(12, dtype=np.float32)

    hue_hist = cv2.calcHist(
        [hsv],
        [0],
        mask,
        [8],
        [0, 180],
    ).reshape(-1)
    sat_hist = cv2.calcHist(
        [hsv],
        [1],
        mask,
        [4],
        [0, 256],
    ).reshape(-1)
    row = np.concatenate([hue_hist, sat_hist]).astype(np.float32)
    denom = float(row.sum())
    if denom:
        row /= denom
    return row


def normalize_tile(image_bgr):
    image_bgr = fit_to_canvas(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = foreground_mask(image_bgr)
    edge = cv2.Canny(gray, 60, 170)
    edge = cv2.dilate(edge, np.ones((2, 2), np.uint8), iterations=1)
    return {
        "image": image_bgr,
        "gray": gray.astype(np.float32) / 255.0,
        "edge": edge.astype(np.float32) / 255.0,
        "mask": mask.astype(np.float32) / 255.0,
        "color": color_descriptor(image_bgr, mask),
    }


def reference_from_image(label, source, path, image_bgr):
    normalized = normalize_tile(image_bgr)
    return ReferenceTile(
        label=label,
        source=source,
        path=path,
        image=normalized["image"],
        gray=normalized["gray"],
        edge=normalized["edge"],
        mask=normalized["mask"],
        color=normalized["color"],
    )


def score_reference(candidate, reference):
    mask_weight = np.maximum(candidate["mask"], reference.mask)
    if not mask_weight.any():
        mask_weight = np.ones_like(mask_weight)

    gray_delta = np.abs(candidate["gray"] - reference.gray)
    gray_score = 1.0 - float(
        (gray_delta * mask_weight).sum() / mask_weight.sum()
    )
    edge_score = 1.0 - float(np.abs(candidate["edge"] - reference.edge).mean())
    mask_score = 1.0 - float(np.abs(candidate["mask"] - reference.mask).mean())
    color_score = 1.0 - float(
        np.abs(candidate["color"] - reference.color).mean()
    )
    image_delta = np.abs(
        candidate["image"].astype(np.float32) - reference.image
    )
    image_score = 1.0 - float(image_delta.mean() / 255.0)
    score = (
        0.44 * image_score
        + 0.24 * gray_score
        + 0.16 * edge_score
        + 0.12 * mask_score
        + 0.04 * color_score
    )
    return float(score)


@lru_cache(maxsize=1)
def load_reference_tiles():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    references = []
    for row in manifest:
        if row["split"] != "train":
            continue
        image_path = DATASET_ROOT / row["path"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        references.append(
            reference_from_image(
                row["label"],
                row["source"],
                row["path"],
                image_bgr,
            )
        )
    if not references:
        raise ValueError(f"no training references found in {MANIFEST_PATH}")
    return tuple(references)


@lru_cache(maxsize=1)
def atlas_reference_tiles():
    return tuple(
        reference
        for reference in load_reference_tiles()
        if reference.source == "atlas"
    )


def score_candidate(
    candidate_bgr,
    allowed_rotations=ROTATIONS,
    references=None,
    include_scores=False,
):
    if references is None:
        references = load_reference_tiles()

    best_by_label = {}
    for rotation in allowed_rotations:
        candidate = normalize_tile(rotate_image(candidate_bgr, rotation))
        for reference in references:
            score = score_reference(candidate, reference)
            prior = best_by_label.get(reference.label)
            if prior is None or score > prior["score"]:
                best_by_label[reference.label] = {
                    "label": reference.label,
                    "score": score,
                    "rotation": rotation,
                    "reference": reference.path,
                    "source": reference.source,
                }

    rows = sorted(best_by_label.values(), key=lambda row: -row["score"])
    top = rows[0]
    second = rows[1]
    result = {
        "label": top["label"],
        "method": "labelled-nearest",
        "confidence": round(max(0.0, min(1.0, top["score"])), 4),
        "margin": round(top["score"] - second["score"], 4),
        "rotation_correction": top["rotation"],
        "scale_correction": 1.0,
        "reference": top["reference"],
        "top3": [
            {
                "label": row["label"],
                "score": round(row["score"], 4),
                "reference": row["reference"],
            }
            for row in rows[:3]
        ],
    }
    if include_scores:
        result["scores"] = {
            row["label"]: round(row["score"], 4) for row in rows
        }
    return result
