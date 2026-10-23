#!/usr/bin/env python3

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RIVER_ANNOTATIONS_PATH = ROOT / "static" / "river_tiles.json"

ROW_INDEX = {"a": 0, "b": 1, "c": 2}
ROW_NAME = {value: key for key, value in ROW_INDEX.items()}
FULL_LABEL_PATTERN = re.compile(
    r"^river_(bottom|top|left|right)_([abc])(\d+)_([a-z0-9]+?)(?:(_riichi))?$"
)
GENERIC_LABEL_PATTERN = re.compile(
    r"^river_tile_([abc])(\d+)_([a-z0-9]+?)(?:(_riichi))?$"
)
UPRIGHT_ORDERS = {
    "bottom": [0, 1, 2, 3],
    "top": [2, 3, 0, 1],
    "left": [1, 2, 3, 0],
    "right": [3, 0, 1, 2],
}
PORTRAIT_SIZE = (80, 126)


@dataclass
class RiverAnnotation:
    id: str
    raw_label: str
    player: str | None
    row: int
    col: int
    expected_label: str
    riichi: bool
    points: np.ndarray
    center: np.ndarray
    inferred_player: bool = False
    assignment_distance: float | None = None


def slot_name(row, col):
    return f"{ROW_NAME[row]}{col}"


def load_river_annotations():
    payload = json.loads(RIVER_ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    explicit = []
    generic = []

    for shape in payload.get("shapes", []):
        raw_label = str(shape.get("label") or "")
        points = np.array(
            [[point["x"], point["y"]] for point in shape["points"]],
            dtype=np.float32,
        )
        center = points.mean(axis=0)

        match = FULL_LABEL_PATTERN.match(raw_label)
        if match:
            player, row_name, col, tile_label, riichi = match.groups()
            explicit.append(
                RiverAnnotation(
                    id=str(shape.get("id") or raw_label),
                    raw_label=raw_label,
                    player=player,
                    row=ROW_INDEX[row_name],
                    col=int(col),
                    expected_label=tile_label,
                    riichi=bool(riichi),
                    points=points,
                    center=center,
                )
            )
            continue

        match = GENERIC_LABEL_PATTERN.match(raw_label)
        if match:
            row_name, col, tile_label, riichi = match.groups()
            generic.append(
                RiverAnnotation(
                    id=str(shape.get("id") or raw_label),
                    raw_label=raw_label,
                    player=None,
                    row=ROW_INDEX[row_name],
                    col=int(col),
                    expected_label=tile_label,
                    riichi=bool(riichi),
                    points=points,
                    center=center,
                )
            )

    return explicit, generic


def sequence_index(row, col):
    return row * 6 + (col - 1)


def logical_cell_corners(row, col):
    return np.array(
        [
            [col - 1.0, row],
            [col * 1.0, row],
            [col * 1.0, row + 1.0],
            [col - 1.0, row + 1.0],
        ],
        dtype=np.float32,
    )


def ordered_observed_corners(annotation):
    return np.array(
        [
            annotation.points[index]
            for index in UPRIGHT_ORDERS[annotation.player]
        ],
        dtype=np.float32,
    )


def assign_generic_players(explicit, generic):
    grouped = defaultdict(list)
    for annotation in explicit:
        grouped[annotation.player].append(annotation.center)

    centroids = {
        player: np.mean(np.stack(points), axis=0)
        for player, points in grouped.items()
    }

    assigned = []
    for annotation in generic:
        player = min(
            centroids,
            key=lambda candidate: float(
                np.linalg.norm(annotation.center - centroids[candidate])
            ),
        )
        assigned.append(
            RiverAnnotation(
                id=annotation.id,
                raw_label=annotation.raw_label,
                player=player,
                row=annotation.row,
                col=annotation.col,
                expected_label=annotation.expected_label,
                riichi=annotation.riichi,
                points=annotation.points,
                center=annotation.center,
                inferred_player=True,
                assignment_distance=float(
                    np.linalg.norm(annotation.center - centroids[player])
                ),
            )
        )
    return assigned


def group_by_player(annotations):
    grouped = defaultdict(list)
    for annotation in annotations:
        grouped[annotation.player].append(annotation)
    return dict(sorted(grouped.items()))


def fit_homography_from_cells(annotations):
    fit_annotations = [
        annotation for annotation in annotations if not annotation.riichi
    ]
    if not fit_annotations:
        raise ValueError(
            "no non-riichi annotations available for homography fit"
        )

    src_points = []
    dst_points = []
    for annotation in fit_annotations:
        src_points.extend(
            logical_cell_corners(annotation.row, annotation.col).tolist()
        )
        dst_points.extend(ordered_observed_corners(annotation).tolist())

    src_points = np.array(src_points, dtype=np.float32).reshape((-1, 1, 2))
    dst_points = np.array(dst_points, dtype=np.float32).reshape((-1, 1, 2))

    homography, _ = cv2.findHomography(src_points, dst_points, 0)
    if homography is None:
        raise ValueError("could not fit homography")

    projected = cv2.perspectiveTransform(src_points, homography).reshape(
        (-1, 2)
    )
    errors = np.linalg.norm(projected - dst_points.reshape((-1, 2)), axis=1)
    return {
        "homography": homography,
        "mean_reprojection_error": float(errors.mean()),
        "max_reprojection_error": float(errors.max()),
        "fit_count": len(fit_annotations),
        "point_count": int(len(src_points)),
    }


def project_cell_quad(homography, row, col):
    corners = logical_cell_corners(row, col).reshape((-1, 1, 2))
    return cv2.perspectiveTransform(corners, homography).reshape((-1, 2))


def ordered_destination(size):
    width, height = size
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def warp_from_points(scene, ordered_points, size):
    width, height = size
    matrix = cv2.getPerspectiveTransform(
        ordered_points.astype(np.float32), ordered_destination(size)
    )
    warped = cv2.warpPerspective(
        scene,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def find_riichi_annotation(annotations):
    riichi_annotations = [
        annotation for annotation in annotations if annotation.riichi
    ]
    if not riichi_annotations:
        return None
    return min(
        riichi_annotations,
        key=lambda annotation: sequence_index(annotation.row, annotation.col),
    )


def fit_player_projection(player, annotations):
    riichi = find_riichi_annotation(annotations)
    all_fit = fit_homography_from_cells(annotations)
    projection = {
        "player": player,
        "riichi_slot": (
            slot_name(riichi.row, riichi.col) if riichi is not None else None
        ),
        "all": all_fit,
        "pre_riichi": None,
        "post_riichi": None,
    }

    if riichi is None:
        return projection

    riichi_index = sequence_index(riichi.row, riichi.col)
    pre_annotations = [
        annotation
        for annotation in annotations
        if sequence_index(annotation.row, annotation.col) < riichi_index
    ]
    post_annotations = [
        annotation
        for annotation in annotations
        if sequence_index(annotation.row, annotation.col) > riichi_index
    ]

    projection["pre_riichi"] = (
        fit_homography_from_cells(pre_annotations) if pre_annotations else None
    )
    projection["post_riichi"] = (
        fit_homography_from_cells(post_annotations)
        if post_annotations
        else None
    )
    return projection
