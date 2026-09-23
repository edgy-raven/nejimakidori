"""Verify call panels against protocol options using upright tile references."""

import json
import sys

import cv2
import numpy

from vision import tile_matcher


def candidates(image, options):
    results = []
    for option in options:
        x, y, width, height = option["box"]
        panel = image[y : y + height, x : x + width]
        mask = numpy.uint8(numpy.min(panel, axis=-1) > 135) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, numpy.ones((7, 5), numpy.uint8)
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        faces = []
        for contour in contours:
            left, top, face_width, face_height = cv2.boundingRect(contour)
            if face_height < 45 or face_width < 20:
                continue
            count = max(1, round(face_width / (face_height * 0.64)))
            for index in range(count):
                start = left + round(face_width * index / count)
                end = left + round(face_width * (index + 1) / count)
                face = panel[top : top + face_height, start:end]
                match = tile_matcher.score_candidate(face)
                faces.append((start, match))
        faces.sort(key=lambda value: value[0])
        labels = [match["label"] for _, match in faces]
        expected = [
            (
                ("east", "south", "west", "north", "white", "green", "red")[
                    int(tile[0]) - 1
                ]
                if tile.endswith("z")
                else tile.replace("0", "5")
            )
            for tile in option["tiles"]
        ]
        # Closed quads may expose only their middle two faces.
        matches = (
            sorted(labels) == sorted(expected)
            or option["type"] in (4, 6)
            and 2 <= len(labels) <= 4
            and len(set(expected + labels)) == 1
        )
        trusted = matches and all(
            match["confidence"] >= 0.7 and match["margin"] >= 0.02
            for _, match in faces
        )
        results.append(
            {"center": option["center"], "labels": labels, "trusted": trusted}
        )
    return {"options": results, "trusted": all(x["trusted"] for x in results)}


if __name__ == "__main__":
    request = json.load(sys.stdin)
    image = cv2.imread(request["image"])
    if image is None or image.shape[:2] != (720, 1280):
        raise ValueError("Call selection needs a 1280x720 screenshot")
    print(json.dumps(candidates(image, request["options"])))
