from __future__ import annotations

import json
import math

from . import config


def load_calibration() -> dict:
    return json.loads(config.CALIBRATION_PATH.read_text())


def point_in_polygon(pt, polygon) -> bool:
    """Standard ray-casting test. polygon: list of [x, y]."""
    if len(polygon) < 3:
        return False
    x, y = pt
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside


def side_of_line(pt, line):
    """Signed distance-ish value: >0 on one side, <0 on the other, 0 on it.
    line: [[x1,y1],[x2,y2]]."""
    (x1, y1), (x2, y2) = line
    x, y = pt
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def heading_alignment(velocity, lane_direction) -> float:
    """Cosine similarity in [-1, 1] between a track's velocity vector and
    the lane's declared traffic direction. -1 means driving the wrong way."""
    vx, vy = velocity
    dx, dy = lane_direction
    v_norm = math.hypot(vx, vy)
    d_norm = math.hypot(dx, dy)
    if v_norm < 1e-6 or d_norm < 1e-6:
        return 0.0
    return (vx * dx + vy * dy) / (v_norm * d_norm)


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def which_lane(pt, lanes: list[dict]) -> int | None:
    for i, lane in enumerate(lanes):
        if point_in_polygon(pt, lane["polygon"]):
            return i
    return None