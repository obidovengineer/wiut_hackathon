
"""
light.py — reduces the calibrated light_roi crop to "red" / "green" /
"unknown" using HSV color mass. Deliberately simple: swap in a small
trained classifier later if this isn't reliable enough on the real footage,
without touching any rule module (they only see the string this returns).
"""
from __future__ import annotations

import cv2
import numpy as np

_RED_RANGES = [((0, 80, 80), (10, 255, 255)), ((170, 80, 80), (180, 255, 255))]
_GREEN_RANGE = ((40, 60, 60), (90, 255, 255))


def light_state(frame, light_roi) -> str:
    if not light_roi:
        return "unknown"
    x1, y1, x2, y2 = map(int, light_roi)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)).sum() for lo, hi in _RED_RANGES)
    green_mask = cv2.inRange(hsv, np.array(_GREEN_RANGE[0]), np.array(_GREEN_RANGE[1])).sum()
    if red_mask < 500 and green_mask < 500:
        return "unknown"
    return "red" if red_mask >= green_mask else "green"