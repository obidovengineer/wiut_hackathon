"""
calibrate.py — run ONCE, locally, against a sample frame from the fixed
camera. Not used at evaluation time and not imported by solution.py.

    python -m src.calibrate videos/sample1.mp4

Click points as prompted; press 'n' to finish the current shape, 'q' to quit
and save. Produces calibration/camera1.json with:

    stop_line:      [[x1,y1],[x2,y2]]              # a line segment
    lanes: [{"polygon": [[x,y],...], "direction": [dx,dy]}, ...]
    crossing_zone:  [[x,y], ...]                    # polygon (pedestrian crossing)
    light_roi:      [x1,y1,x2,y2]                   # crop containing the signal head
    no_uturn_zone:  [[x,y], ...]                    # optional polygon

This is deliberately a dumb click-and-save tool, not a product — you only
run it a handful of times.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from . import config

_current_points: list[list[int]] = []


def _on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _current_points.append([x, y])


def _collect_shape(frame, window, prompt) -> list[list[int]]:
    global _current_points
    _current_points = []
    cv2.setMouseCallback(window, _on_click)
    print(prompt, "(click points, 'n' when done)")
    while True:
        disp = frame.copy()
        for p in _current_points:
            cv2.circle(disp, tuple(p), 4, (0, 0, 255), -1)
        if len(_current_points) >= 2:
            cv2.polylines(disp, [__import__("numpy").array(_current_points)], False, (0, 255, 0), 2)
        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("n"):
            return list(_current_points)
        if key == ord("q"):
            sys.exit(0)


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not video_path:
        print("usage: python -m src.calibrate <video.mp4>")
        return
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {video_path}")

    window = "calibrate (n=finish shape, q=quit/save)"
    cv2.namedWindow(window)

    stop_line = _collect_shape(frame, window, "STOP LINE: click 2 points along the stop line")

    lanes = []
    while True:
        poly = _collect_shape(frame, window, "LANE polygon (empty + 'n' to stop adding lanes)")
        if not poly:
            break
        direction = _collect_shape(frame, window, "LANE direction: click tail then head of an arrow along traffic flow")
        dx = direction[-1][0] - direction[0][0] if len(direction) >= 2 else 0
        dy = direction[-1][1] - direction[0][1] if len(direction) >= 2 else 0
        lanes.append({"polygon": poly, "direction": [dx, dy]})

    crossing_zone = _collect_shape(frame, window, "PEDESTRIAN CROSSING polygon")
    light_roi_pts = _collect_shape(frame, window, "TRAFFIC LIGHT: click top-left then bottom-right of its bounding box")
    light_roi = []
    if len(light_roi_pts) >= 2:
        (x1, y1), (x2, y2) = light_roi_pts[0], light_roi_pts[-1]
        light_roi = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    cv2.destroyAllWindows()

    data = {
        "stop_line": stop_line,
        "lanes": lanes,
        "crossing_zone": crossing_zone,
        "light_roi": light_roi,
        "no_uturn_zone": [],
        "frame_size": [frame.shape[1], frame.shape[0]],
    }
    config.CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CALIBRATION_PATH.write_text(json.dumps(data, indent=2))
    print(f"wrote {config.CALIBRATION_PATH}")


if __name__ == "__main__":
    main()