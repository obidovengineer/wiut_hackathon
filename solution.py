"""
solution.py — the ONLY file the harness imports. Every real decision lives
in src/; this file just wires it up to the required interface.
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config
from src.detection import Detector
from src.events import (accident_and_near_miss, congestion,
                         failure_to_yield, jaywalking, red_light_and_stop_line,
                         stopped_vehicle, wrong_way)
from src.geometry import load_calibration
from src.light import light_state
from src.postprocess import EventAccumulator
from src.risk import RiskModel
from src.tracking import Tracker
from src.trajectory import TrajectoryStore

# Start with the subset the rule engine above actually supports well.
# illegal_u_turn / illegal_turn / solid_line_crossing need extra lane-marking
# calibration; road_obstacle / fire_smoke need an appearance-only detector
# with no tracking. Add each back to CLASSES only once src/events.py (or a
# new sibling module) implements it — never add ids the spec doesn't list.
CLASSES: list[str] = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "stop_line",
    "congestion",
]

RISK_HORIZON_SEC = config.RISK_HORIZON_SEC

_detector = Detector()  # loaded once, reused across videos in one process


def detect_events(video_path: str) -> list[list]:
    calib = load_calibration()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    tracker = Tracker(fps)
    store = TrajectoryStore()
    acc = EventAccumulator()

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if idx % config.DETECT_FRAME_STRIDE == 0:
            dets = _detector.infer(frame)
            tracks = tracker.update(dets, frame)
            store.update(t, tracks)
            light = light_state(frame, calib["light_roi"])

            active: dict[str, set] = {}
            active["wrong_way"] = wrong_way(store, calib)
            active["stopped_vehicle"] = stopped_vehicle(store, calib)
            active["congestion"] = congestion(store, calib)
            active["jaywalking"] = jaywalking(store, calib)
            active["failure_to_yield"] = failure_to_yield(store, calib)
            red_ids, stop_ids = red_light_and_stop_line(store, calib, light)
            active["red_light"] = red_ids
            active["stop_line"] = stop_ids
            accident_pairs, near_miss_pairs = accident_and_near_miss(store, calib)
            active["accident"] = accident_pairs
            active["near_miss"] = near_miss_pairs

            acc.step(t, active)
        idx += 1

    duration = idx / fps
    cap.release()
    events = acc.finalize()
    # clip end times to the actual duration, matching evaluate.py's rule
    return [[s, min(e, duration), label] for s, e, label in events]


class RiskEstimator:
    """Causal wrapper: reset() gets fresh state, step() only ever sees
    frames passed to it here, in order — never opens the video itself."""

    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.model = RiskModel(_detector, meta["fps"])

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self.model.step(frame, t_sec)