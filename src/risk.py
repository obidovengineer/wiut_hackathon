"""
risk.py — turns "smallest current pairwise TTC" into a smoothed P(accident
within RISK_HORIZON_SEC). Strictly causal: only ever sees frames it has
already been given, via the same TrajectoryStore/Detector/Tracker triplet
used by Part A, but as a fresh instance per video (see solution.py).
"""
from __future__ import annotations

from . import config
from .events import min_ttc_to_accident
from .trajectory import TrajectoryStore


def ttc_to_score(ttc: float, horizon: float = config.TTC_HORIZON_S) -> float:
    """1.0 at ttc=0, ~0.5 at ttc=horizon/2, ->0 as ttc grows past horizon."""
    if ttc == float("inf") or ttc > horizon * 2:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ttc / horizon))


class RiskModel:
    """Wraps Detector + Tracker + TrajectoryStore for streaming, per-frame
    use — kept separate from solution.RiskEstimator so it's independently
    unit-testable."""

    def __init__(self, detector, fps: float):
        from .tracking import Tracker  # local import: avoids a cycle at module load
        self.detector = detector
        self.tracker = Tracker(fps)
        self.store = TrajectoryStore()
        self.smoothed = 0.0

    def reset(self, fps: float):
        from .tracking import Tracker
        self.tracker = Tracker(fps)
        self.store = TrajectoryStore()
        self.smoothed = 0.0

    def step(self, frame, t_sec: float) -> float:
        dets = self.detector.infer(frame)
        tracks = self.tracker.update(dets, frame)
        self.store.update(t_sec, tracks)
        ttc = min_ttc_to_accident(self.store)
        instant = ttc_to_score(ttc)
        alpha = config.RISK_SMOOTHING
        self.smoothed = alpha * self.smoothed + (1 - alpha) * instant if instant < self.smoothed \
            else max(self.smoothed, instant)  # rise fast, decay smooth
        return self.smoothed