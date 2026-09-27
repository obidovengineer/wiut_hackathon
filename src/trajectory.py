from __future__ import annotations

from collections import deque

from . import config
from .geometry import bbox_center


class TrackState:
    __slots__ = ("id", "cls", "positions", "times", "bbox", "first_t",
                 "last_t", "stopped_since", "last_stopped")

    def __init__(self, track_id: int, cls: int, t: float, bbox):
        self.id = track_id
        self.cls = cls
        self.positions = deque(maxlen=config.HISTORY_LEN)
        self.times = deque(maxlen=config.HISTORY_LEN)
        self.bbox = bbox
        self.first_t = t
        self.last_t = t
        self.stopped_since: float | None = None
        self.last_stopped = False
        self._push(t, bbox)

    def _push(self, t, bbox):
        self.positions.append(bbox_center(bbox))
        self.times.append(t)
        self.bbox = bbox
        self.last_t = t

    def update(self, t: float, bbox):
        self._push(t, bbox)

    def velocity(self):
        """px/s over the last ~0.3s window, robust to single-frame jitter."""
        if len(self.positions) < 2:
            return (0.0, 0.0)
        # find a sample far enough back in time
        i = len(self.positions) - 1
        j = i
        while j > 0 and self.times[i] - self.times[j] < 0.3:
            j -= 1
        dt = self.times[i] - self.times[j]
        if dt <= 1e-6:
            return (0.0, 0.0)
        (x1, y1), (x2, y2) = self.positions[j], self.positions[i]
        return ((x2 - x1) / dt, (y2 - y1) / dt)

    def speed(self) -> float:
        vx, vy = self.velocity()
        return (vx ** 2 + vy ** 2) ** 0.5

    def acceleration(self):
        """Rough px/s^2 using two half-windows; used for hard-braking detection."""
        if len(self.times) < 3:
            return 0.0
        mid = len(self.positions) // 2
        (x0, y0) = self.positions[0]
        (xm, ym) = self.positions[mid]
        (x1, y1) = self.positions[-1]
        t0, tm, t1 = self.times[0], self.times[mid], self.times[-1]
        if tm - t0 < 1e-6 or t1 - tm < 1e-6:
            return 0.0
        v_early = ((xm - x0) ** 2 + (ym - y0) ** 2) ** 0.5 / (tm - t0)
        v_late = ((x1 - xm) ** 2 + (y1 - ym) ** 2) ** 0.5 / (t1 - tm)
        return (v_late - v_early) / (t1 - tm)

    def dwell_time(self) -> float:
        """How long this track has been below the 'stopped' speed threshold."""
        if self.speed() < config.STOPPED_SPEED_PX_S:
            if self.stopped_since is None:
                self.stopped_since = self.last_t
            return self.last_t - self.stopped_since
        self.stopped_since = None
        return 0.0


class TrajectoryStore:
    def __init__(self):
        self.tracks: dict[int, TrackState] = {}
        # (id_a, id_b) sorted -> consecutive frames the IoU contact gate has
        # been true. Lives here (not in events.py) because events.py's rule
        # functions are stateless — called fresh every frame — while "has
        # this overlap persisted N frames" needs memory across frames.
        self.pair_contact_frames: dict[tuple, int] = {}

    def note_pair_contact(self, key: tuple, in_contact: bool) -> int:
        """Call once per frame per pair being evaluated. Returns the updated
        consecutive-contact count (0 if not currently in contact)."""
        if in_contact:
            self.pair_contact_frames[key] = self.pair_contact_frames.get(key, 0) + 1
        else:
            self.pair_contact_frames.pop(key, None)
        return self.pair_contact_frames.get(key, 0)

    def update(self, t: float, detections: list[dict]) -> list[TrackState]:
        """detections: output of Tracker.update() for this frame."""
        seen = set()
        for d in detections:
            tid = d["id"]
            seen.add(tid)
            if tid not in self.tracks:
                self.tracks[tid] = TrackState(tid, d["cls"], t, d["bbox"])
            else:
                self.tracks[tid].update(t, d["bbox"])
        # drop tracks not seen this frame from the "active" view, but keep
        # their state a little longer in case of brief occlusion
        stale = [tid for tid, tr in self.tracks.items()
                 if t - tr.last_t > 2.0]
        for tid in stale:
            del self.tracks[tid]
        return [self.tracks[tid] for tid in seen]

    def active(self) -> list[TrackState]:
        return list(self.tracks.values())


def time_to_collision(a: TrackState, b: TrackState,
                       horizon: float = config.TTC_HORIZON_S,
                       min_safe_lateral: float = config.MIN_SAFE_LATERAL_PX) -> float:
    """Collision-course TTC, not just closing-speed TTC.

    The old version returned a small TTC for ANY two tracks getting nearer
    for now — including two cars converging on an intersection but passing
    safely, or two cars in adjacent lanes merging distance-wise while still
    staying a lane-width apart. That's why accidents were firing on almost
    every frame of a busy scene.

    Fix: extrapolate both tracks forward at constant velocity and find the
    time t* that MINIMIZES their separation (classic closest-point-of-
    approach). If that minimum separation is still larger than roughly one
    vehicle width (min_safe_lateral), they are not actually headed for the
    same point in space — return inf regardless of how fast they're
    currently closing.
    """
    if len(a.positions) == 0 or len(b.positions) == 0:
        return float("inf")
    ax, ay = a.positions[-1]
    bx, by = b.positions[-1]
    avx, avy = a.velocity()
    bvx, bvy = b.velocity()

    # relative position and velocity of b with respect to a
    rx, ry = bx - ax, by - ay
    rvx, rvy = bvx - avx, bvy - avy
    rel_speed2 = rvx ** 2 + rvy ** 2

    if rel_speed2 < 1e-6:
        # not closing at all (parallel/equal velocity) -> never converges
        return float("inf")

    # t* = argmin_t |r + rv*t|^2, clamped to [0, horizon] since we only
    # trust the constant-velocity extrapolation a few seconds out
    t_star = -(rx * rvx + ry * rvy) / rel_speed2
    t_star = max(0.0, min(horizon, t_star))

    closest_x = rx + rvx * t_star
    closest_y = ry + rvy * t_star
    closest_dist = (closest_x ** 2 + closest_y ** 2) ** 0.5

    if closest_dist > min_safe_lateral:
        return float("inf")  # closing for now, but not headed for the same point
    if t_star <= 1e-3:
        return 0.0  # already at/near closest point — i.e. contact is now
    return t_star