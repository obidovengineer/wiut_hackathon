from __future__ import annotations

import numpy as np
from boxmot import ByteTrack

from . import config


class Tracker:
    def __init__(self, fps: float):
        self.tracker = ByteTrack(
            track_thresh=config.TRACK_THRESH,
            track_buffer=config.TRACK_BUFFER,
            match_thresh=config.MATCH_THRESH,
            frame_rate=fps,
        )

    def update(self, dets: np.ndarray, frame: np.ndarray) -> list[dict]:
        if dets.shape[0] == 0:
            dets = np.zeros((0, 6), dtype=np.float32)
        tracks = self.tracker.update(dets, frame)
        out = []
        for t in tracks:
            x1, y1, x2, y2, track_id, conf, cls = t[:7]
            out.append({
                "id": int(track_id),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "conf": float(conf),
                "cls": int(cls),
            })
        return out