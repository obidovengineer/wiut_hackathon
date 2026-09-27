"""
postprocess.py — turns {t: {label: {active_keys}}} accumulated over a video
into the final [start_sec, end_sec, label] list: one instance-key runs
become one segment, short gaps are bridged, sub-threshold blips are dropped.
"""
from __future__ import annotations

from collections import defaultdict

from . import config


class EventAccumulator:
    """Call `mark(t, label, key)` every frame an instance is active; call
    `finalize()` once at the end of the video."""

    def __init__(self):
        # per (label, key) -> list of [start, end] open/closed intervals
        self._open: dict[tuple, float] = {}
        self._closed: list[tuple] = []  # (start, end, label)
        self._last_t: dict[tuple, float] = {}

    def mark(self, t: float, label: str, key) -> None:
        k = (label, key)
        if k not in self._open:
            self._open[k] = t
        self._last_t[k] = t

    def step(self, t: float, active: dict[str, set]) -> None:
        """active: {label: {keys active this frame}}. Call once per frame
        with every label's active set, including empty sets, so closed
        intervals get flushed."""
        seen_this_frame = set()
        for label, keys in active.items():
            for key in keys:
                self.mark(t, label, key)
                seen_this_frame.add((label, key))
        # close any open interval not renewed this frame
        for k in list(self._open.keys()):
            if k not in seen_this_frame and t - self._last_t[k] > 1e-6:
                start = self._open.pop(k)
                end = self._last_t[k]
                label, _ = k
                if end > start:
                    self._closed.append((start, end, label))
                del self._last_t[k]

    def finalize(self) -> list[list]:
        for k, start in self._open.items():
            end = self._last_t.get(k, start)
            label, _ = k
            if end > start:
                self._closed.append((start, end, label))
        self._open.clear()
        return merge_and_clean(self._closed)


def merge_and_clean(segments: list[tuple],
                     max_gap=config.MAX_MERGE_GAP_S,
                     min_dur=config.MIN_SEGMENT_DURATION_S) -> list[list]:
    """segments: (start, end, label) tuples, ANY order, possibly from many
    instance keys of the same label (already union'd — same-class overlap
    is legal input here; the harness itself drops same-class overlaps in
    the OUTPUT, so we merge first to avoid losing detections to that rule)."""
    by_label = defaultdict(list)
    for s, e, label in segments:
        by_label[label].append([s, e])

    out = []
    for label, spans in by_label.items():
        spans.sort()
        merged = []
        for s, e in spans:
            if merged and s - merged[-1][1] <= max_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            if e - s >= min_dur:
                out.append([round(s, 3), round(e, 3), label])
    out.sort()
    return out