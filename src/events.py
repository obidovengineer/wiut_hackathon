from __future__ import annotations

from . import config
from .geometry import (bbox_iou, heading_alignment, point_in_polygon,
                        side_of_line, which_lane)
from .trajectory import TrackState, TrajectoryStore, time_to_collision


def wrong_way(store: TrajectoryStore, calib: dict) -> set:
    active = set()
    for tr in store.active():
        if tr.cls not in config.VEHICLE_CLASSES or tr.speed() < config.STOPPED_SPEED_PX_S:
            continue
        lane_idx = which_lane(tr.positions[-1], calib["lanes"])
        if lane_idx is None:
            continue
        align = heading_alignment(tr.velocity(), calib["lanes"][lane_idx]["direction"])
        if align < -0.3:  # moving mostly opposite to declared lane direction
            active.add(tr.id)
    return active


def stopped_vehicle(store: TrajectoryStore, calib: dict) -> set:
    active = set()
    for tr in store.active():
        if tr.cls not in config.VEHICLE_CLASSES:
            continue
        if tr.dwell_time() >= config.STOPPED_DURATION_S:
            active.add(tr.id)
    return active


def congestion(store: TrajectoryStore, calib: dict) -> set:
    vehicles = [tr for tr in store.active() if tr.cls in config.VEHICLE_CLASSES]
    if len(vehicles) < config.CONGESTION_MIN_VEHICLES:
        return set()
    stopped = sum(1 for tr in vehicles if tr.dwell_time() > 0)
    if stopped / len(vehicles) >= config.CONGESTION_MIN_STOPPED_FRACTION:
        return {"scene"}  # scene-level event, not per-track
    return set()


def jaywalking(store: TrajectoryStore, calib: dict) -> set:
    active = set()
    for tr in store.active():
        if tr.cls != config.PERSON_CLASS:
            continue
        on_crossing = point_in_polygon(tr.positions[-1], calib["crossing_zone"])
        on_any_lane = which_lane(tr.positions[-1], calib["lanes"]) is not None
        if on_any_lane and not on_crossing:
            active.add(tr.id)
    return active


def failure_to_yield(store: TrajectoryStore, calib: dict) -> set:
    peds_on_crossing = [tr for tr in store.active()
                         if tr.cls == config.PERSON_CLASS
                         and point_in_polygon(tr.positions[-1], calib["crossing_zone"])]
    if not peds_on_crossing:
        return set()
    active = set()
    for tr in store.active():
        if tr.cls not in config.VEHICLE_CLASSES:
            continue
        if point_in_polygon(tr.positions[-1], calib["crossing_zone"]) and tr.speed() > config.STOPPED_SPEED_PX_S:
            active.add(tr.id)
    return active


def red_light_and_stop_line(store: TrajectoryStore, calib: dict, light: str) -> tuple[set, set]:
    """Returns (red_light_active_ids, stop_line_active_ids). Both need the
    signal to be red and the vehicle to be past/at the stop line."""
    red_ids, stop_ids = set(), set()
    if light != "red" or not calib["stop_line"]:
        return red_ids, stop_ids
    for tr in store.active():
        if tr.cls not in config.VEHICLE_CLASSES:
            continue
        side = side_of_line(tr.positions[-1], calib["stop_line"])
        past_line = side < 0  # sign convention fixed once during calibration review
        if not past_line:
            continue
        if tr.speed() < config.STOPPED_SPEED_PX_S:
            stop_ids.add(tr.id)
        else:
            red_ids.add(tr.id)
    return red_ids, stop_ids


def accident_and_near_miss(store: TrajectoryStore, calib: dict) -> tuple[set, set]:
    """Pairwise TTC + bbox overlap. Returns (accident_pair_keys, near_miss_pair_keys).

    Two changes from the naive version that was firing on nearly every
    frame of a busy scene:
      1. IoU contact must PERSIST for CONTACT_MIN_CONSEC_FRAMES before it
         counts as an accident (tracker jitter / boxes brushing for one
         frame no longer qualifies) — tracked via store.note_pair_contact.
      2. near_miss uses the collision-course TTC (see trajectory.py), which
         already excludes tracks that are merely closing distance without
         being headed for the same point in space.
    """
    vehicles = [tr for tr in store.active() if tr.cls in config.VEHICLE_CLASSES]
    accident, near_miss = set(), set()
    seen_pairs = set()
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            a, b = vehicles[i], vehicles[j]
            key = tuple(sorted((a.id, b.id)))
            seen_pairs.add(key)
            iou = bbox_iou(a.bbox, b.bbox)
            in_contact = iou >= config.IOU_CONTACT_THRESH
            consec = store.note_pair_contact(key, in_contact)
            if consec >= config.CONTACT_MIN_CONSEC_FRAMES:
                accident.add(key)
                continue
            ttc = time_to_collision(a, b)
            hard_brake = a.acceleration() < -config.HARD_BRAKE_DECEL_PX_S2 or \
                         b.acceleration() < -config.HARD_BRAKE_DECEL_PX_S2
            if ttc < config.TTC_NEAR_MISS_S and hard_brake:
                near_miss.add(key)
    # pairs that existed before but neither vehicle is tracked anymore this
    # frame still need their contact counter cleared, or a stale count could
    # resurrect a false accident if two unrelated tracks later reuse the
    # same (id_a, id_b) key combination
    for key in list(store.pair_contact_frames.keys()):
        if key not in seen_pairs:
            store.note_pair_contact(key, False)
    return accident, near_miss


def min_ttc_to_accident(store: TrajectoryStore) -> float:
    """Used by RiskEstimator: the single smallest TTC among all vehicle
    pairs right now, for turning into a risk score."""
    vehicles = [tr for tr in store.active() if tr.cls in config.VEHICLE_CLASSES]
    best = float("inf")
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            ttc = time_to_collision(vehicles[i], vehicles[j])
            best = min(best, ttc)
    return best