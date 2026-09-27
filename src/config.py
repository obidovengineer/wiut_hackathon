"""
config.py — every tunable number lives here so rule tuning never means
hunting through five files.
"""
from pathlib import Path

# ---- paths -----------------------------------------------------------
WEIGHTS_DIR = Path("weights")
YOLO_WEIGHTS = WEIGHTS_DIR / "yolo26s.pt"
CALIBRATION_PATH = Path("calibration/camera1.json")  # one fixed camera -> one file

# ---- detection ---------------------------------------------------------
DETECT_CONF = 0.25
DETECT_IOU = 0.5
# COCO ids: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]
VEHICLE_CLASSES = {1, 2, 3, 5, 7}
PERSON_CLASS = 0
# how many frames to skip between YOLO calls in detect_events (Part A can
# afford this; RiskEstimator has its own, tighter budget below)
DETECT_FRAME_STRIDE = 3
# frames to batch into one YOLO forward call in detect_events — batching
# amortizes Python/model call overhead across many frames instead of
# paying it per single frame
DETECT_BATCH_SIZE = 16

# ---- tracking (ByteTrack) ---------------------------------------------
TRACK_THRESH = 0.3
TRACK_BUFFER = 40
MATCH_THRESH = 0.8

# ---- trajectory / motion ------------------------------------------------
HISTORY_LEN = 150          # frames of position/time history kept per track
MIN_MOVE_PIXELS = 5        # ignore jitter below this before logging a point
STOPPED_SPEED_PX_S = 8.0   # px/s below this counts as "not moving"
STOPPED_DURATION_S = 10.0  # -> stopped_vehicle threshold from the task spec

# ---- congestion ---------------------------------------------------------
CONGESTION_MIN_STOPPED_FRACTION = 0.6   # fraction of tracked vehicles stopped
CONGESTION_MIN_VEHICLES = 3

# ---- red light / stop line ------------------------------------------------
RED_LIGHT_GRACE_S = 0.3     # ignore stop-line crossings within this long after amber->red

# ---- near miss / accident (TTC-based) -----------------------------------
TTC_HORIZON_S = 5.0          # matches RISK_HORIZON_SEC in solution.py
TTC_NEAR_MISS_S = 1.5        # TTC below this + on a collision course -> near_miss
HARD_BRAKE_DECEL_PX_S2 = 400 # heuristic deceleration threshold for evasive action

# was 0.15 — far too low for a crowded intersection: adjacent-lane cars and a
# pedestrian near a car's bbox routinely overlap that much with no contact.
IOU_CONTACT_THRESH = 0.40

# a single-frame IoU spike (tracker jitter, boxes brushing at a red light)
# must NOT count as an accident; require it to persist across frames.
CONTACT_MIN_CONSEC_FRAMES = 5

# TTC alone is not enough: two cars converging on an intersection but passing
# safely, or two cars in adjacent lanes closing distance while merging, look
# identical to TTC if you only check closing speed. Require the PREDICTED
# closest-approach distance (constant-velocity extrapolation) to be small
# too, i.e. the tracks are actually headed for the same point in space, not
# just getting nearer for now. Unit: pixels — tune against your footage
# (roughly: smaller than one vehicle width at your camera's resolution).
MIN_SAFE_LATERAL_PX = 40.0

# ---- jaywalking / failure_to_yield --------------------------------------
JAYWALK_MIN_DURATION_S = 0.5

# ---- segment post-processing --------------------------------------------
MIN_SEGMENT_DURATION_S = 0.5   # drop blips shorter than this
MAX_MERGE_GAP_S = 1.0          # merge same-class segments separated by <= this

# ---- risk estimator (Part B) --------------------------------------------
RISK_HORIZON_SEC = 5.0
RISK_STEP_STRIDE = 1            # harness calls every frame; internal work can skip
RISK_SMOOTHING = 0.7            # EMA factor toward new instantaneous risk

# The harness's --risk-stride is fixed at 1 in the official run — every
# frame arrives at step(). But nothing says step() has to run full YOLO
# inference on every one of them; the spec explicitly allows "skipping
# frames internally and returning the previous score." This is that skip,
# entirely internal to RiskModel — separate from the harness's own stride
# option, which we don't control at eval time.
RISK_INTERNAL_DETECT_STRIDE = 3   # run YOLO+tracker every Nth frame only