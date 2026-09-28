# 🚦 MOMENTUM — WIUT Hackathon 2026

### Computer Vision · Traffic Event Detection & Accident Risk Prediction

**MOMENTUM** is our Computer Vision solution for detecting traffic events and estimating accident risk from road-camera videos.

The system processes video using **YOLO + ByteTrack**, builds object trajectories, analyzes scene geometry and motion, and produces the required competition predictions.

---

## 🧠 Pipeline

```text
VIDEO
  │
  ▼
OpenCV
  │
  ▼
YOLO Object Detection
  │
  ▼
ByteTrack
  │
  ▼
Object IDs + Trajectories
  │
  ├───────────────┐
  ▼               ▼
PART A          PART B
Events          Risk
  │               │
  └───────┬───────┘
          ▼
   predictions.json
```

---

## 🔍 Computer Vision Stack

| Component            | Technology               |
| -------------------- | ------------------------ |
| Video processing     | OpenCV                   |
| Object detection     | YOLO                     |
| Object tracking      | ByteTrack                |
| Numerical processing | NumPy                    |
| Trajectories         | `deque`                  |
| Scene understanding  | Camera calibration       |
| Event detection      | Motion + geometric rules |
| Accident risk        | TTC + motion signals     |

### Detected objects

Initially we track:

```text
Person
Bicycle
Car
Motorcycle
Bus
Truck
```

Tracking gives each object a persistent ID:

```text
Car → ID 17
Car → ID 24
Person → ID 8
```

These IDs allow us to analyze object movement over time.

---

# 🚨 Part A — Event Detection

The system detects traffic events such as:

```text
accident
near_miss
red_light
wrong_way
illegal_u_turn
stopped_vehicle
jaywalking
failure_to_yield
illegal_turn
solid_line_crossing
stop_line
congestion
road_obstacle
fire_smoke
```

Events are returned as temporal segments:

```json
[
  [12.4, 18.9, "accident"],
  [40.0, 43.5, "red_light"]
]
```

The system uses tracking, trajectories, scene calibration and temporal post-processing to determine when events start and end.

---

# ⚠️ Part B — Accident Risk

Part B estimates the probability that an accident will occur within the next **5 seconds**.

The risk estimator analyzes information such as:

* Object motion
* Relative motion
* Distance
* Time-to-Collision (TTC)
* Object interactions

The final risk scores are generated frame-by-frame by `RiskEstimator`.

---

# 📦 Output

Part A and Part B are stored in **one `predictions.json` file**:

```json
{
  "team": "MOMENTUM",
  "videos": {
    "test_001.mp4": {
      "events": [
        [12.4, 18.9, "accident"]
      ],
      "risk": [
        [0.00, 0.01],
        [0.04, 0.01]
      ]
    }
  }
}
```

---

# 📁 Repository Structure

```text
wiut_hackathon/
│
├── calibration/
├── examples/
├── src/
├── weights/
│
├── solution.py
├── run_submission.py
├── evaluate.py
├── requirements.txt
└── README.md
```

### Main files

**`solution.py`**
Main implementation containing event detection and risk estimation.

**`run_submission.py`**
Runs the solution on videos and generates predictions.

**`evaluate.py`**
Validates and evaluates predictions.

**`calibration/`**
Camera/scene-specific calibration data.

**`weights/`**
Model weights used by the solution.

---

# ⚙️ Installation

```bash
git clone https://github.com/obidovengineer/wiut_hackathon.git
cd wiut_hackathon

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

---

# ▶️ Run

```bash
python run_submission.py --videos samples --out predictions.json --team MOMENTUM
```

Validate predictions:

```bash
python evaluate.py --pred predictions.json --validate-only
```

---

# 🎯 Development Roadmap

```text
[✓] YOLO detection
[✓] ByteTrack integration
[✓] Object IDs
[✓] Basic trajectories

[ ] Improve tracking
[ ] Scene calibration
[ ] Lane detection
[ ] Direction estimation
[ ] Stopped vehicle
[ ] Wrong-way
[ ] Stop-line / solid-line violations
[ ] Illegal turns / U-turns
[ ] Jaywalking
[ ] Congestion
[ ] Red-light
[ ] Failure-to-yield
[ ] Accident / near-miss
[ ] TTC
[ ] Part B risk estimation
[ ] Temporal post-processing
[ ] Runtime optimization
```

---

## 🚀 MOMENTUM

**Detect. Track. Understand. Predict.**