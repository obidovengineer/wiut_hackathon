from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from . import config

class Detector:
    def __init__(self, weights_path=config.YOLO_WEIGHTS, conf=config.DETECT_CONF,
                 iou=config.DETECT_IOU, classes=config.TARGET_CLASSES):
        self.model = YOLO(str(weights_path))
        self.conf = conf
        self.iou = iou
        self.classes = classes

    def infer(self, frame: np.ndarray) -> np.ndarray:
        r = self.model(frame, conf=self.conf, iou=self.iou, classes=self.classes, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        boxes = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        return np.hstack([boxes, conf.reshape(-1, 1), cls.reshape(-1, 1)]).astype(np.float32)

    def infer_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        results = self.model(frames, conf=self.conf, iou=self.iou, classes=self.classes, verbose=False)
        out = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(np.zeros((0, 6), dtype=np.float32))
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            out.append(np.hstack([boxes, conf.reshape(-1, 1), cls.reshape(-1, 1)]).astype(np.float32))
        return out