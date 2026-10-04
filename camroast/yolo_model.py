# camroast/yolo_model.py
import os

import cv2
import torch
from ultralytics import YOLO


class Detectors:
    def __init__(self, threads: int | None = None):
        # Inference runs in a worker thread next to the camera loop, the stream reader and
        # audio. Torch's default thread count then oversubscribes the CPU and inference gets
        # 3 to 20 times slower and erratic; yolov8n is small, so a few threads are plenty.
        n = threads or max(1, min(4, (os.cpu_count() or 2) // 3))
        torch.set_num_threads(n)
        self.yolo = YOLO("yolov8n.pt")
        self.bsub = cv2.createBackgroundSubtractorMOG2(120, 50)

    def infer(self, frame, conf: float | None = None, imgsz: int = 640):
        # Detect only persons (COCO class 0). Avoids car/bus/truck clutter.
        kw = dict(verbose=False, classes=[0], imgsz=imgsz)
        if conf is not None:
            kw["conf"] = conf
        return self.yolo(frame, **kw)[0]

    def motion_pixels(self, frame):
        return int(cv2.countNonZero(self.bsub.apply(frame)))
