# camroast/yolo_model.py
import cv2
from ultralytics import YOLO

class Detectors:
    def __init__(self):
        self.yolo = YOLO("yolov8n.pt")
        self.bsub = cv2.createBackgroundSubtractorMOG2(120, 50)

    def infer(self, frame):
        # Detect only persons (COCO class 0). Avoids car/bus/truck clutter.
        return self.yolo(frame, verbose=False, classes=[0])[0]

    def motion_pixels(self, frame):
        return int(cv2.countNonZero(self.bsub.apply(frame)))
