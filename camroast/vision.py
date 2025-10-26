import cv2
import numpy as np


def encode_jpg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""


def annotate_and_labels(frame: np.ndarray, results):
    annotated = results.plot()
    labels = {results.names[int(b.cls)] for b in results.boxes}
    y = 20
    for lbl in sorted(labels):
        cv2.putText(annotated, lbl, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 22
    return annotated, labels


def is_dark(frame: np.ndarray, thresh: float = 40.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < thresh


def has_person_box(results) -> bool:
    return any(results.names[int(b.cls)] == "person" for b in results.boxes)


def is_interesting(results, motion_pixels: int) -> bool:
    if motion_pixels < 1500:
        return False
    has_person = any(results.names[int(b.cls)] == "person" for b in results.boxes)
    has_vehicle = any(results.names[int(b.cls)] in {"car", "bus", "truck"} for b in results.boxes)
    return has_person and not has_vehicle

