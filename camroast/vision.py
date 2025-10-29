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
    return has_person


def _gamma_lut(gamma: float):
    gamma = max(0.1, min(3.0, gamma))
    # gamma < 1 brightens
    return np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)


def enhance_low_light(frame: np.ndarray, clahe_clip: float = 2.0, tile_grid: tuple = (8, 8), gamma: float = 0.6) -> np.ndarray:
    # Apply CLAHE on L channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile_grid)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    # Gamma correction
    lut = _gamma_lut(gamma)
    out = cv2.LUT(out, lut)
    return out


def maybe_enhance_for_dark(frame: np.ndarray, dark_thresh: float = 40.0) -> np.ndarray:
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) < dark_thresh:
            return enhance_low_light(frame)
    except Exception:
        pass
    return frame
