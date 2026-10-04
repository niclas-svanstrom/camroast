# camroast/vision.py
import base64

import cv2
import numpy as np


def encode_jpg(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else b""


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    m = max(h, w)
    if max_side <= 0 or m <= max_side:
        return frame
    scale = max_side / float(m)
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def to_b64_jpg(frame: np.ndarray, max_side: int = 1024, quality: int = 85) -> str:
    return base64.b64encode(encode_jpg(resize_max_side(frame, max_side), quality)).decode("ascii")


def person_boxes(results):
    """(x1, y1, x2, y2, conf) for every person box in a YOLO result."""
    out = []
    if results is None or getattr(results, "boxes", None) is None:
        return out
    for b in results.boxes:
        if results.names[int(b.cls)] != "person":
            continue
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        out.append((x1, y1, x2, y2, float(b.conf[0])))
    return out


def crop_to_persons(frame: np.ndarray, boxes, margin: float = 0.35, min_frac: float = 0.45) -> np.ndarray:
    """Crop around all person boxes with some margin, never smaller than min_frac of the frame."""
    h, w = frame.shape[:2]
    if not boxes:
        return frame
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    cw = max((x2 - x1) * (1 + 2 * margin), w * min_frac)
    ch = max((y2 - y1) * (1 + 2 * margin), h * min_frac)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    X1 = int(max(0, cx - cw / 2))
    Y1 = int(max(0, cy - ch / 2))
    X2 = int(min(w, cx + cw / 2))
    Y2 = int(min(h, cy + ch / 2))
    if X2 - X1 < 32 or Y2 - Y1 < 32:
        return frame
    return frame[Y1:Y2, X1:X2]


def is_dark(frame: np.ndarray, thresh: float = 40.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < thresh


def has_person_box(results) -> bool:
    return bool(person_boxes(results))


def _gamma_lut(gamma: float):
    gamma = max(0.1, min(3.0, gamma))
    # gamma < 1 brightens
    return np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)


def enhance_low_light(frame: np.ndarray, clahe_clip: float = 2.0, tile_grid: tuple = (8, 8), gamma: float = 0.6) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile_grid)
    out = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    return cv2.LUT(out, _gamma_lut(gamma))


def maybe_enhance_for_dark(frame: np.ndarray, dark_thresh: float = 40.0) -> np.ndarray:
    try:
        if is_dark(frame, dark_thresh):
            return enhance_low_light(frame)
    except Exception:
        pass
    return frame
