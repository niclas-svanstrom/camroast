# camroast/presence.py
"""Time-based person presence: confirmation, gap tolerance, best-frame choice and re-arming."""
import math
from collections import deque


def frame_score(shape, boxes) -> float:
    """Higher is better: large, centered, confident person boxes."""
    h, w = shape[:2]
    if not boxes or w == 0 or h == 0:
        return 0.0
    best = 0.0
    for x1, y1, x2, y2, conf in boxes:
        area = max(0, x2 - x1) * max(0, y2 - y1) / float(w * h)
        cx = (x1 + x2) / 2.0 / w - 0.5
        cy = (y1 + y2) / 2.0 / h - 0.5
        center = 1.0 - min(1.0, math.hypot(cx, cy) / 0.71)
        best = max(best, area * (0.5 + 0.5 * center) * (0.5 + 0.5 * conf))
    return best * (1.0 + 0.1 * (len(boxes) - 1))


class PresenceTracker:
    def __init__(self, s):
        self.s = s
        self.first_seen: float | None = None   # start of the current continuous presence
        self.last_seen: float | None = None    # last detection of any person
        self.last_show_end: float | None = None
        self.count = 0
        self._armed = True
        self._cands: deque = deque()           # (ts, score, frame, boxes) within best_frame_window_sec

    # ---- updates
    def update(self, now: float, boxes, frame):
        self.count = len(boxes)
        if boxes:
            if self.first_seen is None or (now - self.last_seen) > self.s.person_gap_sec:
                self.first_seen = now
            self.last_seen = now
            self._cands.append((now, frame_score(frame.shape, boxes), frame, list(boxes)))
        elif self.first_seen is not None and (now - self.last_seen) > self.s.person_gap_sec:
            self.first_seen = None
        cutoff = now - self.s.best_frame_window_sec
        while self._cands and self._cands[0][0] < cutoff:
            self._cands.popleft()
        if not self._armed and self.last_show_end is not None:
            if (now - self.last_show_end) >= self.s.rearm_same_scene_sec or self.clear_since_show(now) >= self.s.rearm_clear_sec:
                self._armed = True

    def on_show_end(self, now: float):
        self.last_show_end = now
        self._armed = False

    # ---- queries
    @property
    def armed(self) -> bool:
        return self._armed

    def present(self, now: float) -> bool:
        return self.last_seen is not None and (now - self.last_seen) <= self.s.person_gap_sec

    def recently_seen(self, now: float, within: float) -> bool:
        return self.last_seen is not None and (now - self.last_seen) <= within

    def clear_since_show(self, now: float) -> float:
        """Seconds the scene has been empty, counted from the end of the last show."""
        anchors = [t for t in (self.last_seen, self.last_show_end) if t is not None]
        if not anchors:
            return float("inf")
        return now - max(anchors)

    def confirm_progress(self, now: float) -> float:
        if self.first_seen is None or not self.present(now):
            return 0.0
        return max(0.0, min(1.0, (now - self.first_seen) / max(0.05, self.s.person_confirm_sec)))

    def confirmed(self, now: float) -> bool:
        return self.present(now) and self.first_seen is not None and (now - self.first_seen) >= self.s.person_confirm_sec

    def best(self):
        """(frame, boxes) of the best-looking recent frame with people in it, or None."""
        if not self._cands:
            return None
        _, _, frame, boxes = max(self._cands, key=lambda c: c[1])
        return frame, boxes
