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
        self.show_count = 0                    # people in view when the last show started
        self.rearm_reason = ""
        self._armed = True
        self._more_since: float | None = None  # start of a stretch with more people than show_count
        self._more_last: float | None = None   # last detection in that stretch with more people
        self._more_hits = 0                    # detections in the stretch with more people / all of them
        self._more_seen = 0
        self._fewer_since: float | None = None  # start of a stretch with fewer people than show_count
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
        self._track_count(now)
        if not self._armed and self.last_show_end is not None:
            if self.clear_since_show(now) >= self.s.rearm_clear_sec:
                self._rearm("scene cleared")
            elif self.s.rearm_more_people and self._more_confirmed(now):
                self._rearm("more people")
            elif (now - self.last_show_end) >= self.s.rearm_same_scene_sec:
                self._rearm("same group")

    def _rearm(self, reason: str):
        self._armed = True
        self.rearm_reason = reason

    def _track_count(self, now: float):
        """Compare the head count with the last show's: newcomers can shorten the wait for the next joke."""
        gap = self.s.person_gap_sec
        # Some of the group left: once that has held for rearm_clear_sec, count from the people still here.
        if self.count < self.show_count:
            if self._fewer_since is None:
                self._fewer_since = now
            elif now - self._fewer_since >= self.s.rearm_clear_sec:
                self.show_count = max((len(c[3]) for c in self._cands), default=self.count)
                self._fewer_since = None
        else:
            self._fewer_since = None

        more = self.count > self.show_count
        if more and (self._more_since is None or (now - self._more_last) > gap):
            self._more_since, self._more_hits, self._more_seen = now, 0, 0
        if self._more_since is None:
            return
        if not more and (now - self._more_last) > gap:
            self._more_since = None
            return
        self._more_seen += 1
        if more:
            self._more_hits += 1
            self._more_last = now

    def _more_confirmed(self, now: float) -> bool:
        """More people than at the last show, seen for person_confirm_sec in at least two thirds of
        the detections, so a flickering extra box is not taken for a newcomer."""
        return (self._more_since is not None and (now - self._more_last) <= self.s.person_gap_sec
                and (self._more_last - self._more_since) >= self.s.person_confirm_sec
                and 3 * self._more_hits >= 2 * self._more_seen)

    def on_show_start(self, now: float):
        """Remember how many people the show is for. Uses the most seen in the last second, so a
        missed box right at the start does not make the same group look like newcomers later."""
        self.show_count = max((len(c[3]) for c in self._cands), default=self.count)
        self._more_since = self._fewer_since = None

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

    # Measured up to the last detection, not to now, so the gap tolerance cannot stretch a short glitch.
    def confirm_progress(self, now: float) -> float:
        if self.first_seen is None or not self.present(now):
            return 0.0
        return max(0.0, min(1.0, (self.last_seen - self.first_seen) / max(0.05, self.s.person_confirm_sec)))

    def confirmed(self, now: float) -> bool:
        return (self.present(now) and self.first_seen is not None
                and (self.last_seen - self.first_seen) >= self.s.person_confirm_sec)

    def best(self):
        """(frame, boxes) of the best-looking recent frame with people in it, or None."""
        if not self._cands:
            return None
        _, _, frame, boxes = max(self._cands, key=lambda c: c[1])
        return frame, boxes
