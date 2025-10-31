# camroast/tapo_events.py
import threading, time
from datetime import datetime, timedelta


class TapoEventWatcher:
    """
    Best-effort ONVIF event watcher.
    - Tries to subscribe to ONVIF PullPoint events and watches for motion/human keywords.
    - Degrades gracefully if ONVIF is unavailable or package missing.
    """
    def __init__(self, host: str, user: str, password: str, port: int = 2020, poll_seconds: float = 1.0):
        self.host, self.user, self.password, self.port = host, user, password, port
        self.poll_seconds = max(0.2, float(poll_seconds))
        self._stop = False
        self._ok = False
        self._last_motion = datetime.min
        self._last_human = datetime.min
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            try:
                from onvif import ONVIFCamera
            except Exception:
                return

            # Create camera and event service
            cam = ONVIFCamera(self.host, self.port, self.user, self.password)
            events = cam.create_events_service()
            # Subscribe to pull point
            sub = events.CreatePullPointSubscription({'InitialTerminationTime': 'PT30S'})
            self._ok = True
            while not self._stop:
                try:
                    msgs = events.PullMessages({'Timeout': 'PT1S', 'MessageLimit': 10})
                    nm = getattr(msgs, 'NotificationMessage', [])
                    if not isinstance(nm, (list, tuple)):
                        nm = [nm]
                    for m in nm:
                        txt = _stringify_event(m).lower()
                        if any(k in txt for k in ("human", "person", "people")):
                            self._last_human = datetime.now()
                            self._last_motion = datetime.now()
                        elif "motion" in txt or "cellmotion" in txt:
                            self._last_motion = datetime.now()
                except Exception:
                    time.sleep(self.poll_seconds)
                time.sleep(self.poll_seconds)
        except Exception:
            return

    def stop(self):
        self._stop = True
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass

    def ok(self) -> bool:
        return self._ok

    def motion_recent(self, secs: float = 2.0) -> bool:
        return (datetime.now() - self._last_motion) <= timedelta(seconds=secs)

    def human_recent(self, secs: float = 2.0) -> bool:
        return (datetime.now() - self._last_human) <= timedelta(seconds=secs)


def _stringify_event(m) -> str:
    # Convert a nested ONVIF event object to a string we can keyword-search.
    parts = []
    def walk(obj):
        try:
            if obj is None:
                return
            if isinstance(obj, (str, bytes)):
                parts.append(obj.decode('utf-8', 'ignore') if isinstance(obj, bytes) else obj)
                return
            if isinstance(obj, (int, float, bool)):
                parts.append(str(obj))
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    parts.append(str(k))
                    walk(v)
                return
            if isinstance(obj, (list, tuple, set)):
                for v in obj:
                    walk(v)
                return
            # Dataclass/zeep object: iterate attributes
            for k in dir(obj):
                if k.startswith('_'):
                    continue
                try:
                    v = getattr(obj, k)
                except Exception:
                    continue
                if callable(v):
                    continue
                parts.append(str(k))
                walk(v)
        except Exception:
            pass
    walk(m)
    return ' '.join(parts)

