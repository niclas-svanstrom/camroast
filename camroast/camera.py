# camroast/camera.py
"""Camera input that never blocks or crashes the app.

A background thread opens the source (webcam index, rtsp/http URL or video file), keeps only
the newest frame, and reconnects on its own when the camera is unreachable or the stream
drops. The app keeps running without a picture in the meantime.
"""
import re
import threading
import time

import cv2

_CRED_RE = re.compile(r"(?<=://)([^:/@\s]+):([^@/\s]+)@")


def is_url(src) -> bool:
    return isinstance(src, str) and src.startswith(("rtsp://", "rtsps://", "http://", "https://"))


def mask_source(src) -> str:
    """The source as text, with any password in the URL replaced by ***."""
    return _CRED_RE.sub(r"\1:***@", str(src))


def _open(src, timeout_ms: int):
    """Open a capture or return None. Network sources give up after timeout_ms."""
    if is_url(src):
        params = []
        for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
            prop = getattr(cv2, name, None)
            if prop is not None:
                params += [prop, int(timeout_ms)]
        try:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, params)
        except Exception:
            cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def apply_low_light_props(cap, s):
    if not s.try_camera_low_light:
        return
    props = [
        (cv2.CAP_PROP_AUTO_EXPOSURE, 0.75),
        (cv2.CAP_PROP_EXPOSURE, s.camera_exposure),
        (cv2.CAP_PROP_GAIN, s.camera_gain),
        (cv2.CAP_PROP_BRIGHTNESS, s.camera_brightness),
    ]
    for prop, val in props:
        if val is None:
            continue
        try:
            cap.set(prop, val)
        except Exception:
            pass


class CameraSource:
    """Owns the capture in a background thread. read() never blocks."""

    def __init__(self, src, s, opener=_open):
        self.src = src
        self.s = s
        self.label = mask_source(src)
        self.is_file = isinstance(src, str) and not is_url(src)
        self.state = "connecting"      # connecting | live | retrying
        self.attempts = 0              # failed connection attempts in a row
        self.next_retry = 0.0
        self._open = opener
        self._frame = None
        self._seq = 0
        self._last_frame_ts = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="camera", daemon=True)
        self._t.start()

    # ---- API for the main loop
    def read(self):
        """(newest frame or None, sequence number). The number changes only for a new frame."""
        with self._lock:
            return self._frame, self._seq

    @property
    def live(self) -> bool:
        return self.state == "live" and (time.time() - self._last_frame_ts) < self.s.camera_stall_sec

    def status_text(self) -> str:
        if self.state == "retrying":
            wait = max(0, int(round(self.next_retry - time.time())))
            if wait > 0:
                return f"Försöker igen om {wait} s (försök {self.attempts})"
        if self.attempts:
            return f"Ansluter... (försök {self.attempts + 1})"
        return "Ansluter till kameran..."

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)  # an open in progress may take up to the timeout; the thread is a daemon

    # ---- capture thread
    def _run(self):
        while not self._stop.is_set():
            if self.state != "retrying":
                self.state = "connecting"
            cap = None
            try:
                cap = self._open(self.src, int(self.s.camera_timeout_sec * 1000))
            except Exception as e:
                print(f"Camera: open error: {e}")
            if cap is None:
                self.attempts += 1
                if self.attempts == 1:
                    print(f"Camera: cannot open {self.label}, retrying every {self.s.camera_retry_sec:g} s")
                self.state = "retrying"
                self.next_retry = time.time() + self.s.camera_retry_sec
                self._stop.wait(self.s.camera_retry_sec)
                continue
            try:
                self._read_loop(cap)
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
            if not self._stop.is_set():
                # the stream dropped: try again soon, it is usually a short hiccup
                self.state = "retrying"
                self.next_retry = time.time() + 1.0
                self._stop.wait(1.0)

    def _read_loop(self, cap):
        apply_low_light_props(cap, self.s)
        period = 0.0
        if self.is_file:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            period = 1.0 / fps if fps > 1.0 else 1.0 / 25.0
        got_any = False
        last_ok = time.time()
        while not self._stop.is_set():
            t0 = time.time()
            ok, fr = cap.read()
            if ok and fr is not None:
                now = time.time()
                with self._lock:
                    self._frame = fr  # cap.read() returns a new array each time, no copy needed
                    self._seq += 1
                    self._last_frame_ts = now
                    self.state = "live"
                if not got_any:
                    got_any = True
                    tries = f" after {self.attempts + 1} attempts" if self.attempts else ""
                    print(f"Camera: connected to {self.label}{tries}")
                    self.attempts = 0
                last_ok = now
                if period:
                    self._stop.wait(max(0.0, period - (time.time() - t0)))
                continue
            if time.time() - last_ok > self.s.camera_stall_sec:
                print(f"Camera: no frames for {self.s.camera_stall_sec:g} s, reconnecting")
                return
            if self.is_file and got_any:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the clip
                continue
            self._stop.wait(0.02)
