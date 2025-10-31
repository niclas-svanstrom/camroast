# camroast/app.py
import cv2, asyncio
import threading, time
from datetime import datetime
from .settings import Settings
from .state import UIState, WINDOW_NAME
from .ui import draw_ui_overlay
from .yolo_model import Detectors
from .vision import annotate_and_labels as V_ANN, is_dark as V_DARK, has_person_box as V_HAS_PERSON, encode_jpg, maybe_enhance_for_dark
from .pipeline import roast_once
from .premade import load_premade_pairs, play_premade_pair
from .premade import load_attention_files, play_random_attention
from . import mic
try:
    from .tapo_events import TapoEventWatcher
except Exception:
    TapoEventWatcher = None

class CameraApp:
    def __init__(self, settings: Settings):
        self.s = settings
        self.ui = UIState()
        self.det = Detectors()
        self.last = datetime.min
        from collections import deque
        self._person_hist = deque(maxlen=self.s.person_confirm_window)
        self._tapo = None

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Helper to check if a point is inside a rect (x1,y1,x2,y2)
        def inside(ptx, pty, rect):
            if rect is None:
                return False
            x1, y1, x2, y2 = rect
            return (x1 <= ptx <= x2) and (y1 <= pty <= y2)

        # Roast toggle
        if inside(x, y, self.ui._roast_rect):
            self.ui.roast_enabled = not self.ui.roast_enabled
            return

        # Roast Now
        if inside(x, y, self.ui._now_rect):
            self.ui.request_roast_now = True
            return

        # Premade Now
        if inside(x, y, self.ui._premade_rect):
            self.ui.request_premade_now = True
            return

        # Mic Mode toggle (start/stop detector)
        if inside(x, y, self.ui._mic_rect):
            self.ui.mic_mode_enabled = not self.ui.mic_mode_enabled
            if self.ui.mic_mode_enabled:
                # Start a detector if not present
                if self.ui._mic_detector is None:
                    # Prefer Silero if available, else fallback
                    det = None
                    try:
                        det = mic.SileroSpeechDetector()
                    except Exception:
                        try:
                            det = mic.MicSpeechDetector()
                        except Exception:
                            det = None
                    self.ui._mic_detector = det
                # Try to start it
                if self.ui._mic_detector is not None:
                    ok = False
                    try:
                        ok = self.ui._mic_detector.start()
                    except Exception:
                        ok = False
                    if not ok:
                        self.ui.mic_mode_enabled = False
                        self.ui._gate_text = "Mic failed to start"
                else:
                    self.ui.mic_mode_enabled = False
                    self.ui._gate_text = "No mic backend"
            else:
                # Turning off
                try:
                    if self.ui._mic_detector:
                        self.ui._mic_detector.stop()
                except Exception:
                    pass
            return

    async def run(self, cam=0):
        # Prefer FFmpeg backend for network streams (RTSP/HTTP) when available
        if isinstance(cam, str) and (cam.startswith("rtsp://") or cam.startswith("http://") or cam.startswith("https://")):
            try:
                cap = cv2.VideoCapture(cam, cv2.CAP_FFMPEG)
            except Exception:
                cap = cv2.VideoCapture(cam)
        else:
            cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            raise RuntimeError("Camera source unavailable")

        # Try to reduce capture buffering to lower latency
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # For network streams, keep only the latest frame using a background reader
        reader = None
        if isinstance(cam, str):
            class _LatestFrameReader:
                def __init__(self, cap):
                    self.cap = cap
                    self.frame = None
                    self.ok = False
                    self._stop = False
                    self._lock = threading.Lock()
                    self._t = threading.Thread(target=self._run, daemon=True)
                    self._t.start()

                def _run(self):
                    while not self._stop:
                        ok, fr = self.cap.read()
                        if not ok:
                            time.sleep(0.005)
                            continue
                        with self._lock:
                            self.ok = True
                            self.frame = fr

                def read(self):
                    with self._lock:
                        return self.ok, None if self.frame is None else self.frame.copy()

                def stop(self):
                    self._stop = True
                    try:
                        self._t.join(timeout=0.2)
                    except Exception:
                        pass

            reader = _LatestFrameReader(cap)

        # Try to improve low-light via camera properties if supported
        if self.s.try_camera_low_light:
            try:
                # Enable auto exposure where supported (0.75 on many backends)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            except Exception:
                pass
            try:
                if self.s.camera_exposure is not None:
                    cap.set(cv2.CAP_PROP_EXPOSURE, self.s.camera_exposure)
            except Exception:
                pass
            try:
                if self.s.camera_gain is not None:
                    cap.set(cv2.CAP_PROP_GAIN, self.s.camera_gain)
            except Exception:
                pass
            try:
                if self.s.camera_brightness is not None:
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, self.s.camera_brightness)
            except Exception:
                pass

        # Start Tapo events if enabled
        if self.s.tapo_enable_events and TapoEventWatcher and self.s.tapo_host and self.s.tapo_user and self.s.tapo_password:
            try:
                self._tapo = TapoEventWatcher(
                    host=self.s.tapo_host,
                    user=self.s.tapo_user,
                    password=self.s.tapo_password,
                    port=self.s.tapo_onvif_port,
                    poll_seconds=self.s.tapo_poll_seconds
                )
                # Show Tapo indicator in UI once enabled
                self.ui.tapo_ok = True
            except Exception:
                self._tapo = None

        self.ui.premade_pairs = load_premade_pairs(self.s.premade_dir)
        self.ui.attention_files = load_attention_files(self.s.attention_dir)
        if self.s.show_live:
            cv2.namedWindow(WINDOW_NAME)
            cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        while True:
            if reader is None:
                ok, frame = cap.read()
            else:
                ok, frame = reader.read()
            if not ok:
                await asyncio.sleep(0.01); continue

            is_dark_now = V_DARK(frame, self.s.dark_luma_thresh)
            proc = maybe_enhance_for_dark(frame, self.s.dark_luma_thresh)
            # Fast motion gate before expensive YOLO
            motion_pixels = self.det.motion_pixels(proc)

            results = None
            labels = set()
            if motion_pixels >= self.s.min_motion_pixels:
                # Use a stricter conf at night to cut false positives
                use_conf = (self.s.yolo_conf_night if (self.s.yolo_use_night_conf_when_dark and is_dark_now)
                            else self.s.yolo_conf_day)
                results = self.det.infer(proc, conf=use_conf)
                vis, labels = V_ANN(proc, results)
            else:
                vis = proc.copy()
                class _EmptyResults:
                    names = {0: "person"}
                    boxes = []
                results = _EmptyResults()

            # Update recent person detections history for debounce
            try:
                has_person_now = V_HAS_PERSON(results)
            except Exception:
                has_person_now = False
            self._person_hist.append(bool(has_person_now))

            # Integrate Tapo events
            tapo_human = False
            tapo_motion = False
            if self._tapo is not None:
                try:
                    tapo_human = self._tapo.human_recent(2.0)
                    tapo_motion = self._tapo.motion_recent(2.0)
                    self.ui.tapo_ok = self._tapo.ok()
                except Exception:
                    tapo_human = False
                    tapo_motion = False
                self.ui.tapo_human_recent = bool(tapo_human)
                self.ui.tapo_motion_recent = bool(tapo_motion)

            draw_ui_overlay(vis, self.ui)
            if self.s.show_live:
                cv2.imshow(WINDOW_NAME, vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # triggers
            now = datetime.now()
            cooldown_ok = (now - self.last).total_seconds() > self.s.gpt_cooldown_sec
            # Require enough positive person detections in recent window
            person_recent = sum(1 for x in self._person_hist if x) >= self.s.person_confirm_min
            use_tapo_now = (self._tapo is not None) and ((self.s.tapo_use_only_when_dark and is_dark_now) or (not self.s.tapo_use_only_when_dark))
            # Two ways to auto-trigger:
            #  - Our pipeline says person + interesting (motion) AND debounce met
            #  - Tapo says human (bypass our YOLO) when enabled
            pipeline_positive = person_recent and self._interesting(results, motion_pixels)
            tapo_positive = (use_tapo_now and tapo_human)
            trigger_auto = self.ui.roast_enabled and cooldown_ok and (pipeline_positive or tapo_positive)

            trigger_manual = self.ui.request_roast_now
            trigger_premade_manual = self.ui.request_premade_now and bool(self.ui.premade_pairs)
            trigger_mic = self._mic_trigger(frame, results, now)

            if trigger_manual or trigger_auto:
                self.ui.request_roast_now = False
                import base64
                # Start a non-blocking attention sound while generating
                try:
                    play_random_attention(self.ui.attention_files)
                except Exception:
                    pass
                txt = await roast_once(
                    frame,
                    labels,
                    (self.s.voice_skallepar, self.s.voice_benrangel),
                    v_enc_jpg=lambda fr: base64.b64encode(encode_jpg(fr)).decode("ascii")
                )
                if isinstance(txt, str) and txt:
                    self.ui.last_text, self.last = txt, datetime.now()
            elif trigger_premade_manual:
                self.ui.request_premade_now = False
                pair = self._next_premade()
                play_premade_pair(pair)
            elif trigger_mic and self.ui.premade_pairs:
                pair = self._next_premade()
                play_premade_pair(pair)
                self.ui._mic_last_trigger = datetime.now()

            await asyncio.sleep(0.002)

        # cleanup
        if self.ui._mic_detector:
            self.ui._mic_detector.stop()
        if reader is not None:
            reader.stop()
        if self._tapo is not None:
            try:
                self._tapo.stop()
            except Exception:
                pass
        cap.release(); cv2.destroyAllWindows()

    def _interesting(self, results, motion_pixels):
        try:
            return (motion_pixels >= self.s.min_motion_pixels) and V_HAS_PERSON(results)
        except Exception:
            return False

    def _next_premade(self):
        p = self.ui.premade_pairs[self.ui.premade_idx % len(self.ui.premade_pairs)]
        self.ui.premade_idx += 1
        return p

    def _mic_trigger(self, frame, results, now):
        if not (self.ui.mic_mode_enabled and self.ui._mic_detector):
            return False
        dark_ok = V_DARK(frame, self.s.dark_luma_thresh) if self.s.prem_mic_require_dark else True
        no_person_ok = (not V_HAS_PERSON(results)) if self.s.prem_mic_require_no_person else True
        speech_ok = self.ui._mic_detector.speech_recent(1.0)
        cd_rem = self.s.mic_cooldown_sec - (now - self.ui._mic_last_trigger).total_seconds()
        cooldown_ok = cd_rem <= 0
        self.ui._gate_text = f"gate: dark={'skip' if not self.s.prem_mic_require_dark else dark_ok} no_person={'skip' if not self.s.prem_mic_require_no_person else no_person_ok} speech={speech_ok} cd={'ok' if cooldown_ok else f'{cd_rem:.1f}s'}"
        return dark_ok and no_person_ok and speech_ok and cooldown_ok
