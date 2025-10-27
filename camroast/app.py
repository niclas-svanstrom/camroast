# camroast/app.py
import cv2, asyncio
from datetime import datetime
from .settings import Settings
from .state import UIState, WINDOW_NAME
from .ui import draw_ui_overlay
from .yolo_model import Detectors
from .vision import annotate_and_labels as V_ANN, is_dark as V_DARK, has_person_box as V_HAS_PERSON, encode_jpg
from .pipeline import roast_once
from .premade import load_premade_pairs, play_premade_pair
from .premade import load_attention_files, play_random_attention
from . import mic

class CameraApp:
    def __init__(self, settings: Settings):
        self.s = settings
        self.ui = UIState()
        self.det = Detectors()
        self.last = datetime.min

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
        cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            raise RuntimeError("Webcam unavailable")

        self.ui.premade_pairs = load_premade_pairs(self.s.premade_dir)
        self.ui.attention_files = load_attention_files(self.s.attention_dir)
        if self.s.show_live:
            cv2.namedWindow(WINDOW_NAME)
            cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.01); continue

            results = self.det.infer(frame)
            motion_pixels = self.det.motion_pixels(frame)
            vis, labels = V_ANN(frame, results)

            draw_ui_overlay(vis, self.ui)
            if self.s.show_live:
                cv2.imshow(WINDOW_NAME, vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # triggers
            now = datetime.now()
            cooldown_ok = (now - self.last).total_seconds() > self.s.gpt_cooldown_sec
            trigger_auto = self.ui.roast_enabled and cooldown_ok and \
                (self._interesting(results, motion_pixels))

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
        cap.release(); cv2.destroyAllWindows()

    def _interesting(self, results, motion_pixels):
        from .vision import is_interesting as V_INTERESTING
        return V_INTERESTING(results, motion_pixels)

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
