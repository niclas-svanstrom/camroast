# camroast/app.py
"""Main loop: camera in, detection, overlay, and triggering of the show.

The loop never blocks on network or audio. Everything slow runs in ShowRunner tasks,
so the stream keeps updating while the skeletons talk.
"""
import asyncio
import threading
import time

import cv2
import numpy as np

from . import llm
from .camera import CameraSource
from .premade import load_audio_files, load_premade_pairs
from .presence import PresenceTracker
from .settings import Settings
from .show import ShowRunner
from .state import WINDOW_NAME, UIState
from .ui import draw_no_camera, draw_ui_overlay
from .vision import is_dark, maybe_enhance_for_dark, person_boxes
from .voice import RATE, VoiceListener
from .yolo_model import Detectors

try:
    from .tapo_events import TapoEventWatcher
except Exception:  # optional dependency
    TapoEventWatcher = None


class _DetectorWorker:
    """Runs YOLO in a background thread on the most recent submitted frame.

    The camera loop never waits for detection, so the stream stays smooth even when
    inference takes a few hundred milliseconds on a CPU.
    """

    def __init__(self, det: Detectors):
        self.det = det
        self._job = None       # (frame, conf, imgsz)
        self._result = None    # (frame, boxes, ms)
        self._busy = False
        self._stop = False
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy or self._job is not None

    def submit(self, frame, conf, imgsz):
        with self._lock:
            self._job = (frame, conf, imgsz)
        self._event.set()

    def poll(self):
        with self._lock:
            r, self._result = self._result, None
        return r

    def stop(self):
        self._stop = True
        self._event.set()

    def _run(self):
        while not self._stop:
            self._event.wait(0.2)
            with self._lock:
                job, self._job = self._job, None
                self._event.clear()
                self._busy = job is not None
            if job is None:
                continue
            frame, conf, imgsz = job
            t0 = time.perf_counter()
            try:
                boxes = person_boxes(self.det.infer(frame, conf=conf, imgsz=imgsz))
            except Exception as e:
                print("YOLO error:", e)
                boxes = []
            ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._result = (frame, boxes, ms)
                self._busy = False


def _init_audio():
    """Import pygame and open the mixer up front so the first show does not stall on it."""
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()


class CameraApp:
    def __init__(self, settings: Settings):
        self.s = settings
        self.ui = UIState()
        self.ui.roast_enabled = settings.roast_autostart
        self.det = Detectors(threads=settings.torch_threads or None)
        self.worker = _DetectorWorker(self.det)
        self.presence = PresenceTracker(settings)
        self.voice: VoiceListener | None = None
        self.show: ShowRunner | None = None
        self._tapo = None
        self.camera: CameraSource | None = None
        self._stop = False
        self._frames = 0
        self._fps_t0 = time.time()
        self._yolo_ms = 0.0
        self._last_yolo_ts = 0.0
        self._last_boxes = []

    def stop(self):
        """Ask the main loop to exit (same as pressing q)."""
        self._stop = True

    # ---- input handling
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        def inside(rect):
            if rect is None:
                return False
            x1, y1, x2, y2 = rect
            return x1 <= x <= x2 and y1 <= y <= y2

        if inside(self.ui._roast_rect):
            self.ui.roast_enabled = not self.ui.roast_enabled
        elif inside(self.ui._now_rect):
            self.ui.request_roast_now = True
        elif inside(self.ui._premade_rect):
            self.ui.request_premade_now = True
        elif inside(self.ui._mic_rect):
            self.toggle_mic()

    def _on_key(self, key):
        if key == ord("r"):
            self.ui.roast_enabled = not self.ui.roast_enabled
        elif key == ord("n"):
            self.ui.request_roast_now = True
        elif key == ord("p"):
            self.ui.request_premade_now = True
        elif key == ord("m"):
            self.toggle_mic()

    def toggle_mic(self):
        if self.ui.mic_mode_enabled:
            self.ui.mic_mode_enabled = False
            if self.voice:
                self.voice.stop()
            self.ui.mic_status = ""
            return
        if self.voice is None:
            self.voice = VoiceListener(self.s)
        ok = False
        try:
            ok = self.voice.start()
        except Exception as e:
            print("Mic start failed:", e)
        self.ui.mic_mode_enabled = ok
        self.ui.mic_status = f"{self.voice.backend} - {self.voice.device_name}" if ok else "Mic failed to start"

    def _get_voice(self):
        return self.voice if self.ui.mic_mode_enabled else None

    # ---- startup helpers
    def _start_tapo(self):
        s = self.s
        if not (s.tapo_enable_events and TapoEventWatcher and s.tapo_host and s.tapo_user and s.tapo_password):
            return
        try:
            self._tapo = TapoEventWatcher(host=s.tapo_host, user=s.tapo_user, password=s.tapo_password,
                                          port=s.tapo_onvif_port, poll_seconds=s.tapo_poll_seconds)
            self.ui.tapo_ok = True
        except Exception:
            self._tapo = None

    async def _warmup(self):
        try:
            await asyncio.to_thread(_init_audio)
        except Exception as e:
            self.show.error(f"audio: {e}")
        err = await asyncio.to_thread(llm.warmup, self.s.llm_model)
        if err:
            self.show.error(f"OpenAI: {err}")
        else:
            print(f"OpenAI ok: {self.s.llm_model} (reasoning={self.s.llm_reasoning_effort or 'default'})")
        err = await asyncio.to_thread(self.show.speaker.warmup)
        if err:
            self.show.error(f"ElevenLabs: {err}")
        else:
            print(f"ElevenLabs ok: {self.s.tts_model} / {self.s.tts_output_format}")

    # ---- main loop
    async def run(self, cam=0):
        # The camera connects in the background and reconnects by itself, so the app starts
        # and stays usable (buttons, mic, premade clips) even when the camera is unreachable.
        self.camera = CameraSource(cam, self.s)
        print(f"Camera: {self.camera.label}")
        self._start_tapo()

        premade = load_premade_pairs(self.s.premade_dir)
        attention = load_audio_files(self.s.attention_dir)
        filler = load_audio_files(self.s.filler_dir)
        print(f"premade pairs: {len(premade)}, attention clips: {len(attention)}, filler clips: {len(filler)}")
        self.show = ShowRunner(self.s, self.ui, self.presence, self._get_voice, premade, attention, filler)
        asyncio.create_task(self._warmup())
        if self.s.mic_autostart:
            self.toggle_mic()

        if self.s.show_live:
            cv2.namedWindow(WINDOW_NAME)
            cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        last_seq = -1
        last_frame = None
        last_draw = 0.0
        try:
            while not self._stop:
                # Draw on every new camera frame. Without one, still redraw ten times a second
                # so the buttons, subtitles and the no-camera screen stay alive.
                frame, seq = self.camera.read()
                now = time.time()
                new = frame is not None and seq != last_seq
                if new:
                    last_seq, last_frame = seq, frame
                elif (now - last_draw) < 0.1:
                    await asyncio.sleep(0.003)
                    continue
                last_draw = now

                if self.camera.live and last_frame is not None:
                    vis, proc, boxes, dark = self._process_frame(last_frame, new, now)
                else:
                    self._last_boxes = []
                    vis, proc, boxes, dark = self._no_camera_frame(last_frame), None, [], False

                if new:
                    self._frames += 1
                if now - self._fps_t0 >= 0.5:
                    self.ui.fps = self._frames / (now - self._fps_t0)
                    self.ui.yolo_ms = self._yolo_ms if self.presence.recently_seen(now, 1.0) else 0.0
                    self._frames, self._fps_t0 = 0, now

                self._poll_tapo()
                self._update_ui(now, boxes)
                draw_ui_overlay(vis, self.ui, self.s)
                if self.s.show_live:
                    cv2.imshow(WINDOW_NAME, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key != 0xFF:
                        self._on_key(key)

                self._handle_triggers(now, proc, boxes, dark)
                await asyncio.sleep(0.002)
        finally:
            self.worker.stop()
            if self.voice:
                self.voice.stop()
            self.camera.stop()
            if self._tapo is not None:
                try:
                    self._tapo.stop()
                except Exception:
                    pass
            cv2.destroyAllWindows()

    def _process_frame(self, frame, new, now):
        """Detection and drawing for a live camera frame. Returns (vis, proc, boxes, dark)."""
        dark = is_dark(frame, self.s.dark_luma_thresh)
        proc = maybe_enhance_for_dark(frame, self.s.dark_luma_thresh)

        # Motion only gates YOLO while nobody has been seen recently, so a kid standing
        # still at the door keeps being tracked. A slow idle sweep catches static scenes.
        # Detection runs in a worker thread, capped at yolo_max_hz; video draws every frame.
        motion = self.det.motion_pixels(proc) if new else 0
        due = (now - self._last_yolo_ts) >= 1.0 / max(1.0, self.s.yolo_max_hz)
        wanted = (motion >= self.s.min_motion_pixels
                  or self.presence.recently_seen(now, self.s.person_track_sec)
                  or (now - self._last_yolo_ts) >= self.s.yolo_idle_interval_sec)
        if new and due and wanted and not self.worker.busy:
            self._last_yolo_ts = now
            conf = self.s.yolo_conf_night if (self.s.yolo_use_night_conf_when_dark and dark) else self.s.yolo_conf_day
            self.worker.submit(proc, conf, self.s.yolo_imgsz)
        result = self.worker.poll()
        if result is not None:
            det_frame, self._last_boxes, ms = result
            self._yolo_ms = 0.8 * self._yolo_ms + 0.2 * ms if self._yolo_ms else ms
            self.presence.update(now, self._last_boxes, det_frame)
        elif not self.presence.recently_seen(now, self.s.person_track_sec):
            self._last_boxes = []
        boxes = self._last_boxes
        vis = proc.copy()
        for x1, y1, x2, y2, conf_ in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 200, 60), 2)
            cv2.putText(vis, f"{conf_:.2f}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 200, 60), 1)
        return vis, proc, boxes, dark

    def _no_camera_frame(self, last_frame):
        """A dark screen, the size of the last camera frame, explaining what is going on."""
        h, w = last_frame.shape[:2] if last_frame is not None else (720, 1280)
        vis = np.full((h, w, 3), 24, dtype=np.uint8)
        draw_no_camera(vis, [
            "Ingen bild från kameran",
            self.camera.label,
            self.camera.status_text(),
            "Knappar, mikrofon och förinspelade skämt fungerar ändå.",
        ], self.s)
        return vis

    def _poll_tapo(self):
        if self._tapo is None:
            return
        try:
            self.ui.tapo_human_recent = bool(self._tapo.human_recent(2.0))
            self.ui.tapo_motion_recent = bool(self._tapo.motion_recent(2.0))
            self.ui.tapo_ok = self._tapo.ok()
        except Exception:
            self.ui.tapo_human_recent = self.ui.tapo_motion_recent = False

    def _update_ui(self, now, boxes):
        ui = self.ui
        ui.person_count = len(boxes)
        ui.confirm_progress = self.presence.confirm_progress(now)
        ui.armed = self.presence.armed
        v = self._get_voice()
        ui.mic_active = bool(v and v.speech_recent(0.5))
        ui.mic_muted = bool(v and v.is_muted(now))

    def _best_frame(self, proc, boxes):
        if proc is None or not (self.camera and self.camera.live):
            return None, []  # no camera: the skeletons joke without a picture
        best = self.presence.best()
        return best if best else (proc, boxes)

    def _handle_triggers(self, now, proc, boxes, dark):
        show = self.show
        v = self._get_voice()
        if show.busy:
            if v:
                v.drop_pending()  # anything said while the skeletons talk is stale
            return

        if self.ui.request_roast_now:
            self.ui.request_roast_now = False
            fr, bx = self._best_frame(proc, boxes)
            print("show: manual")
            show.start_roast(fr, bx)
            return
        if self.ui.request_premade_now:
            self.ui.request_premade_now = False
            print("show: premade")
            show.start_premade()
            return

        if v:
            if self.s.mic_action == "premade":
                v.drop_pending()
                if self._premade_mic_trigger(now, v, dark, boxes):
                    print("show: premade (speech heard)")
                    show.start_premade()
                    return
            else:
                utt = v.poll_utterance()
                if utt is not None:
                    audio, speech_sec = utt
                    fr, bx = self._best_frame(proc, boxes)
                    print(f"show: voice ({speech_sec:.1f}s of speech)")
                    show.start_talkback(audio, RATE, fr, bx)
                    return

        if not (self.ui.roast_enabled and self.presence.armed):
            return
        use_tapo = self._tapo is not None and (dark or not self.s.tapo_use_only_when_dark)
        if self.presence.confirmed(now) or (use_tapo and self.ui.tapo_human_recent):
            fr, bx = self._best_frame(proc, boxes)
            print(f"show: auto ({len(bx)} person(s))")
            show.start_roast(fr, bx)

    def _premade_mic_trigger(self, now, v, dark, boxes):
        if not v.speech_recent(1.0):
            return False
        if self.s.prem_mic_require_dark and not dark:
            return False
        if self.s.prem_mic_require_no_person and boxes:
            return False
        last = self.presence.last_show_end or 0.0
        return (now - last) >= self.s.mic_cooldown_sec
