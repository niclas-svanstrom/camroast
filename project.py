#!/usr/bin/env python3
"""
Real-time roasting skeletons – Skalle-pär & Benrangel (no file I/O)
-------------------------------------------------------------------
• GPT-4o-mini vision  → scene description & jokes
• ElevenLabs TTS      → Swedish fast-paced, dialect voices
• YOLOv8n             → person-/car-detection & live overlay
• Pygame              → playback (from memory, no disk writes)
"""

# ── Imports ───────────────────────────────────────────────────────────────
import asyncio, base64, hashlib, os, io
from collections import deque
from datetime import datetime
import cv2
import numpy as np
from openai import OpenAI
import pygame
from dotenv import load_dotenv
from ultralytics import YOLO
import re
import unicodedata
import threading, queue, time

# NEW: ElevenLabs SDK
from elevenlabs.client import ElevenLabs

# ── Config ────────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

SHOW_LIVE        = True          # set False for head-less deploy
FPS              = 10
FRAME_SKIP       = 5
GPT_COOLDOWN_SEC = 10
MIC_COOLDOWN_SEC = 10
DARK_LUMA_THRESH = 40            # average grayscale below = dark

PREMADE_DIR      = os.getenv("PREMADE_DIR", "premade")
PREM_MIC_REQUIRE_DARK = os.getenv("PREM_MIC_REQUIRE_DARK", "0") not in {"0", "false", "False"}
# Default now allows mic triggers even when people are present.
# Set PREM_MIC_REQUIRE_NO_PERSON=1 if you want to restore the old behavior.
PREM_MIC_REQUIRE_NO_PERSON = os.getenv("PREM_MIC_REQUIRE_NO_PERSON", "0") not in {"0", "false", "False"}
PREM_MIC_DEBUG = os.getenv("PREM_MIC_DEBUG", "0") in {"1", "true", "True"}

VOICE_SKALLEPAR  = "NHVO1d5lgqVtAvyYNL2P"   # e.g. "Adam"
VOICE_BENRANGEL  = "S6pZEFGfrgnWx4AETPdD"

SYSTEM_PROMPT = (
    "Du är två sarkastiska skelett – Skalle-pär och Benrangel – som står på gatan och roastar förbipasserande.\n"
    "MÅL: En kvick tvåraders dialog på svenska som känns improviserad och publikvänlig.\n"
    "\n"
    "FORMAT (obligatoriskt):\n"
    "1) Skalle-pär: <en (1) mening>\n"
    "2) Benrangel: <en (1) mening>\n"
    "Exakt två meningar totalt. Inga extra rader, inga emojis.\n"
    "\n"
    "STIL:\n"
    "• Syrlig, självironisk, snabb gatuton. Blanda in att ni är skelett.\n"
    "• Ni kan nämna varandras namn naturligt i början eller mitten av meningen (inte i slutet).\n"
    "• Skämta om situationen, kläder, rörelser, rekvisita – inte om känsliga attribut (ålder, kropp, hälsa, religion, etnicitet, identitet).\n"
    "\n"
    "STENHÅRDA REGLER (inga undantag):\n"
    "• Gör ALDRIG meta-referenser till kamera, bild, AI, modell, detektion, YOLO, neurala nät, algoritmer eller 'jag ser'.\n"
    "• Påstå inte hur ni vet saker – ni står bara där och kommenterar.\n"
    "• Inga uppmaningar, inga förklaringar, ingen extra text före/efter replikerna.\n"
    "\n"
    "OM NÅGON REGEL BRYTS: skriv om direkt tills allt följer reglerna.\n"
    "\n"
    "EXEMPEL (OK):\n"
    "Skalle-pär: Titta där, Benrangel, den kappan svajar som om den flytt från en storm!\n"
    "Benrangel: Du har rätt, Skalle-pär — jag har sett mer stadga i mina lösa leder!\n"
    "\n"
    "EXEMPEL (FÖRBJUDET):\n"
    "Skalle-pär: YOLO sa att en person närmar sig…\n"
    "Benrangel: Kameran fångade det — vi analyserade bilden!\n"
)


SYSTEM_PROMPT_DESCRIBE = (
    "Du beskriver en gatukameras­cen på svenska för ett humorsegment.\n"
    "Var neutral, respektfull och precis. Gör aldrig antaganden om ålder, identitet, kropp, hälsa, religion eller etnicitet.\n"
    "Fokusera på kläder, rörelser, föremål och situationer. Om osäker: markera det.\n"
    "Avsluta med en lättsam oneliner som driver med situationen (inte personen), max 12 ord.\n"
    "Format:\n"
    "Beskrivning: …\n"
    "Oneliner: …"
)


# ── Singletons ────────────────────────────────────────────────────────────
yolo  = YOLO("yolov8n.pt")
bsub  = cv2.createBackgroundSubtractorMOG2(120, 50)
pygame.mixer.init()
history = deque(maxlen=6)

# ── Simple UI State for on-screen buttons ────────────────────────────────
WINDOW_NAME = "RoastCam"

class UIState:
    def __init__(self):
        # Start in detection-only mode; GPT/ElevenLabs off until toggled
        self.roast_enabled = False
        # One-shot manual roast request when pressing the button
        self.request_roast_now = False
        # Last OpenAI text to render on screen
        self.last_text = ""
        self.last_text_ts = datetime.min
        # Button rects are updated every frame in draw
        self._roast_rect = None
        self._now_rect = None
        self._mic_rect = None

        # Mic-based mode
        self.mic_mode_enabled = False
        self._mic_detector = None
        self._mic_last_trigger = datetime.min

        # Premade roasts
        self.premade_pairs = []  # list of tuples (skalle_path, ben_path)
        self.premade_idx = 0

ui_state = UIState()

# ── Helpers ───────────────────────────────────────────────────────────────
def encode_jpg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""

def annotate_and_labels(frame: np.ndarray, results):
    annotated = results.plot()  # ultralytics helper
    labels = {results.names[int(b.cls)] for b in results.boxes}
    y = 20
    for lbl in sorted(labels):
        cv2.putText(annotated, lbl, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 22
    return annotated, labels

def _draw_button(img, rect, text, active=False):
    x1, y1, x2, y2 = rect
    bg = (40, 40, 40)
    bg_active = (60, 120, 60)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_active if active else bg, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1)

def _wrap_text(text, max_width_px, scale=0.6, thickness=1):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw > max_width_px and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines

def draw_ui_overlay(frame: np.ndarray, state: UIState):
    h, w = frame.shape[:2]
    pad = 8
    btn_w, btn_h = 140, 32
    # Buttons at top-left
    roast_rect = (pad, pad, pad + btn_w, pad + btn_h)
    now_rect   = (pad*2 + btn_w, pad, pad*2 + btn_w*2, pad + btn_h)
    mic_rect   = (pad*3 + btn_w*2, pad, pad*3 + btn_w*3, pad + btn_h)

    # Save for click handling
    state._roast_rect = roast_rect
    state._now_rect = now_rect
    state._mic_rect = mic_rect

    _draw_button(frame, roast_rect, f"Roast: {'ON' if state.roast_enabled else 'OFF'}", active=state.roast_enabled)
    _draw_button(frame, now_rect, "Roast Now", active=False)
    _draw_button(frame, mic_rect, f"Mic Mode: {'ON' if state.mic_mode_enabled else 'OFF'}", active=state.mic_mode_enabled)
    # Mic backend + activity indicator
    if state.mic_mode_enabled and state._mic_detector is not None:
        backend = getattr(state._mic_detector, 'backend', 'Mic')
        devname = getattr(state._mic_detector, 'device_name', '')
        # Use simple ASCII separator to avoid glyph issues
        hint = f"{backend}{' - ' + devname if devname else ''}"
        (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = mic_rect[2] + 10
        ty = mic_rect[1] + (mic_rect[3]-mic_rect[1])//2 + th//2
        cv2.putText(frame, hint, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,0), 1)
        # activity dot
        active = False
        try:
            active = state._mic_detector.speech_recent(0.6)
        except Exception:
            active = False
        cx = mic_rect[0] - 10
        cy = mic_rect[1] + (mic_rect[3]-mic_rect[1])//2
        cv2.circle(frame, (cx, cy), 6, (0, 220, 0) if active else (80, 80, 80), -1)
        # Show gate state summary below mic button
        try:
            gate_text = getattr(state, '_gate_text', '')
            if gate_text:
                cv2.putText(frame, gate_text, (tx, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
        except Exception:
            pass

    # Render last OpenAI text as semi-transparent box at bottom
    if state.last_text:
        max_w = int(w * 0.9)
        lines = _wrap_text(state.last_text, max_w)
        scale, th = 0.6, 1
        lh = int(18 * scale) + 8
        box_h = lh * len(lines) + 16
        x1 = pad
        y2 = h - pad
        y1 = max(pad, y2 - box_h)
        x2 = x1 + max_w + 2*pad
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        y = y1 + 16
        for ln in lines:
            cv2.putText(frame, ln, (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), th)
            y += lh

def is_dark(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < DARK_LUMA_THRESH

def has_person_box(results) -> bool:
    return any(results.names[int(b.cls)] == "person" for b in results.boxes)

def load_premade_pairs(root: str):
    def numeric_key(name: str):
        import re as _re
        m = _re.search(r"\d+", name)
        if m:
            try:
                return (0, int(m.group(0)))
            except Exception:
                pass
        return (1, name.lower())

    pairs = []
    if not os.path.isdir(root):
        return pairs
    for sub in sorted(os.listdir(root), key=numeric_key):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        # accept common names, case-insensitive, common extensions
        skalle = None
        ben = None
        for fn in sorted(os.listdir(d)):
            low = fn.lower()
            if any(low.endswith(ext) for ext in (".mp3", ".wav", ".ogg")):
                if low.startswith("skalle") or "skalle" in low:
                    skalle = os.path.join(d, fn)
                elif low.startswith("ben") or "benrangel" in low:
                    ben = os.path.join(d, fn)
        if skalle and ben:
            pairs.append((skalle, ben))
    return pairs

# Optional mic VAD using webrtcvad + sounddevice; falls back to RMS
try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None
try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None
try:
    import torch  # type: ignore
except Exception:
    torch = None

def _try_load_silero():
    if torch is None:
        return None, None
    try:
        # Will use local torch hub cache if present; otherwise tries network
        model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
        return model, utils
    except Exception as e:
        print(f"Silero VAD not available ({e}); falling back.")
        return None, None

class SileroSpeechDetector:
    def __init__(self, rate: int = 16000):
        self.rate = rate
        self.running = False
        self.thread = None
        self.q = queue.Queue(maxsize=50)
        self.last_speech_ts = 0.0
        self._last_eval = 0.0
        self.window_sec = 0.8
        self.max_buffer_sec = 2.0
        self.samples = np.zeros(0, dtype=np.float32)
        self.in_rate = rate
        self.device_name = ""
        self.backend = "Silero"
        self.model, utils = _try_load_silero()
        if self.model is None:
            raise RuntimeError("Silero model not loaded")
        self.get_speech_timestamps = utils.get_speech_timestamps

    def _on_audio(self, indata, frames, time_info, status):
        if not self.running:
            return
        try:
            self.q.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _pick_device(self):
        dev_env = os.getenv("PREM_MIC_DEVICE")
        try:
            if dev_env and dev_env.isdigit():
                return int(dev_env)
        except Exception:
            pass
        try:
            devs = sd.query_devices()
        except Exception:
            devs = []
        # String match
        if dev_env and isinstance(dev_env, str):
            for i, d in enumerate(devs):
                if d.get('max_input_channels', 0) > 0 and dev_env.lower() in str(d.get('name','')).lower():
                    return i
        # Default input
        try:
            din, _ = sd.default.device
            if isinstance(din, int):
                return din
        except Exception:
            pass
        # First input-capable device
        for i, d in enumerate(devs):
            if d.get('max_input_channels', 0) > 0:
                return i
        return None

    def _device_default_rate(self, device):
        try:
            info = sd.query_devices(device)
            self.device_name = str(info.get('name', ''))
            dr = info.get('default_samplerate', None)
            if dr:
                return int(dr)
        except Exception:
            pass
        return self.rate

    def start(self):
        if self.running or sd is None:
            return False
        self.running = True
        try:
            device = self._pick_device()
            self.in_rate = self._device_default_rate(device)
            self.stream = sd.InputStream(samplerate=self.in_rate, channels=1, dtype='int16', callback=self._on_audio, device=device)
            self.stream.start()
        except Exception as e:
            self.running = False
            try:
                print("Mic start failed:", e)
                devs = sd.query_devices()
                print("Available input devices:")
                for i, d in enumerate(devs):
                    if d.get('max_input_channels', 0) > 0:
                        print(f"  {i}: {d.get('name')} (in={d.get('max_input_channels')})")
                print("Set PREM_MIC_DEVICE to index or part of name to select.")
            except Exception:
                pass
            return False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("Mic VAD backend: Silero")
        return True

    def stop(self):
        self.running = False
        try:
            if hasattr(self, 'stream'):
                self.stream.stop(); self.stream.close()
        except Exception:
            pass

    def _run(self):
        while self.running:
            try:
                chunk = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            # Resample to model rate if needed
            if self.in_rate != self.rate and arr.size > 0:
                n_out = max(1, int(round(arr.size * self.rate / self.in_rate)))
                x = np.linspace(0.0, 1.0, arr.size, endpoint=False)
                xi = np.linspace(0.0, 1.0, n_out, endpoint=False)
                arr = np.interp(xi, x, arr).astype(np.float32)
            if arr.size == 0:
                continue
            self.samples = np.concatenate([self.samples, arr])
            # keep last N seconds
            max_len = int(self.max_buffer_sec * self.rate)
            if self.samples.size > max_len:
                self.samples = self.samples[-max_len:]
            # evaluate at most 5 times per second
            now = time.time()
            if now - self._last_eval < 0.2:
                continue
            self._last_eval = now
            need = int(self.window_sec * self.rate)
            if self.samples.size < need:
                continue
            window = self.samples[-need:]
            wav = torch.from_numpy(window)
            try:
                with torch.no_grad():
                    ts = self.get_speech_timestamps(wav, self.model, sampling_rate=self.rate,
                                                    threshold=0.5, min_speech_duration_ms=80)
                if ts:
                    self.last_speech_ts = time.time()
            except Exception:
                # if anything goes wrong, disable running to avoid spam
                pass

    def speech_recent(self, within_sec: float = 1.0) -> bool:
        return (time.time() - self.last_speech_ts) <= within_sec

def create_speech_detector():
    # Prefer Silero if available
    if torch is not None:
        try:
            return SileroSpeechDetector()
        except Exception:
            pass
    # Fall back to WebRTC or NumPy
    return MicSpeechDetector()

class MicSpeechDetector:
    def __init__(self, aggressiveness: int = 2, rate: int = 16000, frame_ms: int = 30):
        self.rate = rate
        self.frame_len = int(rate * frame_ms / 1000)
        self.running = False
        self.thread = None
        self.q = queue.Queue(maxsize=50)
        self.last_speech_ts = 0.0
        self.vad = webrtcvad.Vad(aggressiveness) if webrtcvad else None
        self.in_rate = rate
        self.device_name = ""
        self.backend = "WebRTC" if self.vad is not None else "NumPy"

    def _rms(self, data_i16: np.ndarray) -> float:
        return float(np.sqrt(np.mean((data_i16.astype(np.float32)) ** 2)))

    def _band_energy_ratio(self, data_i16: np.ndarray, f_lo=300, f_hi=3400) -> float:
        """
        Pure-NumPy spectral VAD: ratio of energy in [f_lo, f_hi] to total energy.
        Works as a simple speech activity proxy without compiled deps.
        """
        x = data_i16.astype(np.float32)
        if x.size == 0:
            return 0.0
        # DC removal + Hann window
        x = x - x.mean()
        win = np.hanning(len(x)).astype(np.float32)
        X = np.fft.rfft(x * win)
        psd = (X.real**2 + X.imag**2)
        freqs = np.fft.rfftfreq(len(x), d=1.0/self.rate)
        total = float(psd.sum() + 1e-8)
        band = float(psd[(freqs >= f_lo) & (freqs <= f_hi)].sum())
        return band / total

    def _on_audio(self, indata, frames, time_info, status):
        if not self.running:
            return
        try:
            self.q.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _pick_device(self):
        dev_env = os.getenv("PREM_MIC_DEVICE")
        try:
            if dev_env and dev_env.isdigit():
                return int(dev_env)
        except Exception:
            pass
        try:
            devs = sd.query_devices()
        except Exception:
            devs = []
        # String match
        if dev_env and isinstance(dev_env, str):
            for i, d in enumerate(devs):
                if d.get('max_input_channels', 0) > 0 and dev_env.lower() in str(d.get('name','')).lower():
                    return i
        # Default input
        try:
            din, _ = sd.default.device
            if isinstance(din, int):
                return din
        except Exception:
            pass
        # First input-capable device
        for i, d in enumerate(devs):
            if d.get('max_input_channels', 0) > 0:
                return i
        return None

    def _device_default_rate(self, device):
        try:
            info = sd.query_devices(device)
            self.device_name = str(info.get('name', ''))
            dr = info.get('default_samplerate', None)
            if dr:
                return int(dr)
        except Exception:
            pass
        return self.rate

    def start(self):
        if self.running or sd is None:
            return False
        self.running = True
        try:
            device = self._pick_device()
            self.in_rate = self._device_default_rate(device)
            self.stream = sd.InputStream(samplerate=self.in_rate, channels=1, dtype='int16', callback=self._on_audio, device=device)
            self.stream.start()
        except Exception as e:
            self.running = False
            try:
                print("Mic start failed:", e)
                devs = sd.query_devices()
                print("Available input devices:")
                for i, d in enumerate(devs):
                    if d.get('max_input_channels', 0) > 0:
                        print(f"  {i}: {d.get('name')} (in={d.get('max_input_channels')})")
                print("Set PREM_MIC_DEVICE to index or part of name to select.")
            except Exception:
                pass
            return False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        try:
            if hasattr(self, 'stream'):
                self.stream.stop(); self.stream.close()
        except Exception:
            pass

    def _run(self):
        buf = b""
        frame_bytes = self.frame_len * 2  # int16
        while self.running:
            try:
                chunk = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            buf += chunk.tobytes()
            while len(buf) >= frame_bytes:
                frame = buf[:frame_bytes]; buf = buf[frame_bytes:]
                if self.vad is not None:
                    voiced = self.vad.is_speech(frame, self.in_rate)
                    if voiced:
                        self.last_speech_ts = time.time()
                else:
                    arr = np.frombuffer(frame, dtype=np.int16)
                    # Composite heuristic: energy + band energy ratio
                    # resample to 16k before spectral
                    if self.in_rate != self.rate and arr.size:
                        n_out = max(1, int(round(arr.size * self.rate / self.in_rate)))
                        x = np.linspace(0.0, 1.0, arr.size, endpoint=False)
                        xi = np.linspace(0.0, 1.0, n_out, endpoint=False)
                        arr = np.interp(xi, x, arr).astype(np.float32)
                    if self._rms(arr) > 500 and self._band_energy_ratio(arr) > 0.25:
                        self.last_speech_ts = time.time()

    def speech_recent(self, within_sec: float = 1.0) -> bool:
        return (time.time() - self.last_speech_ts) <= within_sec

def is_interesting(results, motion_pixels: int) -> bool:
    if motion_pixels < 1500:
        return False
    has_person = any(results.names[int(b.cls)] == "person" for b in results.boxes)
    has_vehicle = any(results.names[int(b.cls)] in {"car", "bus", "truck"} for b in results.boxes)
    return has_person and not has_vehicle

def describe(img_b64: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT_DESCRIBE,  # 👈 new
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": "Beskriv bilden enligt formatet ovan."},
                {"type": "input_image",
                 "image_url": f"data:image/jpeg;base64,{img_b64}"},
            ],
        }],
        max_output_tokens=180,
        temperature=0.6,
    )
    out = response.output_text.strip()
    print(out)
    return out


def roast(scene_desc: str) -> str:
    messages = [*history,
                {"role": "user",
                 "content": f"Scenbeskrivning: {scene_desc}\n\nSkriv nu ditt roast."}]
    rsp = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=messages,
        max_output_tokens=120
    )
    out = rsp.output_text
    print(out)
    history.append({"role": "assistant", "content": out})
    return out


SPEAKER_REGEX = re.compile(
    r'^\s*(Skalle[\-\s]?pär|Benrangel)\s*[:：]\s*(.+?)\s*$',
    re.IGNORECASE | re.MULTILINE
)

NAME_AT_END_REGEX = re.compile(
    r'[\s\-\—–,:]*\b(Skalle[\-\s]?pär|Benrangel)\b[\s\.\!\?]*$',
    re.IGNORECASE
)

def _normalize_text(s: str) -> str:
    # NFKC normalisering + ersätt vanliga “fultecken” till kolon/radslut
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("：", ":")
    return s

def _clean_line(text: str) -> str:
    # Ta bort citattecken/emojis i början, och ett ev. talarnamn i slutet
    text = text.strip().strip('“”"\'`')
    text = NAME_AT_END_REGEX.sub("", text).strip()
    return text

def assign_alternating_voices(raw: str):
    raw = _normalize_text(raw)
    # Plocka ut max två repliker med regex
    matches = SPEAKER_REGEX.findall(raw)

    # Om regex inte hittar exakt två, gör en försiktig fallback:
    if len(matches) < 2:
        # Splitta på rader, filtrera tomma
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) >= 2:
            # Försök tolka första som Skalle-pär, andra som Benrangel
            first_text  = _clean_line(re.sub(r'^\s*Skalle[\-\s]?pär\s*[:：]\s*', '', lines[0], flags=re.I))
            second_text = _clean_line(re.sub(r'^\s*Benrangel\s*[:：]\s*', '', lines[1], flags=re.I))
            return [(VOICE_SKALLEPAR, first_text), (VOICE_BENRANGEL, second_text)]
        else:
            return []

    # matches är lista av tuples (speaker, content)
    # Hämta de två första i rätt ordning: Skalle-pär först, sen Benrangel
    # Bygg en liten buffert per talare
    spk_map = {"skalle-pär": None, "skalle pär": None, "benrangel": None}
    ordered = []
    for spk, content in matches:
        key = spk.lower().replace("  ", " ").replace("–", "-").replace("—", "-").replace("  ", " ").replace("  ", " ")
        key = key.replace("skalle pär", "skalle-pär")  # normalisera ev. mellanslag
        txt = _clean_line(content)
        if "skalle" in key:
            spk_map["skalle-pär"] = txt if spk_map["skalle-pär"] is None else spk_map["skalle-pär"]
        elif "benrangel" in key:
            spk_map["benrangel"] = txt if spk_map["benrangel"] is None else spk_map["benrangel"]

    if spk_map["skalle-pär"] is None or spk_map["benrangel"] is None:
        return []

    return [
        (VOICE_SKALLEPAR, spk_map["skalle-pär"]),
        (VOICE_BENRANGEL, spk_map["benrangel"])
    ]


DIALECT_GUIDE = (
    "Voice: Klar och tydlig, men med bred småländsk dialekt – r:en rullar …\n"
    "Tone: Underfundigt neutral med punchline-känsla.\n"
    "Punctuation: kommatecken för luft, tankstreck för avbrott …\n"
    "Delivery: ca 1.5× tempo, mikro-pauser före poänger."
)

# ── TTS (new ElevenLabs SDK, in-memory) ───────────────────────────────────
def tts_bytes(text: str, voice_id: str) -> bytes:
    """
    Synthesize to memory using ElevenLabs SDK (no disk writes).
    Collects the streamed MP3 chunks into a single bytes object.
    """
    stream = eleven.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",   # or "eleven_turbo_v2_5"
        output_format="mp3_44100_128",       # MP3 stream
        # You can also pass 'optimize_streaming_latency' if needed
    )
    return b"".join(chunk for chunk in stream if isinstance(chunk, (bytes, bytearray)))

def play_bytes(mp3_bytes: bytes):
    """Play MP3 bytes entirely from memory using pygame."""
    snd = pygame.mixer.Sound(file=io.BytesIO(mp3_bytes))
    ch = snd.play()
    clock = pygame.time.Clock()
    while ch.get_busy():
        clock.tick(10)

async def synth_audio_async(text: str, voice: str) -> bytes:
    """ Kör ElevenLabs-synth i en bakgrundstråd och returnerar MP3-bytes. """
    return await asyncio.to_thread(tts_bytes, text, voice)

# ── Roast pipeline ────────────────────────────────────────────────────────
async def roast_once(frame: np.ndarray, labels):
    b64 = base64.b64encode(encode_jpg(frame)).decode()
    desc = await asyncio.to_thread(describe, b64)
    desc += " | Upptäckta objekt: " + ", ".join(sorted(labels)) + "."
    joke = await asyncio.to_thread(roast, desc)

    lines = assign_alternating_voices(joke)
    if not lines:
        return joke

    # ➊ starta bakgrunds-tasker direkt
    tasks = [asyncio.create_task(synth_audio_async(txt, vce))
             for vce, txt in lines]

    # ➋ vänta in den första och spela medan nästa laddas
    first_bytes = await tasks[0]
    play_bytes(first_bytes)

    # ➌ vänta in nästa (brukar redan vara klar), spela osv.
    for t in tasks[1:]:
        audio_bytes = await t
        play_bytes(audio_bytes)

    return joke

def play_premade_pair(paths):
    """Play a tuple/list of file paths sequentially using pygame mixer."""
    for p in paths:
        try:
            snd = pygame.mixer.Sound(p)
            ch = snd.play()
            clock = pygame.time.Clock()
            while ch.get_busy():
                clock.tick(10)
        except Exception as e:
            print(f"Failed to play {p}: {e}")

# ── Main loop ─────────────────────────────────────────────────────────────
async def main(cam=0):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError("Webcam unavailable")

    frame_i, last = 0, datetime.min

    # Print microphone devices if available
    if sd is not None:
        try:
            devs = sd.query_devices()
            print("Input devices (set PREM_MIC_DEVICE to index or name):")
            for i, d in enumerate(devs):
                if d.get('max_input_channels', 0) > 0:
                    print(f"  {i}: {d.get('name')} (in={d.get('max_input_channels')}, default_rate={d.get('default_samplerate')})")
        except Exception:
            pass

    # Load premade pairs at startup
    ui_state.premade_pairs = load_premade_pairs(PREMADE_DIR)
    if ui_state.premade_pairs:
        print(f"Loaded {len(ui_state.premade_pairs)} premade roast pairs from '{PREMADE_DIR}'.")
    else:
        print(f"No premade roasts found in '{PREMADE_DIR}'. Create subfolders with Skalle*/Ben* audio.")

    print("Skalle-pär & Benrangel spanar …  (Q = avsluta)")
    if SHOW_LIVE:
        cv2.namedWindow(WINDOW_NAME)

        # Mouse handler for simple on-screen buttons
        def on_mouse(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            r1 = getattr(ui_state, "_roast_rect", None)
            r2 = getattr(ui_state, "_now_rect", None)
            r3 = getattr(ui_state, "_mic_rect", None)
            if r1:
                x1, y1, x2, y2 = r1
                if x1 <= x <= x2 and y1 <= y <= y2:
                    ui_state.roast_enabled = not ui_state.roast_enabled
                    return
            if r2:
                x1, y1, x2, y2 = r2
                if x1 <= x <= x2 and y1 <= y <= y2:
                    ui_state.request_roast_now = True
                    return
            if r3:
                x1, y1, x2, y2 = r3
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Toggle mic mode and start/stop detector
                    ui_state.mic_mode_enabled = not ui_state.mic_mode_enabled
                    if ui_state.mic_mode_enabled:
                        ui_state._mic_detector = create_speech_detector()
                        started = ui_state._mic_detector.start()
                        if not started:
                            print("Mic mode requested but sounddevice not available.")
                            ui_state.mic_mode_enabled = False
                            ui_state._mic_detector = None
                    else:
                        if ui_state._mic_detector:
                            ui_state._mic_detector.stop()
                            ui_state._mic_detector = None
                    return

        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.01); continue

            # one YOLO inference + motion mask
            results = yolo(frame, verbose=False, classes=[0,2,5,7])[0]
            motion_pixels = cv2.countNonZero(bsub.apply(frame))
            vis, lbls = annotate_and_labels(frame, results)
            # Draw UI overlay (buttons + last text)
            draw_ui_overlay(vis, ui_state)

            if SHOW_LIVE:
                cv2.imshow(WINDOW_NAME, vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_i += 1
            if frame_i % FRAME_SKIP:
                continue

            trigger_auto = ui_state.roast_enabled and \
                is_interesting(results, motion_pixels) and \
                (datetime.now() - last).total_seconds() > GPT_COOLDOWN_SEC

            trigger_manual = ui_state.request_roast_now

            # Mic-mode trigger: configurable dark/person gating and cooldown
            trigger_mic = False
            if ui_state.mic_mode_enabled and ui_state._mic_detector is not None:
                dark_req = PREM_MIC_REQUIRE_DARK
                nop_req = PREM_MIC_REQUIRE_NO_PERSON
                dark_ok = is_dark(frame) if dark_req else True
                no_person_ok = (not has_person_box(results)) if nop_req else True
                speech_ok = ui_state._mic_detector.speech_recent(within_sec=1.0)
                cd_rem = MIC_COOLDOWN_SEC - (datetime.now() - ui_state._mic_last_trigger).total_seconds()
                cooldown_ok = cd_rem <= 0
                # stash gate text for overlay (show 'skip' when not required)
                dark_tag = ("skip" if not dark_req else ("True" if dark_ok else "False"))
                nop_tag = ("skip" if not nop_req else ("True" if no_person_ok else "False"))
                ui_state._gate_text = f"gate: dark={dark_tag} no_person={nop_tag} speech={'True' if speech_ok else 'False'} cd={'ok' if cooldown_ok else f'{cd_rem:.1f}s'}"
                if dark_ok and no_person_ok and speech_ok and cooldown_ok:
                    trigger_mic = True
                if PREM_MIC_DEBUG and speech_ok and not trigger_mic:
                    print("[MIC] gated:", {"dark_req": dark_req, "nop_req": nop_req, "dark_ok": dark_ok, "no_person_ok": no_person_ok, "cooldown_ok": cooldown_ok, "cd_rem": max(0.0, cd_rem)})

            if trigger_manual or trigger_auto:
                ui_state.request_roast_now = False
                joke_text = await roast_once(frame, lbls)
                if isinstance(joke_text, str) and joke_text:
                    ui_state.last_text = joke_text
                    ui_state.last_text_ts = datetime.now()
                last = datetime.now()
            elif trigger_mic and ui_state.premade_pairs:
                pair = ui_state.premade_pairs[ui_state.premade_idx % len(ui_state.premade_pairs)]
                ui_state.premade_idx += 1
                print("Mic trigger: playing premade pair:", pair)
                play_premade_pair(pair)
                ui_state._mic_last_trigger = datetime.now()

            await asyncio.sleep(0.002)
    finally:
        if ui_state._mic_detector:
            ui_state._mic_detector.stop()
        cap.release()
        cv2.destroyAllWindows()

# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
