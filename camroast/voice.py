# camroast/voice.py
"""Microphone capture, voice activity detection and utterance segmentation.

VoiceListener runs in a background thread and produces complete utterances
(16 kHz int16 numpy arrays) that the app can hand to speech-to-text. While muted
(the skeletons are talking) all audio is discarded so they never answer themselves.
"""
import math
import queue
import threading
import time
from collections import deque

import numpy as np

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None
try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None
try:
    import torch  # type: ignore
except Exception:
    torch = None

RATE = 16000


def list_input_devices():
    if sd is None:
        return []
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append((i, str(d.get("name", ""))))
    return out


def pick_input_device(spec: str | None):
    """MIC_DEVICE: index or part of the device name. None = system default input."""
    if sd is None:
        return None
    if spec and spec.isdigit():
        return int(spec)
    devs = list_input_devices()
    if spec:
        for i, name in devs:
            if spec.lower() in name.lower():
                return i
        print(f"Mic {spec!r} not found, using default")
    try:
        din = sd.default.device[0]
        if isinstance(din, int) and din >= 0:
            return din
    except Exception:
        pass
    return devs[0][0] if devs else None


# ---- VAD backends: is_speech(int16 chunk) -> bool, chunk = samples per call at 16 kHz

class _SileroVad:
    name = "Silero"
    chunk = 512  # 32 ms, required by Silero v5 at 16 kHz

    def __init__(self, threshold: float = 0.5):
        if torch is None:
            raise RuntimeError("torch missing")
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self.model.eval()
        self.threshold = threshold

    def is_speech(self, chunk_i16: np.ndarray) -> bool:
        x = torch.from_numpy(chunk_i16.astype(np.float32) / 32768.0)
        with torch.no_grad():
            p = float(self.model(x, RATE).item())
        return p >= self.threshold

    def reset(self):
        try:
            self.model.reset_states()
        except Exception:
            pass


class _WebRtcVad:
    name = "WebRTC"
    chunk = 480  # 30 ms

    def __init__(self, aggressiveness: int = 2):
        if webrtcvad is None:
            raise RuntimeError("webrtcvad missing")
        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, chunk_i16: np.ndarray) -> bool:
        return bool(self.vad.is_speech(chunk_i16.tobytes(), RATE))

    def reset(self):
        pass


class _EnergyVad:
    name = "Energy"
    chunk = 480

    def __init__(self, rms_thresh: float = 500.0, band_ratio: float = 0.25):
        self.rms_thresh = rms_thresh
        self.band_ratio = band_ratio

    def is_speech(self, chunk_i16: np.ndarray) -> bool:
        x = chunk_i16.astype(np.float32)
        if x.size == 0:
            return False
        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms < self.rms_thresh:
            return False
        x = x - x.mean()
        spec = np.fft.rfft(x * np.hanning(len(x)).astype(np.float32))
        psd = spec.real ** 2 + spec.imag ** 2
        freqs = np.fft.rfftfreq(len(x), d=1.0 / RATE)
        band = float(psd[(freqs >= 300) & (freqs <= 3400)].sum())
        return band / float(psd.sum() + 1e-8) > self.band_ratio

    def reset(self):
        pass


def make_vad():
    for cls in (_SileroVad, _WebRtcVad):
        try:
            return cls()
        except Exception as e:
            print(f"VAD {cls.name} not available: {e}")
    return _EnergyVad()


class VoiceListener:
    def __init__(self, s):
        self.s = s
        self.vad = None
        self.backend = ""
        self.device_name = ""
        self.in_rate = RATE
        self.running = False
        self.last_speech_ts = 0.0
        self._q: queue.Queue = queue.Queue(maxsize=400)
        self._utts: queue.Queue = queue.Queue(maxsize=4)
        self._thread = None
        self._stream = None
        self._muted = False
        self._mute_until = 0.0
        self._lock = threading.Lock()

    # ---- lifecycle
    def start(self) -> bool:
        if self.running or sd is None:
            return False
        if self.vad is None:
            self.vad = make_vad()
            self.backend = self.vad.name
        device = pick_input_device(self.s.mic_device)
        try:
            info = sd.query_devices(device) if device is not None else sd.query_devices(kind="input")
            self.device_name = str(info.get("name", ""))
            dev_rate = int(info.get("default_samplerate") or RATE)
        except Exception:
            self.device_name, dev_rate = "", RATE
        # Prefer capturing at 16 kHz directly; otherwise take the device rate and resample.
        for rate in dict.fromkeys((RATE, dev_rate)):
            try:
                self._stream = sd.InputStream(samplerate=rate, channels=1, dtype="int16", device=device, callback=self._on_audio)
                self._stream.start()
                self.in_rate = rate
                break
            except Exception as e:
                self._stream = None
                print(f"Mic open at {rate} Hz failed: {e}")
        if self._stream is None:
            print("Available input devices:")
            for i, name in list_input_devices():
                print(f"  {i}: {name}")
            print("Set MIC_DEVICE to an index or part of the name.")
            return False
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"Mic: {self.backend} on {self.device_name!r} at {self.in_rate} Hz")
        return True

    def stop(self):
        self.running = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None

    # ---- gating
    def mute(self, muted: bool, tail_ms: int = 0):
        with self._lock:
            self._muted = muted
            self._mute_until = float("inf") if muted else time.time() + tail_ms / 1000.0

    def is_muted(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self._muted or now < self._mute_until

    # ---- queries
    def speech_recent(self, within_sec: float = 1.0) -> bool:
        return (time.time() - self.last_speech_ts) <= within_sec

    def poll_utterance(self):
        """(int16 samples at 16 kHz, seconds of speech) or None."""
        try:
            return self._utts.get_nowait()
        except queue.Empty:
            return None

    def drop_pending(self):
        while True:
            try:
                self._utts.get_nowait()
            except queue.Empty:
                return

    # ---- internals
    def _on_audio(self, indata, frames, time_info, status):
        if not self.running:
            return
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _resample(self, x: np.ndarray) -> np.ndarray:
        if self.in_rate == RATE:
            return x
        from scipy.signal import resample_poly
        g = math.gcd(RATE, self.in_rate)
        y = resample_poly(x.astype(np.float32), RATE // g, self.in_rate // g)
        return np.clip(y, -32768, 32767).astype(np.int16)

    def _run(self):
        chunk = self.vad.chunk
        chunk_ms = chunk * 1000.0 / RATE
        buf = np.zeros(0, dtype=np.int16)
        preroll: deque = deque(maxlen=max(1, int(300 / chunk_ms)))
        utt: list = []
        in_speech = False
        silence_ms = 0.0
        speech_ms = 0.0

        def reset():
            nonlocal utt, in_speech, silence_ms, speech_ms
            utt, in_speech, silence_ms, speech_ms = [], False, 0.0, 0.0
            preroll.clear()
            self.vad.reset()

        while self.running:
            try:
                block = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            block = self._resample(block)
            buf = np.concatenate([buf, block]) if buf.size else block
            while buf.size >= chunk:
                c, buf = buf[:chunk], buf[chunk:]
                if self.is_muted():
                    if in_speech or preroll:
                        reset()
                    continue
                try:
                    speech = bool(self.vad.is_speech(c))
                except Exception:
                    speech = False
                if speech:
                    self.last_speech_ts = time.time()
                if not in_speech:
                    preroll.append(c)
                    if speech:
                        in_speech = True
                        utt = list(preroll)
                        silence_ms, speech_ms = 0.0, chunk_ms
                    continue
                utt.append(c)
                if speech:
                    silence_ms = 0.0
                    speech_ms += chunk_ms
                else:
                    silence_ms += chunk_ms
                if silence_ms >= self.s.mic_end_silence_ms or len(utt) * chunk_ms >= self.s.mic_max_utterance_sec * 1000:
                    self._finish(utt, speech_ms, silence_ms, chunk_ms)
                    reset()

    def _finish(self, utt, speech_ms, silence_ms, chunk_ms):
        if speech_ms < self.s.mic_min_speech_ms:
            if self.s.mic_debug:
                print(f"mic: dropped short blip ({speech_ms:.0f} ms)")
            return
        keep_tail = int(250 / chunk_ms)
        trailing = int(silence_ms / chunk_ms)
        if trailing > keep_tail:
            utt = utt[: len(utt) - (trailing - keep_tail)]
        audio = np.concatenate(utt)
        if self.s.mic_debug:
            print(f"mic: utterance {len(audio) / RATE:.2f}s ({speech_ms:.0f} ms speech)")
        try:
            self._utts.put_nowait((audio, speech_ms / 1000.0))
        except queue.Full:
            pass
