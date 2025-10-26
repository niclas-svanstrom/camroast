import os
import threading
import queue
import time
import numpy as np

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
        if dev_env and isinstance(dev_env, str):
            for i, d in enumerate(devs):
                if d.get('max_input_channels', 0) > 0 and dev_env.lower() in str(d.get('name','')).lower():
                    return i
        try:
            din, _ = sd.default.device
            if isinstance(din, int):
                return din
        except Exception:
            pass
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
            if self.in_rate != self.rate and arr.size > 0:
                n_out = max(1, int(round(arr.size * self.rate / self.in_rate)))
                x = np.linspace(0.0, 1.0, arr.size, endpoint=False)
                xi = np.linspace(0.0, 1.0, n_out, endpoint=False)
                arr = np.interp(xi, x, arr).astype(np.float32)
            if arr.size == 0:
                continue
            self.samples = np.concatenate([self.samples, arr])
            max_len = int(self.max_buffer_sec * self.rate)
            if self.samples.size > max_len:
                self.samples = self.samples[-max_len:]
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
                pass

    def speech_recent(self, within_sec: float = 1.0) -> bool:
        return (time.time() - self.last_speech_ts) <= within_sec


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
        x = data_i16.astype(np.float32)
        if x.size == 0:
            return 0.0
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
        if dev_env and isinstance(dev_env, str):
            for i, d in enumerate(devs):
                if d.get('max_input_channels', 0) > 0 and dev_env.lower() in str(d.get('name','')).lower():
                    return i
        try:
            din, _ = sd.default.device
            if isinstance(din, int):
                return din
        except Exception:
            pass
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
        frame_bytes = self.frame_len * 2
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
                    if self.in_rate != self.rate and arr.size:
                        n_out = max(1, int(round(arr.size * self.rate / self.in_rate)))
                        x = np.linspace(0.0, 1.0, arr.size, endpoint=False)
                        xi = np.linspace(0.0, 1.0, n_out, endpoint=False)
                        arr = np.interp(xi, x, arr).astype(np.float32)
                    if self._rms(arr) > 500 and self._band_energy_ratio(arr) > 0.25:
                        self.last_speech_ts = time.time()

    def speech_recent(self, within_sec: float = 1.0) -> bool:
        return (time.time() - self.last_speech_ts) <= within_sec


def create_speech_detector():
    if torch is not None:
        try:
            return SileroSpeechDetector()
        except Exception:
            pass
    return MicSpeechDetector()

