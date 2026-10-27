# camroast/tts.py
"""Text to speech for the two skeletons, with two switchable providers.

- elevenlabs: the real skeleton voices, used on Halloween. Needs ELEVENLABS_API_KEY.
- piper: free and local on the CPU, plain Swedish voices for testing without using credits.

TTS_PROVIDER picks the start provider; the "TTS" button or the t key switches live.
If ElevenLabs fails (bad key, no network), the show plays a premade pair instead, or, with
TTS_FALLBACK_PIPER=1, says the line with Piper. A failed provider is skipped for a while.

Audio is played as 16-bit mono PCM through one sounddevice stream, streamed as it is
generated, so the first words play while the rest is still being made.
"""
import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import sounddevice as sd
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from .util import short_err

load_dotenv()
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

SKALLE = "skalle"
BEN = "ben"
PROVIDERS = ("elevenlabs", "piper")
LABELS = {"elevenlabs": "ElevenLabs", "piper": "Piper"}
SHORT = {"elevenlabs": "Eleven", "piper": "Piper"}  # fits on the TTS button

_PCM_RE = re.compile(r"^pcm_(\d+)$")
_play_lock = threading.Lock()  # one skeleton at a time


def pcm_rate(output_format: str) -> int | None:
    m = _PCM_RE.match(output_format or "")
    return int(m.group(1)) if m else None


def resolve_output_device(spec: str | None):
    """AUDIO_OUTPUT_DEVICE: index or part of the device name. None = system default."""
    if not spec:
        return None
    if spec.isdigit():
        return int(spec)
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0 and spec.lower() in str(d.get("name", "")).lower():
            return i
    print(f"Audio output device {spec!r} not found, using default")
    return None


@dataclass
class Clip:
    """A fully synthesized line, ready to play."""
    data: bytes
    rate: int | None   # PCM sample rate, or None for MP3
    provider: str


# ---------------------------------------------------------------- providers

class ElevenLabsTTS:
    name = "elevenlabs"

    def __init__(self, s):
        self.s = s
        self.voices = {SKALLE: s.voice_skallepar, BEN: s.voice_benrangel}
        self.rate = pcm_rate(s.tts_output_format)  # None = MP3 output

    def _kwargs(self, who, text):
        kw = dict(voice_id=self.voices[who], text=text, model_id=self.s.tts_model,
                  output_format=self.s.tts_output_format)
        if "v2_5" in self.s.tts_model:  # only the v2.5 models accept a language hint
            kw["language_code"] = "sv"
        return kw

    def stream(self, who, text) -> Iterator[bytes]:
        for ch in eleven.text_to_speech.stream(**self._kwargs(who, text)):
            if isinstance(ch, (bytes, bytearray)) and ch:
                yield bytes(ch)

    def stream_rate(self, who) -> int | None:
        return self.rate

    def synth(self, who, text) -> Clip:
        return Clip(b"".join(self.stream(who, text)), self.rate, self.name)

    def check(self) -> str | None:
        try:
            eleven.models.list()
            return None
        except Exception as e:
            if "missing_permissions" in str(getattr(e, "body", "")):
                return None  # the key works but may only make speech, which is all the show needs
            return short_err(e)


class PiperTTS:
    name = "piper"

    def __init__(self, s):
        self.s = s
        self.dir = Path(s.piper_dir)
        self.cfg = {
            SKALLE: (s.piper_voice_skallepar, s.piper_speed_skallepar, s.piper_speaker_skallepar),
            BEN: (s.piper_voice_benrangel, s.piper_speed_benrangel, s.piper_speaker_benrangel),
        }
        self._voices: dict = {}
        self._locks: dict = {}
        self._guard = threading.Lock()

    def _load(self, name: str):
        with self._guard:
            lock = self._locks.setdefault(name, threading.Lock())
        with lock:  # one lock per voice, so the two skeletons load in parallel
            if name in self._voices:
                return self._voices[name]
            import onnxruntime
            from piper import PiperVoice
            from piper.config import PiperConfig

            model = self.dir / f"{name}.onnx"
            if not model.exists() or not Path(f"{model}.json").exists():
                from piper.download_voices import download_voice
                print(f"Piper: downloading voice {name} to {self.dir}")
                self.dir.mkdir(parents=True, exist_ok=True)
                download_voice(name, self.dir)
            with open(f"{model}.json", "r", encoding="utf-8") as f:
                config = PiperConfig.from_dict(json.load(f))
            # Few threads: Piper is fast, and more threads fight YOLO and the camera loop for the CPU.
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = max(1, self.s.piper_threads)
            opts.inter_op_num_threads = 1
            session = onnxruntime.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
            voice = PiperVoice(session=session, config=config)
            self._voices[name] = voice
            return voice

    def _syn(self, who):
        from piper import SynthesisConfig
        name, speed, speaker = self.cfg[who]
        return name, SynthesisConfig(length_scale=speed or None, speaker_id=speaker)

    def stream(self, who, text) -> Iterator[bytes]:
        name, syn = self._syn(who)
        for chunk in self._load(name).synthesize(text, syn_config=syn):
            yield chunk.audio_int16_bytes

    def stream_rate(self, who) -> int:
        return int(self._load(self.cfg[who][0]).config.sample_rate)

    def synth(self, who, text) -> Clip:
        return Clip(b"".join(self.stream(who, text)), self.stream_rate(who), self.name)

    def check(self) -> str | None:
        try:
            for who in (SKALLE, BEN):
                self._load(self.cfg[who][0])
            return None
        except Exception as e:
            return short_err(e)

    def preload(self):
        """Start loading both voices in background threads (about 3 s each on first load)."""
        for name in {self.cfg[SKALLE][0], self.cfg[BEN][0]}:
            threading.Thread(target=self._try_load, args=(name,), daemon=True).start()

    def _try_load(self, name):
        try:
            self._load(name)
        except Exception as e:
            print(f"Piper: could not load {name}: {short_err(e)}")


# ---------------------------------------------------------------- playback

def _iter_bytes(data: bytes, size: int = 8192) -> Iterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i:i + size]


def open_output(rate: int, device=None):
    stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16", device=device)
    stream.start()
    return stream


def play_pcm(chunks: Iterable[bytes], rate: int, device=None, on_start: Callable | None = None,
             prebuffer_ms: int = 120, stream=None):
    """Play 16-bit mono PCM chunks as they arrive and return when the audio has finished.

    on_start fires right before the first audio is written. Pass an open stream to reuse it.
    """
    own = stream is None
    if own:
        stream = open_output(rate, device)
    target = int(rate * 2 * prebuffer_ms / 1000)
    prebuf = bytearray()
    carry = b""
    started = False
    t_first = None
    nbytes = 0
    try:
        for ch in chunks:
            if not ch:
                continue
            data = carry + bytes(ch)
            if len(data) % 2:
                carry, data = data[-1:], data[:-1]
            else:
                carry = b""
            if not started:
                prebuf += data
                if len(prebuf) < target:
                    continue
                data, prebuf = bytes(prebuf), bytearray()
                started = True
                t_first = time.time()
                if on_start:
                    on_start()
            stream.write(data)
            nbytes += len(data)
        if not started and prebuf:
            t_first = time.time()
            if on_start:
                on_start()
            stream.write(bytes(prebuf))
            nbytes += len(prebuf)
        if t_first is not None:
            # blocking writes return once queued; wait until the queued audio has actually played
            t_end = t_first + float(stream.latency or 0.0) + nbytes / 2.0 / rate
            remaining = t_end - time.time()
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if own:
            stream.stop()
            stream.close()


def play_mp3(data: bytes):
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    ch = pygame.mixer.Sound(file=io.BytesIO(data)).play()
    clock = pygame.time.Clock()
    while ch is not None and ch.get_busy():
        clock.tick(20)


# ---------------------------------------------------------------- facade

class Speaker:
    """Blocking helpers meant to be called from a worker thread (asyncio.to_thread).

    `who` is SKALLE or BEN; each provider maps it to its own voice.
    """

    def __init__(self, s, on_error: Callable | None = None):
        self.s = s
        self.on_error = on_error or (lambda where, e: print(f"ERROR: {where}: {short_err(e)}"))
        self.device = resolve_output_device(s.audio_output_device)
        self.providers = {"elevenlabs": ElevenLabsTTS(s), "piper": PiperTTS(s)}
        self.provider = s.tts_provider if s.tts_provider in self.providers else "elevenlabs"
        self._down_until: dict = {}
        self._stream = None
        self._stream_rate = None

    # ---- provider choice
    @property
    def label(self) -> str:
        return LABELS[self.provider]

    def set_provider(self, name: str):
        if name in self.providers:
            self.provider = name
            self._down_until.pop(name, None)  # an explicit choice gets a fresh try
            print(f"TTS: {LABELS[name]}")

    def preload(self):
        """Load local voices early if they may be used, so the first joke does not wait."""
        if self.provider == "piper" or self.s.tts_fallback_piper:
            self.providers["piper"].preload()

    def toggle(self) -> str:
        i = PROVIDERS.index(self.provider)
        self.set_provider(PROVIDERS[(i + 1) % len(PROVIDERS)])
        return self.provider

    def _order(self) -> list:
        order = [self.provider]
        if self.s.tts_fallback_piper and self.provider != "piper":
            order.append("piper")
        now = time.time()
        usable = [p for p in order if now >= self._down_until.get(p, 0.0)]
        return usable or order[-1:]

    def _failed(self, name: str, e: Exception, more_to_try: bool):
        if more_to_try:
            self._down_until[name] = time.time() + self.s.tts_retry_sec
            self.on_error(f"{LABELS[name]} (tar Piper)", e)
        else:
            self.on_error(LABELS[name], e)

    # ---- output
    def _output(self, rate: int):
        if self._stream is not None and self._stream_rate != rate:
            self._close_output()
        if self._stream is None:
            self._stream = open_output(rate, self.device)
            self._stream_rate = rate
        return self._stream

    def _close_output(self):
        st, self._stream, self._stream_rate = self._stream, None, None
        if st is not None:
            try:
                st.stop()
                st.close()
            except Exception:
                pass

    def _play(self, chunks, rate, on_start):
        with _play_lock:
            try:
                play_pcm(chunks, rate, on_start=on_start, stream=self._output(rate))
            except sd.PortAudioError:
                self._close_output()  # device may have changed; reopen next time
                raise

    # ---- speaking
    def speak_stream(self, who: str, text: str, on_start: Callable | None = None) -> str:
        """Synthesize and play one line, streaming. Returns the provider that spoke."""
        order = self._order()
        for i, name in enumerate(order):
            started = []

            def _started():
                started.append(True)
                if on_start:
                    on_start()

            prov = self.providers[name]
            try:
                rate = prov.stream_rate(who)
                if rate is None:  # MP3 output from ElevenLabs
                    self.play_clip(prov.synth(who, text), _started)
                else:
                    self._play(prov.stream(who, text), rate, _started)
                return name
            except Exception as e:
                if started:
                    raise  # audio already playing: do not start the line over in another voice
                self._failed(name, e, i + 1 < len(order))
                if i + 1 == len(order):
                    raise
        raise RuntimeError("no TTS provider")

    def prefetch(self, who: str, text: str) -> Clip:
        """Synthesize a whole line ahead of time (for the second skeleton)."""
        order = self._order()
        for i, name in enumerate(order):
            try:
                return self.providers[name].synth(who, text)
            except Exception as e:
                self._failed(name, e, i + 1 < len(order))
                if i + 1 == len(order):
                    raise
        raise RuntimeError("no TTS provider")

    def play_clip(self, clip: Clip, on_start: Callable | None = None):
        if not clip.data:
            return
        if clip.rate:
            self._play(_iter_bytes(clip.data), clip.rate, on_start)
        else:
            if on_start:
                on_start()
            with _play_lock:
                play_mp3(clip.data)

    def warmup(self) -> list:
        """Load/check every provider that may be used. Returns [(label, error or None)]."""
        names = [self.provider]
        if self.s.tts_fallback_piper and "piper" not in names:
            names.append("piper")
        out = []
        for name in names:
            err = self.providers[name].check()
            out.append((LABELS[name], err))
            if err and name != names[-1]:
                self._down_until[name] = time.time() + self.s.tts_retry_sec  # go straight to the backup
        try:
            rate = self.providers[self.provider].stream_rate(SKALLE) or 24000
            with _play_lock:
                self._output(rate)
        except Exception as e:
            out.append(("audio out", short_err(e)))
        return out
