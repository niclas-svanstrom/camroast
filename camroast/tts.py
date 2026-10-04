# camroast/tts.py
"""ElevenLabs synthesis and audio playback.

PCM output formats are streamed straight into a sounddevice output stream, so the first
words play while the rest is still being generated. The stream stays open between lines
to avoid the device start-up delay. MP3 formats fall back to pygame.
"""
import io
import os
import re
import threading
import time
from typing import Callable, Iterable, Iterator

import sounddevice as sd
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from .util import short_err

load_dotenv()
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

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


def synth_stream(text: str, voice_id: str, *, model_id: str, output_format: str, language: str = "sv") -> Iterator[bytes]:
    kwargs = dict(voice_id=voice_id, text=text, model_id=model_id, output_format=output_format)
    if "v2_5" in model_id:  # only the v2.5 models accept a language hint
        kwargs["language_code"] = language
    return eleven.text_to_speech.stream(**kwargs)


def synth_bytes(text: str, voice_id: str, **kw) -> bytes:
    return b"".join(c for c in synth_stream(text, voice_id, **kw) if isinstance(c, (bytes, bytearray)))


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
    snd = pygame.mixer.Sound(file=io.BytesIO(data))
    ch = snd.play()
    clock = pygame.time.Clock()
    while ch is not None and ch.get_busy():
        clock.tick(20)


class Speaker:
    """Blocking helpers meant to be called from a worker thread (asyncio.to_thread)."""

    def __init__(self, s):
        self.s = s
        self.device = resolve_output_device(s.audio_output_device)
        self.rate = pcm_rate(s.tts_output_format)
        self._stream = None

    def _kw(self):
        return dict(model_id=self.s.tts_model, output_format=self.s.tts_output_format)

    def _output(self):
        if self._stream is None:
            self._stream = open_output(self.rate, self.device)
        return self._stream

    def _close_output(self):
        st, self._stream = self._stream, None
        if st is not None:
            try:
                st.stop()
                st.close()
            except Exception:
                pass

    def _play_pcm(self, chunks, on_start):
        with _play_lock:
            try:
                play_pcm(chunks, self.rate, on_start=on_start, stream=self._output())
            except Exception:
                self._close_output()  # device may have changed; reopen next time
                raise

    def speak_stream(self, text: str, voice_id: str, on_start: Callable | None = None):
        if self.rate:
            self._play_pcm(synth_stream(text, voice_id, **self._kw()), on_start)
        else:
            data = synth_bytes(text, voice_id, **self._kw())
            if on_start:
                on_start()
            with _play_lock:
                play_mp3(data)

    def prefetch(self, text: str, voice_id: str) -> bytes:
        return synth_bytes(text, voice_id, **self._kw())

    def speak_bytes(self, data: bytes, on_start: Callable | None = None):
        if not data:
            return
        if self.rate:
            self._play_pcm(_iter_bytes(data), on_start)
        else:
            if on_start:
                on_start()
            with _play_lock:
                play_mp3(data)

    def warmup(self) -> str | None:
        """Open the output device and the HTTPS connection, check the key. Returns an error string or None."""
        if self.rate:
            try:
                with _play_lock:
                    self._output()
            except Exception as e:
                return f"audio out: {e}"
        try:
            eleven.models.list()
            return None
        except Exception as e:
            return short_err(e)
