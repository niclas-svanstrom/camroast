# camroast/stt.py
"""Speech-to-text for the talk-back mode (OpenAI transcription models)."""
import io
import wave

import numpy as np

from .llm import client

STT_PROMPT = "Halloween. Bus eller godis. Skalle-Pär och Benrangel är två skelett."


def wav_from_int16(samples: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


def transcribe(samples: np.ndarray, rate: int, *, model: str, language: str = "sv", timeout: float = 15.0) -> str:
    wav = wav_from_int16(samples, rate)
    rsp = client.audio.transcriptions.create(
        model=model,
        file=("speech.wav", wav, "audio/wav"),
        language=language,
        prompt=STT_PROMPT,
        response_format="json",
        timeout=timeout,
    )
    return (getattr(rsp, "text", "") or "").strip()
