# camroast/stt.py
"""Speech-to-text for the talk-back mode (OpenAI transcription models)."""
import io
import re
import wave

import numpy as np

from .llm import client

STT_PROMPT = "Halloween. Bus eller godis. Skalle-Pär och Benrangel är två skelett."


def _phrase(text: str) -> str:
    """Regex for the words of text in order, whatever the case, punctuation or hyphens between them."""
    return r"\W*".join(re.escape(w) for w in re.findall(r"[^\W\d_]+", text))


# With unclear audio (a crowd, wind) the model can answer with the prompt itself. A kid may well say
# "bus eller godis" but never the prompt's last sentence, so an echo is any tail of it including that.
_PROMPT_SENTENCES = re.split(r"(?<=[.!?])\s+", STT_PROMPT)
_ECHO = re.compile("|".join(_phrase(" ".join(_PROMPT_SENTENCES[i:])) for i in range(len(_PROMPT_SENTENCES))),
                   re.IGNORECASE)

# Not words: hesitation (eh, öhm, hmm, mm), surprise (ah, oh, ojoj) and laughter (haha, hihi).
_NOISE = re.compile(r"e+h+m*|e+m+|ö+h*m*|h+m+|m+h?m*|u+h*m*|a+h+|å+h*|o+h+|(o+j)+|(h+[aeio]+)+h*")
_NOTES = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")  # [skratt], (skrattar), *laughs*


def worth_answering(text: str, min_letters: int) -> bool:
    """False for a transcript of only hesitation, laughter or noise notes, so "Haha!" or "Ehh..."
    get no answer. "Ja" or "Ehh, en pirat" do: min_letters counts the letters in real words."""
    words = re.findall(r"[^\W\d_]+", _NOTES.sub(" ", text.lower()))
    return sum(len(w) for w in words if not _NOISE.fullmatch(w)) >= min_letters


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
    text = (getattr(rsp, "text", "") or "").strip()
    if _ECHO.search(text):
        cleaned = _ECHO.sub("", text).strip(" .,!?;:-")
        print(f"stt: dropped the prompt echoed back: {text!r} -> {cleaned!r}")
        return cleaned
    return text
