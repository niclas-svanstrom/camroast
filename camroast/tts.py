import io
import os
import asyncio
import pygame
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

_mixer_ready = False

def _ensure_mixer():
    global _mixer_ready
    if not _mixer_ready:
        pygame.mixer.init()
        _mixer_ready = True


def tts_bytes(text: str, voice_id: str) -> bytes:
    stream = eleven.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    return b"".join(chunk for chunk in stream if isinstance(chunk, (bytes, bytearray)))


def play_bytes(mp3_bytes: bytes):
    _ensure_mixer()
    snd = pygame.mixer.Sound(file=io.BytesIO(mp3_bytes))
    ch = snd.play()
    clock = pygame.time.Clock()
    while ch.get_busy():
        clock.tick(10)


async def synth_audio_async(text: str, voice: str) -> bytes:
    return await asyncio.to_thread(tts_bytes, text, voice)
