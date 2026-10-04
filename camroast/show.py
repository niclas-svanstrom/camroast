# camroast/show.py
"""The show: everything that happens once the skeletons decide to speak.

Runs as an asyncio task so the camera loop keeps going. Slow work (OpenAI, ElevenLabs,
audio playback) happens in worker threads via asyncio.to_thread.
"""
import asyncio
import time
from collections import deque

from . import llm, stt, tts
from .llm import BEN, SKALLE
from .premade import play_premade_pair, play_random_attention
from .util import short_err
from .vision import crop_to_persons, to_b64_jpg


class ShowRunner:
    def __init__(self, s, ui, presence, get_voice, premade_pairs, attention_files, filler_files):
        self.s = s
        self.ui = ui
        self.presence = presence
        self.get_voice = get_voice                       # () -> VoiceListener | None
        self.premade_pairs = premade_pairs
        self.attention_files = attention_files
        self.filler_files = filler_files or attention_files
        self.speaker = tts.Speaker(s)
        self.recent_jokes: deque = deque(maxlen=max(1, s.joke_history_size))
        self.dialogue: deque = deque(maxlen=max(1, s.dialogue_history_size))
        self._dialogue_ts = 0.0
        self._premade_idx = 0
        self._task: asyncio.Task | None = None

    # ---- public API
    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def start_roast(self, frame, boxes, transcript: str | None = None) -> bool:
        return self._launch(self._roast(frame, boxes, transcript))

    def start_talkback(self, audio, rate, frame, boxes) -> bool:
        return self._launch(self._talkback(audio, rate, frame, boxes))

    def start_premade(self) -> bool:
        if not self.premade_pairs:
            self.error("inga förinspelade par hittades")
            return False
        return self._launch(self._premade())

    def error(self, where: str, exc: BaseException | None = None):
        msg = f"{where}: {short_err(exc)}" if exc is not None else where
        print("ERROR:", msg)
        self.ui.last_error = msg[:140]
        self.ui.error_until = time.time() + 6.0

    # ---- plumbing
    def _launch(self, coro) -> bool:
        if self.busy:
            coro.close()
            return False
        self._task = asyncio.create_task(self._guard(coro))
        return True

    async def _guard(self, coro):
        try:
            await coro
        except Exception as e:
            self.error("show", e)
        finally:
            self.ui.show_state = "idle"
            self._mute(False)

    def _mute(self, on: bool):
        v = self.get_voice()
        if v is not None:
            v.mute(on, tail_ms=self.s.mic_mute_tail_ms)
            if not on:
                v.drop_pending()

    def _subtitle(self, speaker: str, text: str):
        self.ui.subtitle_speaker = speaker
        self.ui.subtitle_text = text
        self.ui.subtitle_until = float("inf")

    def _finish(self):
        self._mute(False)
        self.ui.subtitle_until = time.time() + self.s.subtitle_hold_sec
        self.ui.show_state = "idle"
        self.presence.on_show_end(time.time())

    async def _wait_channel(self, ch, max_sec: float):
        """Wait for a pygame channel (attention clip) to finish, but never longer than max_sec."""
        t0 = time.time()
        while ch is not None and (time.time() - t0) < max_sec:
            try:
                if not ch.get_busy():
                    return
            except Exception:
                return
            await asyncio.sleep(0.05)

    # ---- flows
    async def _talkback(self, audio, rate, frame, boxes):
        self.ui.show_state = "transcribing"
        t0 = time.time()
        try:
            text = await asyncio.to_thread(stt.transcribe, audio, rate, model=self.s.stt_model, language=self.s.stt_language)
        except Exception as e:
            self.error("STT", e)
            return
        print(f"[{time.time() - t0:.1f}s] barn: {text!r}")
        if len(text) < self.s.stt_min_chars:
            return
        self.ui.child_text = text
        self.ui.child_text_until = time.time() + 20.0
        await self._roast(frame, boxes, transcript=text)

    async def _roast(self, frame, boxes, transcript: str | None = None):
        self.ui.show_state = "generating"
        self._mute(True)  # the attention clip must not be heard as a child talking
        clip = await asyncio.to_thread(play_random_attention, self.filler_files if transcript else self.attention_files)
        if time.time() - self._dialogue_ts > self.s.dialogue_reset_sec:
            self.dialogue.clear()

        b64 = None  # no camera: joke without a picture
        if frame is not None:
            img = crop_to_persons(frame, boxes) if boxes else frame
            b64 = to_b64_jpg(img, self.s.llm_image_max_side)
        t0 = time.time()
        try:
            lines = await asyncio.to_thread(
                llm.generate_lines,
                b64,
                model=self.s.llm_model,
                effort=self.s.llm_reasoning_effort,
                detail=self.s.llm_image_detail,
                transcript=transcript,
                dialogue=list(self.dialogue),
                recent_jokes=list(self.recent_jokes),
                timeout=self.s.llm_timeout_sec,
                service_tier=self.s.llm_service_tier,
            )
        except Exception as e:
            self.error("LLM", e)
            lines = None
        if not lines:
            await self._wait_channel(clip, 3.0)
            await self._play_premade()
            return

        sk, be = lines
        print(f"[{time.time() - t0:.1f}s] {SKALLE}: {sk}\n{' ' * 7}{BEN}: {be}")
        prefetch = asyncio.create_task(asyncio.to_thread(self.speaker.prefetch, be, self.s.voice_benrangel))
        await self._wait_channel(clip, 4.0)
        self.ui.show_state = "speaking"
        spoke = False
        try:
            await asyncio.to_thread(self.speaker.speak_stream, sk, self.s.voice_skallepar, lambda: self._subtitle(SKALLE, sk))
            spoke = True
            try:
                data = await prefetch
            except Exception as e:
                self.error("TTS", e)
                data = b""
            if data:
                await asyncio.to_thread(self.speaker.speak_bytes, data, lambda: self._subtitle(BEN, be))
        except Exception as e:
            self.error("TTS", e)
            prefetch.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        if not spoke:
            # the voices are down: let the skeletons say something anyway
            await self._play_premade()
            return
        self._finish()
        self.recent_jokes.append((sk, be))
        self.dialogue.append((transcript or "", sk, be))
        self._dialogue_ts = time.time()

    async def _premade(self):
        await self._play_premade()

    async def _play_premade(self):
        if not self.premade_pairs:
            self._finish()
            return
        pair = self.premade_pairs[self._premade_idx % len(self.premade_pairs)]
        self._premade_idx += 1
        self._mute(True)
        self.ui.show_state = "premade"
        self._subtitle("", "Förinspelat skämt")
        try:
            await asyncio.to_thread(play_premade_pair, pair)
        finally:
            self._finish()
