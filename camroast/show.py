# camroast/show.py
"""The show: everything that happens once the skeletons decide to speak.

Runs as an asyncio task so the camera loop keeps going. Slow work (OpenAI, text to speech,
audio playback) happens in worker threads via asyncio.to_thread.
"""
import asyncio
import os
import time
from collections import deque

from . import jokelog, llm, stt, tts
from .llm import BEN, SKALLE
from .premade import play_premade_pair, play_random_attention
from .util import short_err
from .vision import crop_to_persons, to_b64_jpg
from .voice import RATE


class ShowRunner:
    def __init__(self, s, ui, presence, get_voice, premade_pairs, attention_files, filler_files):
        self.s = s
        self.ui = ui
        self.presence = presence
        self.get_voice = get_voice                       # () -> VoiceListener | None
        self.premade_pairs = premade_pairs
        self.attention_files = attention_files
        self.filler_files = filler_files or attention_files
        self.speaker = tts.Speaker(s, on_error=self.error)
        self.recent_jokes: deque = deque(maxlen=max(1, s.joke_history_size))
        self.dialogue: deque = deque(maxlen=max(1, s.dialogue_history_size))
        self._dialogue_ts = 0.0
        self._premade_idx = 0
        self._task: asyncio.Task | None = None
        self._why = ""                                   # what started the current show, for the joke log

    # ---- public API (why: what started the show, shown in the joke log)
    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def start_roast(self, frame, boxes, transcript: str | None = None, why: str = "") -> bool:
        return self._launch(self._roast(frame, boxes, transcript), why)

    def start_talkback(self, audio, rate, frame, boxes, greet: bool = False, why: str = "") -> bool:
        """greet: a new group, so this is the photo roast with what they said added (attention clip, not a filler)."""
        return self._launch(self._talkback(audio, rate, frame, boxes, greet), why)

    def start_premade(self, why: str = "") -> bool:
        if not self.premade_pairs:
            self.error("inga förinspelade par hittades")
            return False
        return self._launch(self._premade(), why)

    def error(self, where: str, exc: BaseException | None = None):
        msg = f"{where}: {short_err(exc)}" if exc is not None else where
        print("ERROR:", msg)
        self.ui.last_error = msg[:140]
        self.ui.error_until = time.time() + 6.0
        jokelog.write(self.s.log_dir, f"Fel: {msg}")

    # ---- plumbing
    def _launch(self, coro, why: str = "") -> bool:
        if self.busy:
            coro.close()
            return False
        self._why = why
        self._task = asyncio.create_task(self._guard(coro))
        self.presence.on_show_start(time.time())
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

    def _log(self, *lines: str):
        jokelog.write(self.s.log_dir, self._why, *lines)

    def _subtitle(self, speaker: str, text: str):
        self.ui.subtitle_speaker = speaker
        self.ui.subtitle_text = text
        self.ui.subtitle_until = float("inf")

    def _finish(self):
        self._mute(False)
        self.ui.subtitle_until = time.time() + self.s.subtitle_hold_sec
        self.ui.show_state = "idle"
        self.presence.on_show_end(time.time())

    def _ask(self, b64, transcript):
        """Start a joke request in a worker thread."""
        return asyncio.create_task(asyncio.to_thread(
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
        ))

    async def _heard_before_clip(self) -> str | None:
        """What people said just before the skeletons started, transcribed, or None.

        The camera's sound is late, so it is still coming in while the attention clip plays; the
        listener cuts it off where the clip begins. A local mic has it ready right away.
        """
        v = self.get_voice()
        if v is None or self.s.mic_action == "premade":
            return None
        await asyncio.sleep(max(0.0, v.mute_from - time.time()) + 0.3)
        heard = v.take_all()
        if heard is None:
            return None
        audio, speech_sec = heard
        t0 = time.time()
        try:
            text = await asyncio.to_thread(stt.transcribe, audio, RATE, model=self.s.stt_model,
                                           language=self.s.stt_language, timeout=4.0)
        except Exception as e:
            self.error("STT", e)
            return None
        print(f"[{time.time() - t0:.1f}s] barn ({speech_sec:.1f}s före klippet): {text!r}")
        if not stt.worth_answering(text, self.s.stt_min_chars):
            return None
        self.ui.child_text = text
        self.ui.child_text_until = time.time() + 20.0
        return text

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
    async def _talkback(self, audio, rate, frame, boxes, greet=False):
        self.ui.show_state = "transcribing"
        clip = None
        if greet:  # a new group is greeted right away: the roast follows whatever they said
            self._mute(True)  # the clip must not be heard as a child talking
            clip = await asyncio.to_thread(play_random_attention, self.attention_files)
        t0 = time.time()
        try:
            text = await asyncio.to_thread(stt.transcribe, audio, rate, model=self.s.stt_model, language=self.s.stt_language)
        except Exception as e:
            self.error("STT", e)
            text = ""
        else:
            print(f"[{time.time() - t0:.1f}s] barn: {text!r}")
        if not stt.worth_answering(text, self.s.stt_min_chars):
            if greet:  # nothing usable was said: still roast the picture, as with the mic off
                await self._roast(frame, boxes, clip=clip)
            elif text:
                self._log(f'Barn: "{text}" (inget svar)')
            return  # a laugh or an "ehh" gets no answer, and no filler either
        # between roasts the filler starts in _roast, now that there is something to answer
        self.ui.child_text = text
        self.ui.child_text_until = time.time() + 20.0
        await self._roast(frame, boxes, transcript=text, clip=clip)

    async def _roast(self, frame, boxes, transcript: str | None = None, clip=None):
        self.ui.show_state = "generating"
        self._mute(True)  # the attention clip must not be heard as a child talking
        if clip is None:  # a greeted talk-back has already started its clip
            clip = await asyncio.to_thread(play_random_attention, self.filler_files if transcript else self.attention_files)
        if time.time() - self._dialogue_ts > self.s.dialogue_reset_sec:
            self.dialogue.clear()

        b64 = None  # no camera: joke without a picture
        if frame is not None:
            img = crop_to_persons(frame, boxes) if boxes else frame
            b64 = to_b64_jpg(img, self.s.llm_image_max_side)
        t0 = time.time()
        job = self._ask(b64, transcript)
        if transcript is None:
            # The joke on the picture alone is on its way. If they said something just before the
            # clip, ask again with their words too and use that one instead.
            heard = await self._heard_before_clip()
            if heard:
                transcript, t0 = heard, time.time()
                job.add_done_callback(lambda t: t.cancelled() or t.exception())
                job = self._ask(b64, heard)
                await self._wait_channel(clip, 4.0)
                if not job.done():  # a filler covers the wait for the new joke
                    clip = await asyncio.to_thread(play_random_attention, self.filler_files)
        try:
            lines = await job
        except Exception as e:
            self.error("LLM", e)
            lines = None
        if not lines:
            await self._wait_channel(clip, 3.0)
            await self._play_premade("inget skämt från OpenAI")
            return

        sk, be = lines
        print(f"[{time.time() - t0:.1f}s] {SKALLE}: {sk}\n{' ' * 7}{BEN}: {be}")
        prefetch = asyncio.create_task(asyncio.to_thread(self.speaker.prefetch, tts.BEN, be))
        await self._wait_channel(clip, 4.0)
        self.ui.show_state = "speaking"
        spoke = False
        try:
            await asyncio.to_thread(self.speaker.speak_stream, tts.SKALLE, sk, lambda: self._subtitle(SKALLE, sk))
            spoke = True
            try:
                ben_clip = await prefetch
            except Exception:
                ben_clip = None  # already reported by the speaker
            if ben_clip is not None:
                await asyncio.to_thread(self.speaker.play_clip, ben_clip, lambda: self._subtitle(BEN, be))
        except Exception:
            prefetch.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        if not spoke:
            # the voices are down: let the skeletons say something anyway
            await self._play_premade("rösterna fungerade inte")
            return
        heard = [f'Barn: "{transcript}"'] if transcript else []
        self._log(*heard, f"{SKALLE}: {sk}", f"{BEN}: {be}")
        self._finish()
        self.recent_jokes.append((sk, be))
        self.dialogue.append((transcript or "", sk, be))
        self._dialogue_ts = time.time()

    async def _premade(self):
        await self._play_premade()

    async def _play_premade(self, reason: str = ""):
        if not self.premade_pairs:
            self._finish()
            return
        pair = self.premade_pairs[self._premade_idx % len(self.premade_pairs)]
        self._premade_idx += 1
        self._log(f"Förinspelat skämt {os.path.basename(os.path.dirname(pair[0]))}" + (f" ({reason})" if reason else ""))
        self._mute(True)
        self.ui.show_state = "premade"
        self._subtitle("", "Förinspelat skämt")
        try:
            await asyncio.to_thread(play_premade_pair, pair)
        finally:
            self._finish()
