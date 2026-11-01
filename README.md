# CamRoast

Two skeletons, Skalle-Pär and Benrangel, watch the porch through a camera and comment on
the trick-or-treaters in Swedish. Jokes come from OpenAI (gpt-6-luna by default), the voices
from ElevenLabs or from Piper, a free local voice engine. With the mic on, the skeletons also answer what people say.

## Run

```
.venv\Scripts\activate
python project.py
```

Copy `.env.example` to `.env`, fill in `OPENAI_API_KEY` and `ELEVENLABS_API_KEY`, and set
`CAM_SOURCE` (webcam index or an rtsp:// URL; a video file also works and loops, handy for testing).

The app starts even if the camera is unreachable. It shows a "no camera" screen, keeps retrying
in the background, and reconnects by itself if the stream drops. Buttons, mic talk-back and
premade clips work meanwhile; "Roast now" then jokes without a picture.

## Controls

Buttons in the window or keys:

| Key | Action |
|-----|--------|
| r   | Roast on/off (auto mode) |
| n   | Roast now |
| m   | Mic on/off (talk-back) |
| p   | Play a premade pair |
| t   | Switch voices: ElevenLabs / Piper |
| q   | Quit |

The status line shows what is going on (Tänker, Pratar, Redo, Väntar på nya barn), the
frame rate and YOLO time. Subtitles at the bottom show what the skeletons say while they
say it, and what the mic heard.

## Voices

- **ElevenLabs** is what the skeletons use on Halloween. It needs `ELEVENLABS_API_KEY`, a key
  that starts with `sk_`. The app uses the Flash v2.5 model, about 160 characters per show.
- **Piper** is free and runs locally on the CPU, handy for testing without using credits.
  Skalle-Pär uses the Swedish male voice `sv_SE-nst-medium`, Benrangel the female
  `sv_SE-lisa-medium`. The voices download to `models/piper` the first time.

`TTS_PROVIDER` picks the start voice and the TTS button (or t) switches live. If ElevenLabs
fails, a premade pair plays instead; `TTS_FALLBACK_PIPER=1` says the line with Piper instead.

## How a show works

1. YOLO sees a person; after `PERSON_CONFIRM_SEC` (0.5 s) of continuous presence the show starts.
2. An attention clip plays while the best recent frame, cropped around the people, goes to the model.
3. The model returns one line per skeleton as JSON. Skalle-Pär's line streams from ElevenLabs
   straight to the speaker while Benrangel's line is fetched in parallel.
4. Afterwards the skeletons wait until the scene has been empty for `REARM_CLEAR_SEC`, or
   `REARM_SAME_SCENE_SEC` if the same group stays. If more people are in view than when the
   last show started, steadily for `PERSON_CONFIRM_SEC`, the next show starts right away
   (`REARM_MORE_PEOPLE`).

Talk-back: the mic is segmented into utterances by a voice activity detector, transcribed with
`gpt-4o-mini-transcribe`, and the transcript plus the current frame go to the model. With both
Roast and Mic on, the voice backs up the photo: the attention clip plays the moment a new group is
seen, the joke on the picture is requested right away, and if they said something just before the
clip, the joke is asked again with their words (a filler clip covers the wait). Between roasts,
anything said by someone in the picture gets its own answer with a fresh picture, so the skeletons
can chat while they wait; voices from outside the picture are ignored (without a camera, everyone
is answered). Nonstop chatter is cut off after `MIC_MAX_UTTERANCE_SEC` (5 s) and answered, so the
skeletons keep going. Laughter, hums and hesitation ("haha", "mm", "ehh") get no answer; a new
group then gets the joke on the picture alone.

The mic is muted while the skeletons talk so they never answer themselves. With `MIC_DEVICE=camera`
the mic is the RTSP camera's own (a Tapo sends 8 kHz audio when its microphone is on). Its audio
arrives late, so the mute is shifted: it starts `MIC_CAMERA_LAG_MIN_MS` after the skeletons start
(what came in before that was said before they did, and is kept) and lasts `MIC_CAMERA_LAG_MS`
after they stop. `MIC_DEBUG=1` prints the real lag at every show; test alone at the camera.

Every show is written to a joke log, `logs/skamt-YYYY-MM-DD.txt` (`LOG_DIR`): the time, what started
it (picture, voice, newcomers, button), what the kids said and both lines. Premade pairs, unanswered
laughs and errors are in it too, so afterwards you can see why it went quiet or played a premade.

If OpenAI (jokes) or ElevenLabs (voices) fail, a premade pair from `media/premade` plays instead and the error
shows on screen for a few seconds.

## Layout

- `camroast/app.py` camera loop, detection thread, triggers
- `camroast/presence.py` time-based confirmation, best frame, re-arm
- `camroast/show.py` the show as an asyncio task
- `camroast/llm.py` prompt and structured output
- `camroast/stt.py`, `camroast/tts.py`, `camroast/voice.py` speech in and out
- `camroast/ui.py` overlay and subtitles
- `camroast/premade.py` pre-recorded clips
- `camroast/jokelog.py` the joke log

All settings are documented in `.env.example`.
