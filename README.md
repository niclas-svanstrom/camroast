# CamRoast

Two skeletons, Skalle-Pär and Benrangel, watch the porch through a camera and comment on
the trick-or-treaters in Swedish. Jokes come from OpenAI (gpt-6-luna by default), the voices
from ElevenLabs. With the mic on, the skeletons also answer what people say.

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
| q   | Quit |

The status line shows what is going on (Tänker, Pratar, Redo, Väntar på nya barn), the
frame rate and YOLO time. Subtitles at the bottom show what the skeletons say while they
say it, and what the mic heard.

## How a show works

1. YOLO sees a person; after `PERSON_CONFIRM_SEC` (0.5 s) of continuous presence the show starts.
2. An attention clip plays while the best recent frame, cropped around the people, goes to the model.
3. The model returns one line per skeleton as JSON. Skalle-Pär's line streams from ElevenLabs
   straight to the speaker while Benrangel's line is fetched in parallel.
4. Afterwards the skeletons wait until the scene has been empty for `REARM_CLEAR_SEC`, or
   `REARM_SAME_SCENE_SEC` if the same group stays.

Talk-back: the mic is segmented into utterances by a voice activity detector, transcribed with
`gpt-4o-mini-transcribe`, and the transcript plus the current frame go to the model. The mic is
muted while the skeletons talk so they never answer themselves.

If OpenAI or ElevenLabs fail, a premade pair from `media/premade` plays instead and the error
shows on screen for a few seconds.

## Layout

- `camroast/app.py` camera loop, detection thread, triggers
- `camroast/presence.py` time-based confirmation, best frame, re-arm
- `camroast/show.py` the show as an asyncio task
- `camroast/llm.py` prompt and structured output
- `camroast/stt.py`, `camroast/tts.py`, `camroast/voice.py` speech in and out
- `camroast/ui.py` overlay and subtitles
- `camroast/premade.py` pre-recorded clips

All settings are documented in `.env.example`.
