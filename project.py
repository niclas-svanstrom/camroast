#!/usr/bin/env python3
"""
Real-time roasting skeletons – Skalle-pär & Benrangel
----------------------------------------------------
• GPT-4o-mini vision  → scene description & jokes
• OpenAI TTS-1        → Swedish fast-paced, dialect voices
• YOLOv8n             → person-/car-detection & live overlay
• Pygame              → playback
"""

# ── Imports ───────────────────────────────────────────────────────────────
import asyncio, base64, hashlib, os
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI
import pygame
from dotenv import load_dotenv
from ultralytics import YOLO
import requests


ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{}"

# ── Config ────────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# VOICE_SKALLEPAR  = "ash"
# VOICE_BENRANGEL  = "coral"
SHOW_LIVE        = True          # set False for head-less deploy
FPS              = 10
FRAME_SKIP       = 5
GPT_COOLDOWN_SEC = 10
MEDIA_ROOT       = Path("media")
CACHE_DIR        = Path("cache_audio")
KEEP_DAYS        = 3

VOICE_SKALLEPAR  = "S6pZEFGfrgnWx4AETPdD"   # e.g. "Adam"
VOICE_BENRANGEL  = "NHVO1d5lgqVtAvyYNL2P"

SYSTEM_PROMPT = (
    "Du är två sarkastiska skelett … Max två meningar totalt. "
    "Varje roast består exakt av två repliker: "
    "rad 1 börjar med 'Skalle-pär:' och rad 2 börjar med 'Benrangel:'. "
    "Blanda även in er själva och skoja med varandra."
)

# ── Singletons ────────────────────────────────────────────────────────────
yolo  = YOLO("yolov8n.pt")
bsub  = cv2.createBackgroundSubtractorMOG2(120, 50)
pygame.mixer.init()
CACHE_DIR.mkdir(exist_ok=True)
history = deque(maxlen=10)

# ── Helpers ───────────────────────────────────────────────────────────────
def encode_jpg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""

def annotate_and_labels(frame: np.ndarray, results):
    annotated = results.plot()  # ultralytics helper
    labels = {results.names[int(b.cls)] for b in results.boxes}
    y = 20
    for lbl in sorted(labels):
        cv2.putText(annotated, lbl, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 22
    return annotated, labels

def is_interesting(results, motion_pixels: int) -> bool:
    if motion_pixels < 1500:
        return False
    has_person = any(results.names[int(b.cls)] == "person" for b in results.boxes)
    has_vehicle = any(results.names[int(b.cls)] in {"car", "bus", "truck"} for b in results.boxes)
    return has_person and not has_vehicle

def describe(img_b64: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "Beskriv vad du ser och roa publiken." },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{img_b64}",
                    },
                ],
            }
        ],
        max_output_tokens=120
    )
    return response.output_text  

def roast(scene_desc: str) -> str:
    messages = [*history,
                {"role": "user",
                 "content": f"Scenbeskrivning: {scene_desc}\n\nSkriv nu ditt roast."}]
    rsp = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=messages,
        max_output_tokens=120
    )
    out = rsp.output_text
    history.append({"role": "assistant", "content": out})
    return out

def assign_alternating_voices(raw: str):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    def strip_pref(l,p): return l.split(":",1)[1].strip() if l.lower().startswith(p) else l
    first  = strip_pref(lines[0], "skalle-pär:")
    second = strip_pref(lines[1], "benrangel:")
    return [(VOICE_SKALLEPAR, first), (VOICE_BENRANGEL, second)]

DIALECT_GUIDE = (
    "Voice: Klar och tydlig, men med bred småländsk dialekt – r:en rullar …\n"
    "Tone: Underfundigt neutral med punchline-känsla.\n"
    "Punctuation: kommatecken för luft, tankstreck för avbrott …\n"
    "Delivery: ca 1.5× tempo, mikro-pauser före poänger."
)

# def tts(text: str, voice: str) -> Path:
#     p = CACHE_DIR / f"{voice}_{hashlib.md5(text.encode()).hexdigest()}.mp3"
#     if p.exists():
#         return p
#     with client.audio.speech.with_streaming_response.create(
#             model="tts-1",
#             voice=voice,
#             input=text,
#             instructions=DIALECT_GUIDE
#     ) as s:
#         s.stream_to_file(p)
#     return p

def tts(text: str, voice_id: str) -> Path:
    path = CACHE_DIR / f"11_{voice_id}_{hashlib.md5(text.encode()).hexdigest()}.mp3"
    if path.exists():
        return path

    payload = {
        "text"      : text,
        "model_id"  : "eleven_multilingual_v2",   # or "eleven_turbo_v2_5"
        "voice_settings": {                       # optional fine-tune
            "stability"        : 0.31,
            "similarity_boost" : 0.97,
            "style"            : 0.50,
            "use_speaker_boost": True
        }
    }
    headers = {
        "xi-api-key"  : ELEVEN_KEY,
        "Content-Type": "application/json"
    }
    r = requests.post(ELEVEN_URL.format(voice_id), json=payload, headers=headers, timeout=45)
    r.raise_for_status()                       # -> 429 if quota hit
    path.write_bytes(r.content)                # MP3 bytes in body
    return path

def play(path: Path):
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

async def synth_audio_async(text: str, voice: str) -> Path:
    """ Kör ElevenLabs-synth i en bakgrundstråd och returnerar Path. """
    return await asyncio.to_thread(tts, text, voice)


# ── Roast pipeline ────────────────────────────────────────────────────────
async def roast_once(frame: np.ndarray, labels, out_dir: Path):
    b64 = base64.b64encode(encode_jpg(frame)).decode()
    desc = await asyncio.to_thread(describe, b64)
    desc += " | YOLO såg: " + ", ".join(sorted(labels)) + "."
    joke = await asyncio.to_thread(roast, desc)

    lines = assign_alternating_voices(joke)
    if not lines:                      # inget att säga
        return

    # ➊ starta bakgrunds-tasker direkt
    tasks = [asyncio.create_task(synth_audio_async(txt, vce))
             for vce, txt in lines]

    # ➋ vänta in den första,
    #    spela den medan nästa fortfarande syntetiseras
    path_first = await tasks[0]
    play(path_first)                   # blockar tills klart

    # ➌ vänta in nästa (brukar redan vara klar), spela osv.
    for t in tasks[1:]:
        path = await t
        play(path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"{ts}.jpg").write_bytes(base64.b64decode(b64))

# ── Main loop ─────────────────────────────────────────────────────────────
async def main(cam=0):
    MEDIA_ROOT.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError("Webcam unavailable")

    frame_i, last = 0, datetime.min
    out_dir = MEDIA_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(exist_ok=True)

    print("Skalle-pär & Benrangel spanar …  (tryck Q för att avsluta)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.01); continue

            # one YOLO inference + motion mask
            results = yolo(frame, verbose=False, classes=[0,2,5,7])[0]
            motion_pixels = cv2.countNonZero(bsub.apply(frame))
            vis, lbls = annotate_and_labels(frame, results)

            if SHOW_LIVE:
                cv2.imshow("RoastCam", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_i += 1
            if frame_i % FRAME_SKIP:
                continue

            if is_interesting(results, motion_pixels) and \
               (datetime.now() - last).total_seconds() > GPT_COOLDOWN_SEC:
                await roast_once(frame, lbls, out_dir)
                last = datetime.now()

            await asyncio.sleep(0.002)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # clean old media
        cutoff = datetime.now() - timedelta(days=KEEP_DAYS)
        for p in MEDIA_ROOT.rglob("*"):
            if p.is_file() and datetime.fromtimestamp(p.stat().st_mtime) < cutoff:
                try: p.unlink()
                except: pass

# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
