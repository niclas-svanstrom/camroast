#!/usr/bin/env python3
"""
Real-time roasting skeletons – Skalle-pär & Benrangel (no file I/O)
-------------------------------------------------------------------
• GPT-4o-mini vision  → scene description & jokes
• ElevenLabs TTS      → Swedish fast-paced, dialect voices
• YOLOv8n             → person-/car-detection & live overlay
• Pygame              → playback (from memory, no disk writes)
"""

# ── Imports ───────────────────────────────────────────────────────────────
import asyncio, base64, hashlib, os, io
from collections import deque
from datetime import datetime
import cv2
import numpy as np
from openai import OpenAI
import pygame
from dotenv import load_dotenv
from ultralytics import YOLO
import re
import unicodedata

# NEW: ElevenLabs SDK
from elevenlabs.client import ElevenLabs

# ── Config ────────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

SHOW_LIVE        = True          # set False for head-less deploy
FPS              = 10
FRAME_SKIP       = 5
GPT_COOLDOWN_SEC = 10

VOICE_SKALLEPAR  = "NHVO1d5lgqVtAvyYNL2P"   # e.g. "Adam"
VOICE_BENRANGEL  = "S6pZEFGfrgnWx4AETPdD"

SYSTEM_PROMPT = (
    "Du är två sarkastiska skelett som står på gatan och roastar folk som går förbi. "
    "Var kvick, syrlig och självironisk – ni skojar både om personen och varandra. "
    "Varje roast består exakt av två repliker: "
    "rad 1 börjar med 'Skalle-pär:' och rad 2 börjar med 'Benrangel:'. "
    "Ibland använder ni varandras namn i replikerna, naturligt i början eller mitten av meningen "
    "(t.ex. 'Titta där, Skalle-pär...' eller 'Du har rätt, Benrangel...'), inte efteråt. "
    "Blanda in skämt om att ni är skelett och låt dialogen kännas som ett snabbt, bitskt gattsnack. "
    "Skriv alltid exakt två meningar totalt (en per rad). "
    "Exempel:\n"
    "Skalle-pär: Titta där, Benrangel, den där killen ser ut som han tappade sin spegel för tio år sen!\n"
    "Benrangel: Haha, snälla Skalle-pär, vi har mer kött på benen än han har självkänsla!"
)


# ── Singletons ────────────────────────────────────────────────────────────
yolo  = YOLO("yolov8n.pt")
bsub  = cv2.createBackgroundSubtractorMOG2(120, 50)
pygame.mixer.init()
history = deque(maxlen=6)

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
        input=[{
            "role": "user",
            "content": [
                { "type": "input_text", "text": "Beskriv vad du ser och roa publiken." },
                { "type": "input_image",
                  "image_url": f"data:image/jpeg;base64,{img_b64}" },
            ],
        }],
        max_output_tokens=120
    )
    return response.output_text

def roast(scene_desc: str) -> str:
    messages = [*history,
                {"role": "user",
                 "content": f"Scenbeskrivning: {scene_desc}\n\nSkriv nu ditt roast."}]
    rsp = client.responses.create(
        model="gpt-5-mini",
        instructions=SYSTEM_PROMPT,
        input=messages,
        max_output_tokens=120
    )
    out = rsp.output_text
    print(out)
    history.append({"role": "assistant", "content": out})
    return out


SPEAKER_REGEX = re.compile(
    r'^\s*(Skalle[\-\s]?pär|Benrangel)\s*[:：]\s*(.+?)\s*$',
    re.IGNORECASE | re.MULTILINE
)

NAME_AT_END_REGEX = re.compile(
    r'[\s\-\—–,:]*\b(Skalle[\-\s]?pär|Benrangel)\b[\s\.\!\?]*$',
    re.IGNORECASE
)

def _normalize_text(s: str) -> str:
    # NFKC normalisering + ersätt vanliga “fultecken” till kolon/radslut
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("：", ":")
    return s

def _clean_line(text: str) -> str:
    # Ta bort citattecken/emojis i början, och ett ev. talarnamn i slutet
    text = text.strip().strip('“”"\'`')
    text = NAME_AT_END_REGEX.sub("", text).strip()
    return text

def assign_alternating_voices(raw: str):
    raw = _normalize_text(raw)
    # Plocka ut max två repliker med regex
    matches = SPEAKER_REGEX.findall(raw)

    # Om regex inte hittar exakt två, gör en försiktig fallback:
    if len(matches) < 2:
        # Splitta på rader, filtrera tomma
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) >= 2:
            # Försök tolka första som Skalle-pär, andra som Benrangel
            first_text  = _clean_line(re.sub(r'^\s*Skalle[\-\s]?pär\s*[:：]\s*', '', lines[0], flags=re.I))
            second_text = _clean_line(re.sub(r'^\s*Benrangel\s*[:：]\s*', '', lines[1], flags=re.I))
            return [(VOICE_SKALLEPAR, first_text), (VOICE_BENRANGEL, second_text)]
        else:
            return []

    # matches är lista av tuples (speaker, content)
    # Hämta de två första i rätt ordning: Skalle-pär först, sen Benrangel
    # Bygg en liten buffert per talare
    spk_map = {"skalle-pär": None, "skalle pär": None, "benrangel": None}
    ordered = []
    for spk, content in matches:
        key = spk.lower().replace("  ", " ").replace("–", "-").replace("—", "-").replace("  ", " ").replace("  ", " ")
        key = key.replace("skalle pär", "skalle-pär")  # normalisera ev. mellanslag
        txt = _clean_line(content)
        if "skalle" in key:
            spk_map["skalle-pär"] = txt if spk_map["skalle-pär"] is None else spk_map["skalle-pär"]
        elif "benrangel" in key:
            spk_map["benrangel"] = txt if spk_map["benrangel"] is None else spk_map["benrangel"]

    if spk_map["skalle-pär"] is None or spk_map["benrangel"] is None:
        return []

    return [
        (VOICE_SKALLEPAR, spk_map["skalle-pär"]),
        (VOICE_BENRANGEL, spk_map["benrangel"])
    ]


DIALECT_GUIDE = (
    "Voice: Klar och tydlig, men med bred småländsk dialekt – r:en rullar …\n"
    "Tone: Underfundigt neutral med punchline-känsla.\n"
    "Punctuation: kommatecken för luft, tankstreck för avbrott …\n"
    "Delivery: ca 1.5× tempo, mikro-pauser före poänger."
)

# ── TTS (new ElevenLabs SDK, in-memory) ───────────────────────────────────
def tts_bytes(text: str, voice_id: str) -> bytes:
    """
    Synthesize to memory using ElevenLabs SDK (no disk writes).
    Collects the streamed MP3 chunks into a single bytes object.
    """
    stream = eleven.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",   # or "eleven_turbo_v2_5"
        output_format="mp3_44100_128",       # MP3 stream
        # You can also pass 'optimize_streaming_latency' if needed
    )
    return b"".join(chunk for chunk in stream if isinstance(chunk, (bytes, bytearray)))

def play_bytes(mp3_bytes: bytes):
    """Play MP3 bytes entirely from memory using pygame."""
    snd = pygame.mixer.Sound(file=io.BytesIO(mp3_bytes))
    ch = snd.play()
    clock = pygame.time.Clock()
    while ch.get_busy():
        clock.tick(10)

async def synth_audio_async(text: str, voice: str) -> bytes:
    """ Kör ElevenLabs-synth i en bakgrundstråd och returnerar MP3-bytes. """
    return await asyncio.to_thread(tts_bytes, text, voice)

# ── Roast pipeline ────────────────────────────────────────────────────────
async def roast_once(frame: np.ndarray, labels):
    b64 = base64.b64encode(encode_jpg(frame)).decode()
    desc = await asyncio.to_thread(describe, b64)
    desc += " | YOLO såg: " + ", ".join(sorted(labels)) + "."
    joke = await asyncio.to_thread(roast, desc)

    lines = assign_alternating_voices(joke)
    if not lines:
        return

    # ➊ starta bakgrunds-tasker direkt
    tasks = [asyncio.create_task(synth_audio_async(txt, vce))
             for vce, txt in lines]

    # ➋ vänta in den första och spela medan nästa laddas
    first_bytes = await tasks[0]
    play_bytes(first_bytes)

    # ➌ vänta in nästa (brukar redan vara klar), spela osv.
    for t in tasks[1:]:
        audio_bytes = await t
        play_bytes(audio_bytes)

# ── Main loop ─────────────────────────────────────────────────────────────
async def main(cam=0):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError("Webcam unavailable")

    frame_i, last = 0, datetime.min

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
                await roast_once(frame, lbls)
                last = datetime.now()

            await asyncio.sleep(0.002)
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
