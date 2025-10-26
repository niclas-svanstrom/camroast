import os
import re
import unicodedata
from collections import deque
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Voice IDs (can be overridden by env if needed)
VOICE_SKALLEPAR = os.getenv("VOICE_SKALLEPAR", "NHVO1d5lgqVtAvyYNL2P")
VOICE_BENRANGEL = os.getenv("VOICE_BENRANGEL", "S6pZEFGfrgnWx4AETPdD")

SYSTEM_PROMPT = (
    "Du är två sarkastiska skelett – Skalle-pär och Benrangel – som står på gatan och roastar förbipasserande.\n"
    "MÅL: En kvick tvåraders dialog på svenska som känns improviserad och publikvänlig.\n"
    "\n"
    "FORMAT (obligatoriskt):\n"
    "1) Skalle-pär: <en (1) mening>\n"
    "2) Benrangel: <en (1) mening>\n"
    "Exakt två meningar totalt. Inga extra rader, inga emojis.\n"
    "\n"
    "STIL:\n"
    "• Syrlig, självironisk, snabb gatuton. Blanda in att ni är skelett.\n"
    "• Ni kan nämna varandras namn naturligt i början eller mitten av meningen (inte i slutet).\n"
    "• Skämta om situationen, kläder, rörelser, rekvisita – inte om känsliga attribut (ålder, kropp, hälsa, religion, etnicitet, identitet).\n"
    "\n"
    "STENHÅRDA REGLER (inga undantag):\n"
    "• Gör ALDRIG meta-referenser till kamera, bild, AI, modell, detektion, YOLO, neurala nät, algoritmer eller 'jag ser'.\n"
    "• Påstå inte hur ni vet saker – ni står bara där och kommenterar.\n"
    "• Inga uppmaningar, inga förklaringar, ingen extra text före/efter replikerna.\n"
    "\n"
    "OM NÅGON REGEL BRYTS: skriv om direkt tills allt följer reglerna.\n"
    "\n"
    "EXEMPEL (OK):\n"
    "Skalle-pär: Titta där, Benrangel, den kappan svajar som om den flytt från en storm!\n"
    "Benrangel: Du har rätt, Skalle-pär — jag har sett mer stadga i mina lösa leder!\n"
    "\n"
    "EXEMPEL (FÖRBJUDET):\n"
    "Skalle-pär: YOLO sa att en person närmar sig…\n"
    "Benrangel: Kameran fångade det — vi analyserade bilden!\n"
)

SYSTEM_PROMPT_DESCRIBE = (
    "Du beskriver en gatukameras­cen på svenska för ett humorsegment.\n"
    "Var neutral, respektfull och precis. Gör aldrig antaganden om ålder, identitet, kropp, hälsa, religion eller etnicitet.\n"
    "Fokusera på kläder, rörelser, föremål och situationer. Om osäker: markera det.\n"
    "Avsluta med en lättsam oneliner som driver med situationen (inte personen), max 12 ord.\n"
    "Format:\n"
    "Beskrivning: …\n"
    "Oneliner: …"
)

# Keep a short conversation history (for tone consistency)
history = deque(maxlen=6)


def describe(img_b64: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT_DESCRIBE,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Beskriv bilden enligt formatet ovan."},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"},
            ],
        }],
        max_output_tokens=180,
        temperature=0.6,
    )
    out = response.output_text.strip()
    print(out)
    return out


def roast(scene_desc: str) -> str:
    messages = [*history, {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Scenbeskrivning: {scene_desc}\n\nSkriv nu ditt roast."}]
    rsp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
        max_output_tokens=120,
        temperature=0.8,
    )
    out = rsp.output_text.strip()
    history.append({"role": "assistant", "content": out})
    return out


SPEAKER_REGEX = re.compile(r'^\s*(Skalle[\-\s]?pär|Benrangel)\s*[:\-]\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
NAME_AT_END_REGEX = re.compile(r'[\s\-,:]*\b(Skalle[\-\s]?pär|Benrangel)\b[\s\.\!\?]*$', re.IGNORECASE)


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace(":s", ":")  # fallback for odd encodings
    return s


def _clean_line(text: str) -> str:
    text = text.strip().strip('“”’‘"\'`')
    text = NAME_AT_END_REGEX.sub("", text).strip()
    return text


def assign_alternating_voices(raw: str, voice_skallepar: str = VOICE_SKALLEPAR, voice_benrangel: str = VOICE_BENRANGEL):
    raw = _normalize_text(raw)
    matches = SPEAKER_REGEX.findall(raw)

    if len(matches) < 2:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) >= 2:
            first_text = _clean_line(re.sub(r'^\s*Skalle[\-\s]?pär\s*[:\-]\s*', '', lines[0], flags=re.I))
            second_text = _clean_line(re.sub(r'^\s*Benrangel\s*[:\-]\s*', '', lines[1], flags=re.I))
            return [(VOICE_SKALLEPAR, first_text), (VOICE_BENRANGEL, second_text)]
        return []

    spk_map = {"skalle-pär": None, "benrangel": None}
    for spk, content in matches:
        key = spk.lower().replace("skalle pär", "skalle-pär")
        txt = _clean_line(content)
        if "skalle" in key:
            spk_map["skalle-pär"] = spk_map["skalle-pär"] or txt
        elif "benrangel" in key:
            spk_map["benrangel"] = spk_map["benrangel"] or txt

    if spk_map["skalle-pär"] is None or spk_map["benrangel"] is None:
        return []

    return [
        (VOICE_SKALLEPAR, spk_map["skalle-pär"]),
        (VOICE_BENRANGEL, spk_map["benrangel"]),
    ]
