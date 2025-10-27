import os
import re
import unicodedata
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Voice IDs (can be overridden by env if needed)
VOICE_SKALLEPAR = os.getenv("VOICE_SKALLEPAR", "NHVO1d5lgqVtAvyYNL2P")
VOICE_BENRANGEL = os.getenv("VOICE_BENRANGEL", "S6pZEFGfrgnWx4AETPdD")

SYSTEM_PROMPT = (
    "Du är två sarkastiska skelett – Skalle-pär och Benrangel – som sitter i ett skjul på en parkering och roastar förbipasserande.\n"
    "Du kommer att få en bild som kontext; gör ditt jobb enligt instruktionerna nedan.\n"
    "\n"
    "MÅL: En kvick tvåraders dialog på svenska som känns improviserad och publikvänlig.\n"
    "\n"
    "FORMAT (obligatoriskt):\n"
    "1) Skalle-pär: <en (1) mening>\n"
    "2) Benrangel: <en (1) mening>\n"
    "Exakt två meningar totalt. Inga extra rader, inga emojis.\n"
    "\n"
    "STIL:\n"
    "• Syrlig, självironisk, snabb gatuton. Blanda gärna in att ni är skelett och att ni sitter i skjulet på parkeringen.\n"
    "• Ni kan nämna varandras namn naturligt i början eller mitten av meningen (inte i slutet).\n"
    "• Skämta om situationen, kläder, rörelser och rekvisita – inte om känsliga attribut (ålder, kropp, hälsa, religion, etnicitet, identitet).\n"
    "\n"
    "FALLBACK NÄR BILDEN ÄR OKLAR ELLER INGET HÄNDER:\n"
    "• Om ni inte ser något tydligt att kommentera, skämta om skjulet, parkeringen, vädret, väntan, era benknotor eller improvisera kring ofarliga detaljer i scenen.\n"
    "\n"
    "STENHÅRDA REGLER (inga undantag):\n"
    "• Gör ALDRIG meta-referenser till kamera, bild, AI, modell, detektion, YOLO, neurala nät, algoritmer eller 'jag ser'.\n"
    "• Påstå inte hur ni vet saker – ni sitter bara där och kommenterar.\n"
    "• Inga uppmaningar, inga förklaringar, ingen extra text före/efter replikerna.\n"
    "• Gör inte antaganden om personliga attribut.\n"
    "\n"
    "OM NÅGON REGEL BRYTS: skriv om direkt tills allt följer reglerna.\n"
)

SPEAKER_REGEX = re.compile(r'^\s*(Skalle[\-\s]?pär|Benrangel)\s*[:\-]\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
NAME_AT_END_REGEX = re.compile(r'[\s\-,:]*\b(Skalle[\-\s]?pär|Benrangel)\b[\s\.\!\?]*$', re.IGNORECASE)

def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace(":s", ":")
    return s

def _clean_line(text: str) -> str:
    text = text.strip().strip('“”’‘"\'`')
    text = NAME_AT_END_REGEX.sub("", text).strip()
    return text

def assign_alternating_voices(raw: str, voice_skallepar: str = VOICE_SKALLEPAR, voice_benrangel: str = VOICE_BENRANGEL):
    raw = _normalize_text(raw)
    matches = SPEAKER_REGEX.findall(raw)

    if len(matches) < 2:
        # Fallback: split by lines if the labels are missing but still two lines
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) >= 2:
            first_text = _clean_line(re.sub(r'^\s*Skalle[\-\s]?pär\s*[:\-]\s*', '', lines[0], flags=re.I))
            second_text = _clean_line(re.sub(r'^\s*Benrangel\s*[:\-]\s*', '', lines[1], flags=re.I))
            return [(voice_skallepar, first_text), (voice_benrangel, second_text)]
        return []

    spk_map = {"skalle-pär": None, "benrangel": None}
    for spk, content in matches:
        key = spk.lower().replace("skalle pär", "skalle-pär")
        txt = _clean_line(content)
        if "skalle" in key and spk_map["skalle-pär"] is None:
            spk_map["skalle-pär"] = txt
        elif "benrangel" in key and spk_map["benrangel"] is None:
            spk_map["benrangel"] = txt

    if spk_map["skalle-pär"] is None or spk_map["benrangel"] is None:
        return []

    return [
        (voice_skallepar, spk_map["skalle-pär"]),
        (voice_benrangel, spk_map["benrangel"]),
    ]

def generate_roast_from_image(img_b64: str) -> str:
    """
    Single-call vision → roast.
    Uses SYSTEM_PROMPT and the image as context to produce exactly two lines.
    """
    rsp = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": (
                    "Du får nu en bild som kontext. Skriv roast enligt system prompten.\n"
                    "Följ formatet exakt med två meningar och rätt talarnamn."
                 )},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"},
            ],
        }],
        max_output_tokens=120,
        temperature=0.8,
    )
    return rsp.output_text.strip()

def voices_for_image_roast(img_b64: str):
    """
    Convenience: returns [(voice_id, text), (voice_id, text)] ready for TTS.
    """
    raw = generate_roast_from_image(img_b64)
    return assign_alternating_voices(raw)
