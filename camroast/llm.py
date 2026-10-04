# camroast/llm.py
"""Joke generation with the OpenAI Responses API. Returns one line per skeleton as structured JSON."""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .util import short_err

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SKALLE = "Skalle-Pär"
BEN = "Benrangel"

SYSTEM_PROMPT = (
    "Ni är två skelett, Skalle-Pär och Benrangel, som hänger utanför ett hus på Halloween "
    "och pratar med barnen som kommer för bus eller godis.\n"
    "Du får en bild av vem som står framför er, och ibland något som någon just sa. "
    "Skriv en kort dialog på svenska: Skalle-Pär säger en replik, sedan Benrangel. Max 15 ord per replik.\n"
    "Ton: busig, spöklik och snäll. Barnen ska skratta, inte bli ledsna. Skoja om utklädnader, färger, "
    "poser, godispåsar, vädret eller era egna knarrande ben. Gärna ordvitsar och små överdrifter.\n"
    "Om någon sa något: svara på det, gärna med en motfråga eller en liten utmaning så att samtalet fortsätter.\n"
    "Aldrig: elakt om utseende eller kropp, inget om bilar, hus eller grannar, och säg aldrig att ni ser via "
    "kamera eller bild eller att ni är en AI. Fråga aldrig efter namn eller var barnen bor.\n"
    "Ser du inga barn tydligt: skoja om väntan, om varandra eller om spöken.\n"
    "Upprepa inte skämt från listan över tidigare repliker."
)

ROAST_FORMAT = {
    "type": "json_schema",
    "name": "skelett_dialog",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["skallepar", "benrangel"],
        "properties": {
            "skallepar": {"type": "string", "description": "Skalle-Pärs replik"},
            "benrangel": {"type": "string", "description": "Benrangels replik"},
        },
    },
}


def _user_text(transcript, dialogue, recent_jokes, has_image: bool = True) -> str:
    parts = []
    if dialogue:
        parts.append("Samtalet hittills:")
        for child, sk, be in dialogue:
            if child:
                parts.append(f"Barn: {child}")
            parts.append(f"{SKALLE}: {sk}")
            parts.append(f"{BEN}: {be}")
        parts.append("")
    if transcript:
        parts.append(f'Någon sa just: "{transcript}"')
        parts.append("Svara på det" + (", och koppla gärna till vad ni ser i bilden." if has_image else "."))
    else:
        parts.append("Här kommer några nya. Kommentera det ni ser i bilden." if has_image
                     else "Ni ser ingen just nu. Skoja om väntan, mörkret eller varandra.")
    if recent_jokes:
        parts.append("")
        parts.append("Tidigare repliker (upprepa inte):")
        for sk, be in recent_jokes:
            parts.append(f"- {sk} / {be}")
    return "\n".join(parts)


def generate_lines(
    img_b64: str | None,
    *,
    model: str,
    effort: str | None = "none",
    detail: str = "auto",
    transcript: str | None = None,
    dialogue=(),
    recent_jokes=(),
    timeout: float = 20.0,
    service_tier: str | None = None,
) -> tuple[str, str]:
    """One call: image (+ optional transcript) -> (Skalle-Pär line, Benrangel line). Raises on failure."""
    content = [{"type": "input_text", "text": _user_text(transcript, dialogue, recent_jokes, bool(img_b64))}]
    if img_b64:
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}", "detail": detail})
    kwargs = dict(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": content}],
        text={"format": ROAST_FORMAT},
        max_output_tokens=200,
        timeout=timeout,
    )
    if effort and not model.startswith("gpt-4"):
        kwargs["reasoning"] = {"effort": effort}
    if service_tier:
        kwargs["service_tier"] = service_tier
    rsp = client.responses.create(**kwargs)
    raw = (rsp.output_text or "").strip()
    if not raw:
        raise RuntimeError("tomt svar från modellen")
    data = json.loads(raw)
    sk = str(data.get("skallepar", "")).strip()
    be = str(data.get("benrangel", "")).strip()
    if not sk or not be:
        raise RuntimeError(f"ofullständigt svar: {raw[:120]}")
    return sk, be


def warmup(model: str) -> str | None:
    """Validate the model id and open the HTTPS connection. Returns an error string or None."""
    try:
        client.models.retrieve(model)
        return None
    except Exception as e:
        return short_err(e)
