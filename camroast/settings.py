# camroast/settings.py
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

BOOL_TRUE = {"1","true","True"}

@dataclass(frozen=True)
class Settings:
    show_live: bool = True
    fps: int = 10
    frame_skip: int = 5
    gpt_cooldown_sec: int = 10
    mic_cooldown_sec: int = 10
    dark_luma_thresh: int = 40
    premade_dir: str = os.getenv("PREMADE_DIR", "premade")
    attention_dir: str = os.getenv("ATTENTION_DIR", "attention")
    prem_mic_require_dark: bool = os.getenv("PREM_MIC_REQUIRE_DARK", "0") in BOOL_TRUE
    prem_mic_require_no_person: bool = os.getenv("PREM_MIC_REQUIRE_NO_PERSON", "0") in BOOL_TRUE
    prem_mic_debug: bool = os.getenv("PREM_MIC_DEBUG", "0") in BOOL_TRUE
    voice_skallepar: str = os.getenv("VOICE_SKALLEPAR", "NHVO1d5lgqVtAvyYNL2P")
    voice_benrangel: str = os.getenv("VOICE_BENRANGEL", "S6pZEFGfrgnWx4AETPdD")
