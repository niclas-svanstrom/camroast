# camroast/settings.py
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

BOOL_TRUE = {"1","true","True"}


def _env_float(name: str):
    v = os.getenv(name)
    if v is None:
        return None
    try:
        v = v.strip()
        if not v:
            return None
        return float(v)
    except Exception:
        return None

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


def _env_cam_source() -> int | str:
    v = os.getenv("CAM_SOURCE")
    if v is None or v.strip() == "":
        return 0
    v = v.strip()
    # If user passed a plain integer like "0" or "1", use it as device index
    if v.isdigit():
        try:
            return int(v)
        except Exception:
            return 0
    # Otherwise treat as a URI (rtsp/http/file path)
    return v

@dataclass(frozen=True)
class Settings:
    show_live: bool = True
    fps: int = 10
    frame_skip: int = 5
    gpt_cooldown_sec: int = 10
    mic_cooldown_sec: int = 10
    dark_luma_thresh: int = _env_int("DARK_LUMA_THRESH", 40)
    # Motion/detection gating
    min_motion_pixels: int = _env_int("MIN_MOTION_PIXELS", 1500)
    person_confirm_window: int = _env_int("PERSON_CONFIRM_WINDOW", 5)
    person_confirm_min: int = _env_int("PERSON_CONFIRM_MIN", 3)
    premade_dir: str = os.getenv("PREMADE_DIR", "premade")
    attention_dir: str = os.getenv("ATTENTION_DIR", "attention")
    # Camera
    camera_source: int | str = _env_cam_source()
    try_camera_low_light: bool = os.getenv("TRY_CAMERA_LOW_LIGHT", "1") in BOOL_TRUE
    camera_exposure: float | None = _env_float("CAMERA_EXPOSURE")
    camera_gain: float | None = _env_float("CAMERA_GAIN")
    camera_brightness: float | None = _env_float("CAMERA_BRIGHTNESS")
    # YOLO thresholds
    yolo_conf_day: float | None = _env_float("YOLO_CONF_DAY") or 0.4
    yolo_conf_night: float | None = _env_float("YOLO_CONF_NIGHT") or 0.6
    yolo_use_night_conf_when_dark: bool = os.getenv("YOLO_USE_NIGHT_CONF_WHEN_DARK", "1") in BOOL_TRUE
    # Tapo/ONVIF event integration
    tapo_enable_events: bool = os.getenv("TAPO_ENABLE_EVENTS", "0") in BOOL_TRUE
    tapo_host: str | None = os.getenv("TAPO_HOST")
    tapo_user: str | None = os.getenv("TAPO_USER")
    tapo_password: str | None = os.getenv("TAPO_PASSWORD")
    tapo_onvif_port: int = _env_int("TAPO_ONVIF_PORT", 2020)
    tapo_poll_seconds: float = float(os.getenv("TAPO_POLL_SECONDS", "1.0"))
    tapo_use_only_when_dark: bool = os.getenv("TAPO_USE_ONLY_WHEN_DARK", "1") in BOOL_TRUE
    prem_mic_require_dark: bool = os.getenv("PREM_MIC_REQUIRE_DARK", "0") in BOOL_TRUE
    prem_mic_require_no_person: bool = os.getenv("PREM_MIC_REQUIRE_NO_PERSON", "0") in BOOL_TRUE
    prem_mic_debug: bool = os.getenv("PREM_MIC_DEBUG", "0") in BOOL_TRUE
    voice_skallepar: str = os.getenv("VOICE_SKALLEPAR", "NHVO1d5lgqVtAvyYNL2P")
    voice_benrangel: str = os.getenv("VOICE_BENRANGEL", "S6pZEFGfrgnWx4AETPdD")
