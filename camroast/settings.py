# camroast/settings.py
"""All tunables, read from environment / .env once at import time. See .env.example."""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

BOOL_TRUE = {"1", "true", "yes", "on"}


def _env_str(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    return v.strip()


def _env_float(name: str, default: float | None = None) -> float | None:
    v = _env_str(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = _env_str(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = _env_str(name)
    if v is None:
        return default
    return v.lower() in BOOL_TRUE


def _env_cam_source() -> int | str:
    v = _env_str("CAM_SOURCE")
    if v is None:
        return 0
    if v.isdigit():
        return int(v)
    # rtsp/http URL or a video file path
    return v


@dataclass(frozen=True)
class Settings:
    # --- Window / UI
    show_live: bool = _env_bool("SHOW_LIVE", True)
    roast_autostart: bool = _env_bool("ROAST_AUTOSTART", False)
    subtitle_hold_sec: float = _env_float("SUBTITLE_HOLD_SEC", 2.5)
    subtitle_font: str | None = _env_str("SUBTITLE_FONT")

    # --- Camera
    camera_source: int | str = _env_cam_source()
    try_camera_low_light: bool = _env_bool("TRY_CAMERA_LOW_LIGHT", True)
    camera_exposure: float | None = _env_float("CAMERA_EXPOSURE")
    camera_gain: float | None = _env_float("CAMERA_GAIN")
    camera_brightness: float | None = _env_float("CAMERA_BRIGHTNESS")
    dark_luma_thresh: int = _env_int("DARK_LUMA_THRESH", 40)
    camera_timeout_sec: float = _env_float("CAMERA_TIMEOUT_SEC", 5.0)      # give up one connection attempt after this
    camera_retry_sec: float = _env_float("CAMERA_RETRY_SEC", 5.0)          # wait between attempts while unreachable
    camera_stall_sec: float = _env_float("CAMERA_STALL_SEC", 5.0)          # no frames this long -> reconnect

    # --- Detection / presence
    min_motion_pixels: int = _env_int("MIN_MOTION_PIXELS", 1500)
    yolo_conf_day: float = _env_float("YOLO_CONF_DAY", 0.4)
    yolo_conf_night: float = _env_float("YOLO_CONF_NIGHT", 0.6)
    yolo_use_night_conf_when_dark: bool = _env_bool("YOLO_USE_NIGHT_CONF_WHEN_DARK", True)
    yolo_imgsz: int = _env_int("YOLO_IMGSZ", 640)                          # smaller = faster, 416 is fine for kids at a door
    yolo_max_hz: float = _env_float("YOLO_MAX_HZ", 10.0)                    # detection rate cap; the video still draws at full rate
    torch_threads: int = _env_int("TORCH_THREADS", 0)                       # 0 = auto (a third of the logical CPUs, max 4)
    person_confirm_sec: float = _env_float("PERSON_CONFIRM_SEC", 0.5)     # how long a person must be seen before a roast
    person_gap_sec: float = _env_float("PERSON_GAP_SEC", 0.4)             # missed detections shorter than this do not reset
    person_track_sec: float = _env_float("PERSON_TRACK_SEC", 2.0)         # keep running YOLO this long after last person, even without motion
    yolo_idle_interval_sec: float = _env_float("YOLO_IDLE_INTERVAL_SEC", 1.0)  # run YOLO at least this often even without motion
    best_frame_window_sec: float = _env_float("BEST_FRAME_WINDOW_SEC", 1.0)
    rearm_clear_sec: float = _env_float("REARM_CLEAR_SEC", 3.0)           # scene empty this long -> ready for the next group
    rearm_same_scene_sec: float = _env_float("REARM_SAME_SCENE_SEC", 30.0)  # same group still there -> next joke after this long

    # --- Jokes (OpenAI)
    llm_model: str = _env_str("LLM_MODEL", "gpt-6-luna")
    llm_reasoning_effort: str = _env_str("LLM_REASONING_EFFORT", "none")
    llm_image_max_side: int = _env_int("LLM_IMAGE_MAX_SIDE", 768)
    llm_image_detail: str = _env_str("LLM_IMAGE_DETAIL", "low")
    llm_service_tier: str | None = _env_str("LLM_SERVICE_TIER")            # "priority" = lowest latency, higher price
    llm_timeout_sec: float = _env_float("LLM_TIMEOUT_SEC", 20.0)
    joke_history_size: int = _env_int("JOKE_HISTORY_SIZE", 6)

    # --- Voices (ElevenLabs)
    tts_model: str = _env_str("ELEVEN_MODEL", "eleven_flash_v2_5")
    tts_output_format: str = _env_str("ELEVEN_OUTPUT_FORMAT", "pcm_24000")
    audio_output_device: str | None = _env_str("AUDIO_OUTPUT_DEVICE")
    voice_skallepar: str = _env_str("VOICE_SKALLEPAR", "NHVO1d5lgqVtAvyYNL2P")
    voice_benrangel: str = _env_str("VOICE_BENRANGEL", "S6pZEFGfrgnWx4AETPdD")

    # --- Mic / talk-back
    mic_device: str | None = _env_str("MIC_DEVICE") or _env_str("PREM_MIC_DEVICE")
    mic_autostart: bool = _env_bool("MIC_AUTOSTART", False)
    mic_action: str = _env_str("MIC_ACTION", "talkback")                 # talkback | premade
    mic_end_silence_ms: int = _env_int("MIC_END_SILENCE_MS", 600)
    mic_min_speech_ms: int = _env_int("MIC_MIN_SPEECH_MS", 250)
    mic_max_utterance_sec: float = _env_float("MIC_MAX_UTTERANCE_SEC", 8.0)
    mic_mute_tail_ms: int = _env_int("MIC_MUTE_TAIL_MS", 500)
    mic_cooldown_sec: float = _env_float("MIC_COOLDOWN_SEC", 10.0)       # premade mode only
    mic_debug: bool = _env_bool("MIC_DEBUG", False) or _env_bool("PREM_MIC_DEBUG", False)
    prem_mic_require_dark: bool = _env_bool("PREM_MIC_REQUIRE_DARK", False)
    prem_mic_require_no_person: bool = _env_bool("PREM_MIC_REQUIRE_NO_PERSON", False)
    stt_model: str = _env_str("STT_MODEL", "gpt-4o-mini-transcribe")
    stt_language: str = _env_str("STT_LANGUAGE", "sv")
    stt_min_chars: int = _env_int("STT_MIN_CHARS", 3)
    dialogue_history_size: int = _env_int("DIALOGUE_HISTORY_SIZE", 4)
    dialogue_reset_sec: float = _env_float("DIALOGUE_RESET_SEC", 90.0)

    # --- Media
    premade_dir: str = _env_str("PREMADE_DIR", "media/premade")
    attention_dir: str = _env_str("ATTENTION_DIR", "media/attention")
    filler_dir: str = _env_str("FILLER_DIR", "media/filler")

    # --- Tapo/ONVIF event integration
    tapo_enable_events: bool = _env_bool("TAPO_ENABLE_EVENTS", False)
    tapo_host: str | None = _env_str("TAPO_HOST")
    tapo_user: str | None = _env_str("TAPO_USER")
    tapo_password: str | None = _env_str("TAPO_PASSWORD")
    tapo_onvif_port: int = _env_int("TAPO_ONVIF_PORT", 2020)
    tapo_poll_seconds: float = _env_float("TAPO_POLL_SECONDS", 1.0)
    tapo_use_only_when_dark: bool = _env_bool("TAPO_USE_ONLY_WHEN_DARK", True)
