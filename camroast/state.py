# camroast/state.py
from dataclasses import dataclass

WINDOW_NAME = "RoastCam"


@dataclass
class UIState:
    # toggles / one-shot requests (set by mouse or keyboard)
    roast_enabled: bool = False
    request_roast_now: bool = False
    request_premade_now: bool = False
    mic_mode_enabled: bool = False
    # button hit boxes, filled in by the overlay
    _roast_rect: tuple | None = None
    _now_rect: tuple | None = None
    _mic_rect: tuple | None = None
    _premade_rect: tuple | None = None
    # live status
    show_state: str = "idle"          # idle | generating | transcribing | speaking | premade
    armed: bool = True
    confirm_progress: float = 0.0     # 0..1 while a person is being confirmed
    person_count: int = 0
    fps: float = 0.0
    yolo_ms: float = 0.0
    mic_status: str = ""
    mic_active: bool = False
    mic_muted: bool = False
    # subtitles
    subtitle_speaker: str = ""
    subtitle_text: str = ""
    subtitle_until: float = 0.0       # time.time() deadline, inf while speaking
    child_text: str = ""
    child_text_until: float = 0.0
    last_error: str = ""
    error_until: float = 0.0
    # Tapo indicators
    tapo_ok: bool = False
    tapo_human_recent: bool = False
    tapo_motion_recent: bool = False
