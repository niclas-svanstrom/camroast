# camroast/state.py
from dataclasses import dataclass, field
from datetime import datetime

WINDOW_NAME = "RoastCam"

@dataclass
class UIState:
    roast_enabled: bool = False
    request_roast_now: bool = False
    request_premade_now: bool = False
    last_text: str = ""
    last_text_ts: datetime = field(default_factory=lambda: datetime.min)
    _roast_rect: tuple | None = None
    _now_rect: tuple | None = None
    _mic_rect: tuple | None = None
    _premade_rect: tuple | None = None
    mic_mode_enabled: bool = False
    _mic_detector: object | None = None
    _mic_last_trigger: datetime = field(default_factory=lambda: datetime.min)
    _gate_text: str = ""
    premade_pairs: list[tuple[str,str]] = field(default_factory=list)
    premade_idx: int = 0
