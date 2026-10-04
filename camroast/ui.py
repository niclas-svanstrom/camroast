# camroast/ui.py
"""On-screen overlay: buttons, status line and subtitles.

Anything with Swedish characters is drawn with Pillow, because OpenCV's built-in
fonts cannot render å, ä and ö. Rendered text is cached as RGBA images and only
alpha-blended onto the frame, so the overlay costs a few milliseconds per frame.
"""
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .llm import BEN, SKALLE

SKALLE_RGB = (255, 200, 70)
BEN_RGB = (120, 205, 255)
CHILD_RGB = (235, 235, 235)
ERROR_RGB = (255, 90, 90)
STATUS_RGB = (220, 220, 220)

STATE_LABELS = {
    "idle": "",
    "generating": "Tänker...",
    "transcribing": "Hörde något, lyssnar...",
    "speaking": "Pratar",
    "premade": "Förinspelat",
}

_FONT_NAMES = ["segoeuib.ttf", "arialbd.ttf", "segoeui.ttf", "arial.ttf"]
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_fonts: dict = {}
_cache: OrderedDict = OrderedDict()
_CACHE_MAX = 64


def _font(size: int, path: str | None = None):
    key = (size, path)
    if key in _fonts:
        return _fonts[key]
    win_fonts = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
    candidates = ([path] if path else []) + [os.path.join(win_fonts, n) for n in _FONT_NAMES] + _FONT_PATHS
    font = None
    for c in candidates:
        try:
            if os.path.exists(c):
                font = ImageFont.truetype(c, size)
                break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except Exception:
            font = ImageFont.load_default()
    _fonts[key] = font
    return font


def _wrap(text: str, font, max_w: int):
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if not cur or font.getlength(cand) <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


class _Sprite:
    """Pre-multiplied RGBA text image ready for fast blending onto BGR frames."""

    def __init__(self, rgba: np.ndarray):
        self.h, self.w = rgba.shape[:2]
        # crop to the painted area so blending touches as few pixels as possible
        ys, xs = np.nonzero(rgba[:, :, 3])
        if len(ys) == 0:
            self.ox = self.oy = 0
            self.bgr = np.zeros((0, 0, 3), np.uint8)
            self.w_src = self.w_dst = np.zeros((0, 0), np.float32)
            return
        self.oy, self.ox = int(ys.min()), int(xs.min())
        crop = rgba[self.oy:int(ys.max()) + 1, self.ox:int(xs.max()) + 1]
        self.bgr = np.ascontiguousarray(crop[:, :, 2::-1])            # RGB -> BGR
        self.w_src = np.ascontiguousarray(crop[:, :, 3].astype(np.float32) / 255.0)
        self.w_dst = np.ascontiguousarray(1.0 - self.w_src)

    def blend(self, frame, x: int, y: int):
        """Alpha-blend at (x, y) = the sprite's top-left corner, clipped to the frame."""
        h, w = frame.shape[:2]
        x += self.ox
        y += self.oy
        ph, pw = self.w_src.shape[:2]
        sx = sy = 0
        if x < 0:
            sx, x = -x, 0
        if y < 0:
            sy, y = -y, 0
        x2, y2 = min(w, x + pw - sx), min(h, y + ph - sy)
        if x >= x2 or y >= y2:
            return
        rows, cols = slice(sy, sy + y2 - y), slice(sx, sx + x2 - x)
        dst = frame[y:y2, x:x2]
        dst[:] = cv2.blendLinear(dst, np.ascontiguousarray(self.bgr[rows, cols]),
                                 np.ascontiguousarray(self.w_dst[rows, cols]),
                                 np.ascontiguousarray(self.w_src[rows, cols]))


def _cached(key, make):
    hit = _cache.get(key)
    if hit is None:
        hit = make()
        _cache[key] = hit
        if len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return hit


def _render_line(text: str, size: int, rgb, font_path) -> _Sprite:
    f = _font(size, font_path)
    w = max(1, int(f.getlength(text)) + 8)
    h = max(1, int(size * 1.4))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((2, 0), text, font=f, fill=tuple(rgb) + (255,), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return _Sprite(np.asarray(img))


def _render_block(items, width: int, font_path, pad: int) -> _Sprite:
    rows = []
    for text, rgb, size in items:
        f = _font(size, font_path)
        for ln in _wrap(text, f, width - 4 * pad):
            rows.append((ln, rgb, f, int(size * 1.25)))
    total = sum(r[3] for r in rows) + 2 * pad
    img = Image.new("RGBA", (max(1, width), max(1, total)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = pad
    for ln, rgb, f, lh in rows:
        draw.text((2 * pad, y), ln, font=f, fill=tuple(rgb) + (255,), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        y += lh
    return _Sprite(np.asarray(img))


def draw_text(frame, text: str, xy, size: int, rgb, font_path=None):
    """One line of Unicode text with a dark outline, top-left at xy."""
    sprite = _cached(("line", text, size, tuple(rgb), font_path), lambda: _render_line(text, size, rgb, font_path))
    sprite.blend(frame, int(xy[0]), int(xy[1]))


def draw_text_block(frame, items, bottom: int, font_path=None, pad: int = 10, alpha: float = 0.6):
    """Wrapped text rows in a translucent box whose lower edge is `bottom`.

    items: list of (text, rgb, size). Returns the top y of the box.
    """
    width = frame.shape[1]
    items = tuple((t, tuple(c), s) for t, c, s in items)
    sprite = _cached(("block", items, width, font_path, pad), lambda: _render_block(items, width, font_path, pad))
    top = max(0, bottom - sprite.h)
    region = frame[top:bottom, :]
    cv2.addWeighted(region, 1.0 - alpha, region, 0.0, 0.0, dst=region)  # darken in place
    sprite.blend(frame, 0, bottom - sprite.h)
    return top


def draw_no_camera(frame, lines, s=None):
    """Centered message lines for the no-camera screen. The first line is the headline."""
    h, w = frame.shape[:2]
    font_path = getattr(s, "subtitle_font", None) if s is not None else None
    big, small = max(26, w // 30), max(16, w // 55)
    sizes = [big] + [small] * (len(lines) - 1)
    colors = [(255, 200, 70)] + [(220, 220, 220)] * (len(lines) - 1)
    total = sum(int(sz * 1.6) for sz in sizes)
    y = max(60, (h - total) // 2)
    for text, sz, rgb in zip(lines, sizes, colors):
        tw = int(_font(sz, font_path).getlength(text))
        draw_text(frame, text, ((w - tw) // 2, y), sz, rgb, font_path)
        y += int(sz * 1.6)


def _draw_button(img, rect, text, active=False):
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 120, 60) if active else (40, 40, 40), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, text, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)


def draw_ui_overlay(frame, state, s=None):
    h, w = frame.shape[:2]
    now = time.time()
    font_path = getattr(s, "subtitle_font", None) if s is not None else None
    pad, btn_w, btn_h = 8, 140, 32
    rects = [(pad + i * (btn_w + pad), pad, pad + i * (btn_w + pad) + btn_w, pad + btn_h) for i in range(4)]
    state._roast_rect, state._now_rect, state._mic_rect, state._premade_rect = rects
    _draw_button(frame, rects[0], f"Roast: {'ON' if state.roast_enabled else 'OFF'}", active=state.roast_enabled)
    _draw_button(frame, rects[1], "Roast Now")
    _draw_button(frame, rects[2], f"Mic: {'ON' if state.mic_mode_enabled else 'OFF'}", active=state.mic_mode_enabled)
    _draw_button(frame, rects[3], "Premade Now")

    small = max(16, w // 50)
    big = max(20, w // 36)

    # status line under the buttons
    parts = []
    label = STATE_LABELS.get(state.show_state, state.show_state)
    if label:
        parts.append(label)
    if state.roast_enabled and state.show_state == "idle":
        parts.append("Redo" if state.armed else "Väntar på nya barn")
    if state.person_count:
        parts.append(f"{state.person_count} pers")
    if state.fps > 0:
        parts.append(f"{state.fps:.0f} fps" + (f", yolo {state.yolo_ms:.0f} ms" if state.yolo_ms > 0 else ""))
    y = pad + btn_h + 8
    if parts:
        draw_text(frame, "  ·  ".join(parts), (pad, y), small, STATUS_RGB, font_path)
    if 0.0 < state.confirm_progress < 1.0 and state.show_state == "idle":
        bx, by, bw, bh = pad, y + int(small * 1.5), 160, 8
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * state.confirm_progress), by + bh), (60, 200, 60), -1)
    if state.last_error and now < state.error_until:
        draw_text(frame, state.last_error, (pad, y + int(small * 2.2)), small - 2, ERROR_RGB, font_path)

    # mic indicator left of the mic button, device hint to the right of the buttons
    if state.mic_mode_enabled:
        mx1, my1, mx2, my2 = rects[2]
        cy = (my1 + my2) // 2
        color = (80, 80, 80) if state.mic_muted else ((0, 220, 0) if state.mic_active else (0, 120, 0))
        cv2.circle(frame, (mx1 - 10, cy), 6, color, -1)
        if state.mic_status:
            draw_text(frame, state.mic_status, (rects[3][2] + 12, my1 + 4), small - 2, (220, 220, 0), font_path)

    # Tapo indicators
    if state.tapo_ok or state.tapo_human_recent or state.tapo_motion_recent:
        tx = rects[3][2] + 12
        ty = rects[3][3] + 6
        cv2.putText(frame, "Tapo", (tx, ty + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cx, cy = tx + 60, ty + 7
        cv2.circle(frame, (cx, cy), 6, (0, 220, 0) if state.tapo_human_recent else (80, 80, 80), -1)
        cv2.circle(frame, (cx + 20, cy), 6, (0, 220, 220) if state.tapo_motion_recent else (80, 80, 80), -1)

    # subtitles at the bottom
    items = []
    if state.child_text and now < state.child_text_until:
        items.append((f"Barn: {state.child_text}", CHILD_RGB, small))
    if state.subtitle_text and now < state.subtitle_until:
        rgb = SKALLE_RGB if state.subtitle_speaker == SKALLE else BEN_RGB if state.subtitle_speaker == BEN else STATUS_RGB
        txt = f"{state.subtitle_speaker}: {state.subtitle_text}" if state.subtitle_speaker else state.subtitle_text
        items.append((txt, rgb, big))
    if items:
        draw_text_block(frame, items, bottom=h - 12, font_path=font_path)
