import cv2


def _draw_button(img, rect, text, active=False):
    x1, y1, x2, y2 = rect
    bg = (40, 40, 40)
    bg_active = (60, 120, 60)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_active if active else bg, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)


def _wrap_text(text, max_width_px, scale=0.6, thickness=1):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw > max_width_px and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines


def draw_ui_overlay(frame, state):
    h, w = frame.shape[:2]
    pad = 8
    btn_w, btn_h = 140, 32
    roast_rect = (pad, pad, pad + btn_w, pad + btn_h)
    now_rect = (pad * 2 + btn_w, pad, pad * 2 + btn_w * 2, pad + btn_h)
    mic_rect = (pad * 3 + btn_w * 2, pad, pad * 3 + btn_w * 3, pad + btn_h)
    premade_rect = (pad * 4 + btn_w * 3, pad, pad * 4 + btn_w * 4, pad + btn_h)

    state._roast_rect = roast_rect
    state._now_rect = now_rect
    state._mic_rect = mic_rect
    state._premade_rect = premade_rect

    _draw_button(frame, roast_rect, f"Roast: {'ON' if state.roast_enabled else 'OFF'}", active=state.roast_enabled)
    _draw_button(frame, now_rect, "Roast Now", active=False)
    _draw_button(frame, mic_rect, f"Mic Mode: {'ON' if state.mic_mode_enabled else 'OFF'}", active=state.mic_mode_enabled)
    _draw_button(frame, premade_rect, "Premade Now", active=False)

    if state.mic_mode_enabled and state._mic_detector is not None:
        backend = getattr(state._mic_detector, 'backend', 'Mic')
        devname = getattr(state._mic_detector, 'device_name', '')
        hint = f"{backend}{' - ' + devname if devname else ''}"
        (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = mic_rect[2] + 10
        ty = mic_rect[1] + (mic_rect[3] - mic_rect[1]) // 2 + th // 2
        cv2.putText(frame, hint, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 0), 1)
        active = False
        try:
            active = state._mic_detector.speech_recent(0.6)
        except Exception:
            active = False
        cx = mic_rect[0] - 10
        cy = mic_rect[1] + (mic_rect[3] - mic_rect[1]) // 2
        cv2.circle(frame, (cx, cy), 6, (0, 220, 0) if active else (80, 80, 80), -1)
        try:
            gate_text = getattr(state, '_gate_text', '')
            if gate_text:
                cv2.putText(frame, gate_text, (tx, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        except Exception:
            pass

    if state.last_text:
        max_w = int(w * 0.9)
        lines = _wrap_text(state.last_text, max_w)
        scale, th = 0.6, 1
        lh = int(18 * scale) + 8
        box_h = lh * len(lines) + 16
        x1 = pad
        y2 = h - pad
        y1 = max(pad, y2 - box_h)
        x2 = x1 + max_w + 2 * pad
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        y = y1 + 16
        for ln in lines:
            cv2.putText(frame, ln, (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th)
            y += lh

