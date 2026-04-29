import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import time
import threading
import math

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# ══════════════════════════════════════════════════════════════════════════════
#  COLOR THEME  ── change anything here to restyle the entire HUD
# ══════════════════════════════════════════════════════════════════════════════
THEME = {
    # ── backgrounds ──────────────────────────────────────────────────────────
    'bg_dark':       (15,  15,  20),    # main panel bg
    'bg_panel':      (25,  28,  38),    # card / overlay bg
    'bg_accent':     (30,  38,  55),    # highlighted section bg
    'bg_warn':       (60,  20,  20),    # warning bar bg
    'bg_success':    (10,  50,  30),    # success bar bg

    # ── text ─────────────────────────────────────────────────────────────────
    'text_primary':  (230, 230, 240),   # main labels
    'text_secondary':(150, 155, 170),   # sub-labels / hints
    'text_title':    (255, 255, 255),   # big headings
    'text_dim':      (90,  95, 110),    # very faint / disabled

    # ── accent colours ───────────────────────────────────────────────────────
    'accent_green':  (0,   220,  90),   # OK / active / go
    'accent_cyan':   (0,   200, 255),   # info / move cursor
    'accent_yellow': (255, 200,   0),   # caution / drag
    'accent_orange': (255, 140,  30),   # steer left
    'accent_pink':   (255,  70, 180),   # steer right / right-click
    'accent_blue':   (60,  140, 255),   # scroll down / brake
    'accent_red':    (220,  40,  60),   # stop / danger
    'accent_purple': (160,  80, 255),   # calibrate / special

    # ── gesture gesture_colours ──────────────────────────────────────────────
    'gest_move':         (255, 200,   0),
    'gest_left_click':   (  0, 200, 255),
    'gest_right_click':  (255,  70, 180),
    'gest_drag':         (  0, 120, 255),
    'gest_scroll_up':    (180, 255,  80),
    'gest_scroll_down':  ( 80, 180, 255),
    'gest_idle':         (120, 120, 130),

    # ── hand skeleton ────────────────────────────────────────────────────────
    'hand_bone':     (0,   200, 110),
    'hand_joint':    (255, 255, 255),
    'hand_joint_bd': (0,   150,  80),

    # ── progress / bars ──────────────────────────────────────────────────────
    'bar_track':     (55,  58,  70),
    'bar_zone':      (0,   90,  40),
    'bar_fill_ok':   (0,   210,  80),
    'bar_fill_warn': (0,   80,  255),
    'bar_fill_far':  (0,   190, 255),

    # ── borders / dividers ───────────────────────────────────────────────────
    'border':        (55,  60,  80),
    'divider':       (40,  42,  58),
}

# ══════════════════════════════════════════════════════════════════════════════
#  TUNING
# ══════════════════════════════════════════════════════════════════════════════
SMOOTHING        = 0.3
PINCH_THRESH     = 0.03
CLICK_COOLDOWN   = 0.5
SCROLL_SPEED     = 3
LONG_PRESS_TIME  = 0.5
SCROLL_BUFFER    = 0

TARGET_DIST      = 0.18
DIST_TOL         = 0.03
DIST_OK_HOLD     = 5.0   # hold at correct distance for 5s before entering running
CALIB_DURATION   = 5.0

INTRO_DURATION   = 30.0
ZONE_DURATION    = 30.0
GUIDE_DURATION   = 30.0
SKIP_LOCKOUT     = 3.0   # thumb-up does nothing in first 3s of any screen

HOLD_META        = 3.0   # seconds to hold pause / start / recal
HOLD_CLOSE       = 3.0   # seconds to hold close (both fists)
HOLD_GAME        = 5.0   # seconds to hold a game option

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Zone presets  (fraction of screen mapped to)
ZONE_PRESETS = {
    'small':  0.25,   # centre 25% of frame
    'medium': 0.55,
    'large':  0.90,
}

# ══════════════════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════════════════
app_state      = 'intro'
intro_start_t  = None
zone_intro_start_t = None   # new info-only screen before zone pick
zone_start_t   = None
guide_start_t  = None
chosen_zone    = 'medium'

dist_ok_since  = None

range_min_x = range_max_x = None
range_min_y = range_max_y = None

last_click_t   = 0.0
prev_scroll_y  = None
scroll_entry_t = None
scroll_active  = False
tm_pinch_start = None
tm_drag_active = False

smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2

# meta-gesture hold timers
meta_hold = {
    'start':    None,
    'stop':     None,
    'close':    None,
    'recal':    None,
    'game_opt': None,
}
game_option_pending = None   # 1-4
game_opt_number     = None   # which option is being held

# ── threading ─────────────────────────────────────────────────────────────────
_latest_frame  = None
_latest_result = None
_frame_lock    = threading.Lock()
_result_lock   = threading.Lock()
_stop_thread   = False

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def hand_size(lms):
    return np.hypot(lms[0].x - lms[9].x, lms[0].y - lms[9].y)

def tip_dist(lms, a, b):
    return np.hypot(lms[a].x - lms[b].x, lms[a].y - lms[b].y)

def is_finger_extended(lms, tip, pip):
    tip_d = np.hypot(lms[tip].x - lms[0].x, lms[tip].y - lms[0].y)
    pip_d = np.hypot(lms[pip].x - lms[0].x, lms[pip].y - lms[0].y)
    return tip_d > pip_d

def fingers_extended(lms):
    return {
        'index':  is_finger_extended(lms, 8,  6),
        'middle': is_finger_extended(lms, 12, 10),
        'ring':   is_finger_extended(lms, 16, 14),
        'pinky':  is_finger_extended(lms, 20, 18),
    }

def count_fingers_up(lms):
    f = fingers_extended(lms)
    return sum(f.values())

def is_thumbs_up(lms):
    """Thumb tip clearly above wrist and all other fingers curled."""
    thumb_up = lms[4].y < lms[3].y < lms[2].y   # thumb tip above knuckles
    f = fingers_extended(lms)
    others_down = not (f['index'] or f['middle'] or f['ring'] or f['pinky'])
    return thumb_up and others_down

def is_open_palm(lms):
    f = fingers_extended(lms)
    return all(f.values())

def is_fist(lms):
    f = fingers_extended(lms)
    return not any(f.values())

def is_shaka(lms):
    """Pinky + thumb extended, index/middle/ring curled."""
    f = fingers_extended(lms)
    thumb_ext = lms[4].y < lms[3].y
    return thumb_ext and f['pinky'] and not f['index'] and not f['middle'] and not f['ring']

def is_metal_sign(lms):
    """Index + pinky up, middle + ring curled, thumb optional."""
    f = fingers_extended(lms)
    return f['index'] and f['pinky'] and not f['middle'] and not f['ring']

def is_peace_sign(lms):
    """Index + middle up, ring + pinky curled — peace/victory sign."""
    f = fingers_extended(lms)
    return f['index'] and f['middle'] and not f['ring'] and not f['pinky']

def get_game_option(lms, lms2):
    """
    Left hand = fist, right hand = 1-4 fingers up.
    Returns option number 1-4 or None.
    """
    if lms is None or lms2 is None:
        return None
    # Try both hand assignments since MediaPipe doesn't label L/R reliably
    for fist_hand, finger_hand in [(lms, lms2), (lms2, lms)]:
        if is_fist(fist_hand):
            n = count_fingers_up(finger_hand)
            if 1 <= n <= 4:
                return n
    return None

def get_gesture(lms):
    f = fingers_extended(lms)

    if f['index'] and f['middle'] and f['ring'] and not f['pinky']:
        return 'scroll_up'
    if not f['index'] and not f['middle'] and not f['ring'] and not f['pinky']:
        return 'scroll_down'
    if tip_dist(lms, 4, 12) < PINCH_THRESH:
        return 'tm_pinch'
    if tip_dist(lms, 4, 16) < PINCH_THRESH:
        return 'right_click'

    index_tip_dist = np.hypot(lms[8].x - lms[0].x, lms[8].y - lms[0].y)
    def other_curled(tip_idx):
        d = np.hypot(lms[tip_idx].x - lms[0].x, lms[tip_idx].y - lms[0].y)
        return d < index_tip_dist * 0.85

    if f['index'] and other_curled(12) and other_curled(16) and other_curled(20):
        return 'move'
    return 'idle'

def set_zone_from_choice(zone_name):
    """Auto-compute range_min/max from zone preset (centred on frame)."""
    global range_min_x, range_max_x, range_min_y, range_max_y
    half = ZONE_PRESETS.get(zone_name, 0.55) / 2
    range_min_x = 0.5 - half
    range_max_x = 0.5 + half
    range_min_y = 0.5 - half
    range_max_y = 0.5 + half

def map_cursor(tip_x, tip_y):
    span_x = range_max_x - range_min_x
    span_y = range_max_y - range_min_y
    rx = 0.5 if span_x < 0.01 else np.clip((tip_x - range_min_x) / span_x, 0, 1)
    ry = 0.5 if span_y < 0.01 else np.clip((tip_y - range_min_y) / span_y, 0, 1)
    return rx * SCREEN_W, ry * SCREEN_H

def handle_pinch_release(now):
    global tm_pinch_start, tm_drag_active, last_click_t
    if tm_pinch_start is None:
        return
    held = now - tm_pinch_start
    if tm_drag_active:
        pyautogui.mouseUp()
        tm_drag_active = False
    elif held < LONG_PRESS_TIME and (now - last_click_t > CLICK_COOLDOWN):
        pyautogui.click()
        last_click_t = now
    tm_pinch_start = None

# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def draw_hand(img, lms):
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], THEME['hand_bone'], 2)
    for pt in pts:
        cv2.circle(img, pt, 5, THEME['hand_joint'],    -1)
        cv2.circle(img, pt, 5, THEME['hand_joint_bd'],  2)

def filled_rect(img, x1, y1, x2, y2, color, alpha=1.0):
    if alpha >= 1.0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def progress_bar(img, x, y, w, h, frac, fill_color, track_color=None):
    tc = track_color or THEME['bar_track']
    cv2.rectangle(img, (x, y), (x + w, y + h), tc, -1)
    if frac > 0:
        cv2.rectangle(img, (x, y), (x + int(frac * w), y + h), fill_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), THEME['border'], 1)

def text_centered(img, text, cy, font_scale, color, thickness=1):
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(img, text, ((w - tw) // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_top_pill(img, label, color):
    """Thin pill badge at top-left — no leading spaces, uses getTextSize."""
    label = label.strip()
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
    pad_x, pad_y = 8, 5
    x1, y1 = 8, 8
    x2 = x1 + tw + pad_x * 2
    y2 = y1 + th + baseline + pad_y * 2
    cv2.rectangle(img, (x1, y1), (x2, y2), THEME['bg_panel'], -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.putText(img, label, (x1 + pad_x, y1 + pad_y + th),
                font, scale, color, thick, cv2.LINE_AA)

def hold_arc(img, cx, cy, r, frac, color):
    """Draws a circular hold-progress arc."""
    if frac <= 0:
        return
    axes    = (r, r)
    start_a = -90
    end_a   = int(-90 + 360 * min(frac, 1.0))
    cv2.ellipse(img, (cx, cy), axes, 0, start_a, end_a, color, 4)

# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. INTRO SCREEN ───────────────────────────────────────────────────────────

INTRO_SLIDES = [
    {
        'title':    'Welcome to HandMouse',
        'subtitle': 'Control your computer with hand gestures',
        'steps': [
            '1. Distance Check  - stand at the right distance',
            '2. Zone Size Pick  - choose your movement area',
            '3. Running Mode    - move, click, scroll & drag',
        ],
        'hint': 'Thumb Up to skip  -  Auto-advances in 30s',
    },
]

def draw_intro(img, elapsed, skip_frac):
    h, w = img.shape[:2]
    filled_rect(img, 0, 0, w, h, THEME['bg_dark'], 0.92)

    # decorative top bar
    cv2.rectangle(img, (0, 0), (w, 4), THEME['accent_cyan'], -1)

    slide = INTRO_SLIDES[0]

    text_centered(img, slide['title'],    h // 2 - 100, 1.1,
                  THEME['text_title'], 2)
    text_centered(img, slide['subtitle'], h // 2 - 60,  0.6,
                  THEME['text_secondary'], 1)

    cv2.line(img, (w // 2 - 120, h // 2 - 42),
                  (w // 2 + 120, h // 2 - 42), THEME['divider'], 1)

    for i, step in enumerate(slide['steps']):
        text_centered(img, step, h // 2 - 18 + i * 30, 0.55,
                      THEME['text_primary'], 1)

    cv2.line(img, (w // 2 - 120, h // 2 + 78),
                  (w // 2 + 120, h // 2 + 78), THEME['divider'], 1)

    text_centered(img, slide['hint'], h // 2 + 100, 0.45,
                  THEME['text_secondary'], 1)

    # bottom countdown bar
    progress_bar(img, 10, h - 22, w - 20, 10,
                 elapsed / INTRO_DURATION, THEME['accent_cyan'])

    # thumb-up skip arc (top-right)
    if skip_frac > 0:
        hold_arc(img, w - 45, 45, 28, skip_frac, THEME['accent_green'])
        cv2.putText(img, 'SKIP', (w - 65, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_green'], 1)

    draw_top_pill(img, '  INTRO', THEME['accent_cyan'])


# ── 2a. ZONE INTRO (info only, no control) ───────────────────────────────────

ZONE_INTRO_DURATION = 30.0

def draw_zone_intro(img, elapsed, skip_frac):
    h, w = img.shape[:2]
    filled_rect(img, 0, 0, w, h, THEME['bg_dark'], 0.92)
    cv2.rectangle(img, (0, 0), (w, 4), THEME['accent_purple'], -1)

    text_centered(img, 'Next: Choose Movement Zone', h // 2 - 100, 0.9,
                  THEME['text_title'], 2)

    cv2.line(img, (w // 2 - 150, h // 2 - 62),
                  (w // 2 + 150, h // 2 - 62), THEME['divider'], 1)

    lines = [
        'On the next screen you will set your hand movement area.',
        'Show 1, 2 or 3 fingers and hold for 5 seconds to select:',
        '',
        '1 Finger  =  Small   (small precise movements)',
        '2 Fingers =  Medium  (balanced, recommended)',
        '3 Fingers =  Large   (wide sweeping arm movements)',
    ]
    colors = [
        THEME['text_secondary'], THEME['text_secondary'], THEME['text_secondary'],
        THEME['accent_cyan'], THEME['accent_green'], THEME['accent_orange'],
    ]
    for i, (line, col) in enumerate(zip(lines, colors)):
        text_centered(img, line, h // 2 - 42 + i * 26, 0.48, col, 1)

    cv2.line(img, (w // 2 - 150, h // 2 + 120),
                  (w // 2 + 150, h // 2 + 120), THEME['divider'], 1)

    text_centered(img, 'Thumb Up to skip  -  Auto-advances in 30s',
                  h // 2 + 140, 0.42, THEME['text_dim'], 1)

    progress_bar(img, 10, h - 22, w - 20, 10,
                 elapsed / ZONE_INTRO_DURATION, THEME['accent_purple'])

    if skip_frac > 0:
        hold_arc(img, w - 45, 45, 28, skip_frac, THEME['accent_green'])
        cv2.putText(img, 'SKIP', (w - 65, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_green'], 1)

    draw_top_pill(img, 'NEXT: ZONE SETUP', THEME['accent_purple'])


# ── 2b. ZONE PICK SCREEN ─────────────────────────────────────────────────────

ZONE_CONFIRM_TIME = 5.0   # seconds to hold finger gesture

def draw_zone_pick(img, elapsed, chosen, skip_frac, confirm_frac, confirm_secs_left):
    h, w = img.shape[:2]
    filled_rect(img, 0, 0, w, h, THEME['bg_dark'], 0.92)
    cv2.rectangle(img, (0, 0), (w, 4), THEME['accent_purple'], -1)

    text_centered(img, 'Choose Movement Zone', h // 2 - 115, 0.9,
                  THEME['text_title'], 2)
    text_centered(img, 'Hold finger gesture for 5 seconds to confirm',
                  h // 2 - 78, 0.48, THEME['text_secondary'], 1)

    zones   = [('small', '1 Finger', 'Small', 'precise moves'),
               ('medium','2 Fingers','Medium','recommended'),
               ('large', '3 Fingers','Large', 'wide movement')]
    colours = [THEME['accent_cyan'], THEME['accent_green'], THEME['accent_orange']]

    box_w, box_h = 155, 90
    gap          = 14
    total_w      = 3 * box_w + 2 * gap
    start_x      = (w - total_w) // 2
    top_y        = h // 2 - 38

    for i, (key, gesture, name, subdesc) in enumerate(zones):
        bx  = start_x + i * (box_w + gap)
        col = colours[i]
        active = (chosen == key)
        bg     = THEME['bg_accent'] if active else THEME['bg_panel']
        cv2.rectangle(img, (bx, top_y), (bx + box_w, top_y + box_h), bg, -1)
        bord_col   = col if active else THEME['border']
        bord_thick = 2   if active else 1
        cv2.rectangle(img, (bx, top_y), (bx + box_w, top_y + box_h),
                      bord_col, bord_thick)

        (gw, _), _ = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, gesture, (bx + (box_w - gw) // 2, top_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        (nw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(img, name, (bx + (box_w - nw) // 2, top_y + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(subdesc, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
        cv2.putText(img, subdesc, (bx + (box_w - sw) // 2, top_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, THEME['text_secondary'], 1)

        # per-box confirm bar if this zone is active
        if active and confirm_frac > 0:
            bar_y = top_y + box_h + 4
            progress_bar(img, bx, bar_y, box_w, 6, confirm_frac, col)

    if chosen and confirm_frac > 0:
        msg = f'Hold... {confirm_secs_left:.1f}s  ({chosen.upper()})'
        text_centered(img, msg, h // 2 + 78, 0.5, THEME['accent_green'], 1)

    progress_bar(img, 10, h - 22, w - 20, 10,
                 elapsed / ZONE_DURATION, THEME['accent_purple'])

    if skip_frac > 0:
        hold_arc(img, w - 45, 45, 28, skip_frac, THEME['accent_green'])
        cv2.putText(img, 'SKIP', (w - 65, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_green'], 1)

    draw_top_pill(img, 'ZONE SETUP', THEME['accent_purple'])


# ── 3. GUIDE SCREEN ──────────────────────────────────────────────────────────

def draw_guide(img, elapsed, skip_frac):
    h, w = img.shape[:2]
    filled_rect(img, 0, 0, w, h, THEME['bg_dark'], 0.92)
    cv2.rectangle(img, (0, 0), (w, 4), THEME['accent_green'], -1)

    text_centered(img, 'Gesture Guide', 28, 0.8, THEME['text_title'], 2)

    # ── mouse gestures (left column) ─────────────────────────────────────────
    col1_x = 18
    y0     = 52

    mouse_gestures = [
        ('MOVE',        'Index finger up only',          THEME['gest_move']),
        ('LEFT CLICK',  'Thumb+Middle pinch (<0.5s)',     THEME['gest_left_click']),
        ('DRAG',        'Thumb+Middle pinch (hold 0.5s)', THEME['gest_drag']),
        ('RIGHT CLICK', 'Thumb+Ring pinch',               THEME['gest_right_click']),
        ('SCROLL UP',   'Index+Middle+Ring up',           THEME['gest_scroll_up']),
        ('SCROLL DOWN', 'All 4 fingers curled',           THEME['gest_scroll_down']),
    ]

    cv2.putText(img, 'MOUSE CONTROLS', (col1_x, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, THEME['text_dim'], 1)
    for i, (name, desc, col) in enumerate(mouse_gestures):
        cy = y0 + 18 + i * 28
        cv2.circle(img, (col1_x + 5, cy - 4), 4, col, -1)
        cv2.putText(img, name, (col1_x + 14, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(img, desc, (col1_x + 14, cy + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, THEME['text_secondary'], 1)

    # divider
    mid_x = w // 2 - 10
    cv2.line(img, (mid_x, 46), (mid_x, h - 55), THEME['divider'], 1)

    # ── game + meta controls (right column) ──────────────────────────────────
    col2_x = w // 2

    game_gestures = [
        ('GAME OPT 1', '1 Fist + 1 finger (hold 5s)', THEME['accent_cyan']),
        ('GAME OPT 2', '1 Fist + 2 fingers (hold 5s)',THEME['accent_green']),
        ('GAME OPT 3', '1 Fist + 3 fingers (hold 5s)',THEME['accent_orange']),
        ('GAME OPT 4', '1 Fist + 4 fingers (hold 5s)',THEME['accent_pink']),
    ]

    cv2.putText(img, 'GAME OPTIONS', (col2_x, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, THEME['text_dim'], 1)
    for i, (name, desc, col) in enumerate(game_gestures):
        cy = y0 + 18 + i * 28
        cv2.circle(img, (col2_x + 5, cy - 4), 4, col, -1)
        cv2.putText(img, name, (col2_x + 14, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(img, desc, (col2_x + 14, cy + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, THEME['text_secondary'], 1)

    # separator between game + meta
    sep_y = y0 + 18 + len(game_gestures) * 28 + 8
    cv2.line(img, (col2_x, sep_y), (w - 10, sep_y), THEME['divider'], 1)

    meta_gestures = [
        ('START/RESUME', 'Both peace signs (3s)', THEME['accent_green']),
        ('PAUSE',        'Open palm (3s)',        THEME['accent_yellow']),
        ('RECALIBRATE',  'Shaka (3s) -> zone',   THEME['accent_purple']),
        ('GUIDE',        'Metal sign (index+pinky)', THEME['accent_cyan']),
        ('CLOSE APP',    'Both fists (3s)',       THEME['accent_red']),
    ]

    cv2.putText(img, 'META CONTROLS', (col2_x, sep_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, THEME['text_dim'], 1)
    for i, (name, desc, col) in enumerate(meta_gestures):
        cy = sep_y + 30 + i * 24
        cv2.circle(img, (col2_x + 5, cy - 4), 3, col, -1)
        cv2.putText(img, f'{name}: {desc}', (col2_x + 14, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)

    progress_bar(img, 10, h - 22, w - 20, 10,
                 elapsed / GUIDE_DURATION, THEME['accent_green'])

    if skip_frac > 0:
        hold_arc(img, w - 45, 45, 28, skip_frac, THEME['accent_green'])
        cv2.putText(img, 'SKIP', (w - 65, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_green'], 1)

    draw_top_pill(img, 'GESTURE GUIDE', THEME['accent_green'])


# ── 4. DISTANCE CHECK ─────────────────────────────────────────────────────────

def draw_distance_ui(img, dist, has_hand, hold_frac=0.0):
    h, w = img.shape[:2]

    filled_rect(img, 0, 0, w, 110, THEME['bg_dark'], 0.9)

    if not has_hand:
        text_centered(img, 'Show your hand to the camera',
                      55, 0.8, THEME['text_secondary'], 1)
        draw_top_pill(img, '  DISTANCE CHECK', THEME['accent_cyan'])
        return

    too_close = dist > TARGET_DIST + DIST_TOL
    too_far   = dist < TARGET_DIST - DIST_TOL
    if too_close:
        msg, col = 'Move FARTHER from camera', THEME['bar_fill_warn']
    elif too_far:
        msg, col = 'Move CLOSER to camera',   THEME['bar_fill_far']
    else:
        msg, col = 'Distance OK!  Hold still...', THEME['bar_fill_ok']

    text_centered(img, msg, 42, 0.75, col, 2)

    bx, by, bw, bh = 10, 58, w - 20, 14
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), THEME['bar_track'], -1)
    fill = int(np.clip(dist / (TARGET_DIST * 2), 0, 1) * bw)
    cv2.rectangle(img, (bx, by), (bx + fill, by + bh), col, -1)
    lo = int((TARGET_DIST - DIST_TOL) / (TARGET_DIST * 2) * bw) + bx
    hi = int((TARGET_DIST + DIST_TOL) / (TARGET_DIST * 2) * bw) + bx
    cv2.rectangle(img, (lo, by), (hi, by + bh), THEME['accent_green'], 2)

    if hold_frac > 0:
        progress_bar(img, 10, 80, w - 20, 8, hold_frac, THEME['accent_green'])
        text_centered(img, f'Hold still... {hold_frac*DIST_OK_HOLD:.1f}s',
                      100, 0.42, THEME['accent_green'], 1)

    draw_top_pill(img, '  DISTANCE CHECK', THEME['accent_cyan'])


# ── 5. RUNNING HUD ────────────────────────────────────────────────────────────

GESTURE_COLOR = {
    'move':        THEME['gest_move'],
    'left_click':  THEME['gest_left_click'],
    'right_click': THEME['gest_right_click'],
    'drag':        THEME['gest_drag'],
    'scroll_up':   THEME['gest_scroll_up'],
    'scroll_down': THEME['gest_scroll_down'],
    'idle':        THEME['gest_idle'],
}

def draw_running_ui(img, gesture, dist_ok, dist_drift_msg,
                    scroll_entry_t, scroll_active,
                    tm_pinch_start, tm_drag_active,
                    meta_hold_fracs, game_opt_frac, game_opt_number):
    h, w = img.shape[:2]

    # ── Fixed pixel layout — nothing overlaps ─────────────────────────────────
    # [0     .. BAR_BOT]  = gesture bar          (70px tall)
    # [BAR_BOT+1 .. ARC_BOT] = arc strip         (52px tall)
    # [ARC_BOT+1 .. h]   = free / legend at bottom
    BAR_BOT     = 70
    ARC_BOT     = BAR_BOT + 52
    CONTENT_Y   = ARC_BOT + 4   # first usable y below arcs
    LEGEND_SP   = 19
    LEGEND_LINES= 7
    LEGEND_BOT  = h - 6
    LEGEND_TOP  = LEGEND_BOT - (LEGEND_LINES - 1) * LEGEND_SP

    # ── Gesture bar ───────────────────────────────────────────────────────────
    filled_rect(img, 0, 0, w, BAR_BOT, THEME['bg_dark'], 0.92)
    cv2.line(img, (0, BAR_BOT), (w, BAR_BOT), THEME['border'], 2)

    if tm_drag_active:
        label, col = 'DRAG & DROP  [HOLDING]', GESTURE_COLOR['drag']
    elif gesture == 'tm_pinch' and tm_pinch_start:
        held  = time.time() - tm_pinch_start
        pct   = min(held / LONG_PRESS_TIME, 1.0)
        col   = GESTURE_COLOR['left_click']
        label = f'PINCH  {held:.2f}s'
        progress_bar(img, 0, BAR_BOT - 6, w, 6, pct, THEME['accent_cyan'])
        hint  = '>> DRAG' if pct >= 1.0 else f'release=CLICK  hold=DRAG'
        cv2.putText(img, hint, (w - 180, BAR_BOT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, THEME['text_secondary'], 1)
    elif gesture == 'right_click': label, col = 'RIGHT CLICK',   GESTURE_COLOR['right_click']
    elif gesture == 'move':        label, col = 'MOVE',           GESTURE_COLOR['move']
    elif gesture == 'scroll_up':   label, col = 'SCROLL UP  ^',  GESTURE_COLOR['scroll_up']
    elif gesture == 'scroll_down': label, col = 'SCROLL DOWN v', GESTURE_COLOR['scroll_down']
    else:                          label, col = 'IDLE',           GESTURE_COLOR['idle']

    # Gesture text vertically centered in bar, with left margin only
    full_label = f'Gesture: {label}'
    (_, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    text_y = (BAR_BOT + th) // 2   # vertically center in bar
    cv2.putText(img, full_label, (12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)

    # ── Arc strip ─────────────────────────────────────────────────────────────
    filled_rect(img, 0, BAR_BOT + 1, w, ARC_BOT, THEME['bg_panel'], 0.82)
    cv2.line(img, (0, ARC_BOT), (w, ARC_BOT), THEME['border'], 2)

    ARC_CY = BAR_BOT + 1 + (ARC_BOT - BAR_BOT - 1) // 2   # vertically centre
    ARC_R  = 14
    arc_items = [
        ('PAUSE',  'stop',     THEME['accent_yellow'], 'Palm 3s'),
        ('RECAL',  'recal',    THEME['accent_purple'], 'Shaka 3s'),
        ('CLOSE',  'close',    THEME['accent_red'],    'Fists 3s'),
        ('GUIDE',  'guide',    THEME['accent_cyan'],   'Metal sign'),
        (f'OPT{game_opt_number or "?"}', 'game_opt',
         THEME['accent_orange'], f'Fist+{game_opt_number or "?"}F 5s'),
    ]
    spacing = w // len(arc_items)
    for i, (name, key, ic, hint) in enumerate(arc_items):
        cx   = i * spacing + spacing // 2
        frac = meta_hold_fracs.get(key, 0.0)
        cv2.circle(img, (cx, ARC_CY), ARC_R, THEME['bg_dark'], -1)
        cv2.circle(img, (cx, ARC_CY), ARC_R,
                   ic if frac > 0 else THEME['border'], 1)
        hold_arc(img, cx, ARC_CY, ARC_R - 2, frac, ic)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.27, 1)
        cv2.putText(img, name, (cx - tw // 2, ARC_CY + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, ic, 1, cv2.LINE_AA)
        (hw2, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.24, 1)
        cv2.putText(img, hint, (cx - hw2 // 2, ARC_BOT - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, THEME['text_dim'], 1, cv2.LINE_AA)

    # ── Content zone (stacks downward from CONTENT_Y) ─────────────────────────
    cy = CONTENT_Y

    if game_opt_frac > 0 and game_opt_number:
        opt_colors = [THEME['accent_cyan'], THEME['accent_green'],
                      THEME['accent_orange'], THEME['accent_pink']]
        oc = opt_colors[(game_opt_number - 1) % 4]
        filled_rect(img, 0, cy, w, cy + 18, THEME['bg_accent'], 0.9)
        progress_bar(img, 0, cy, w, 18, game_opt_frac, oc)
        secs = max(0.0, HOLD_GAME * (1.0 - game_opt_frac))
        text_centered(img, f'GAME OPT {game_opt_number}  {secs:.1f}s', cy + 13,
                      0.38, THEME['text_title'], 1)
        cy += 20

    if dist_drift_msg:
        filled_rect(img, 0, cy, w, cy + 18, THEME['bg_warn'], 0.88)
        cv2.putText(img, dist_drift_msg, (8, cy + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_yellow'], 1, cv2.LINE_AA)
        cy += 20

    if gesture in ('scroll_up', 'scroll_down'):
        if scroll_entry_t and not scroll_active:
            remaining = max(0.0, SCROLL_BUFFER - (time.time() - scroll_entry_t))
            filled_rect(img, 0, cy, 260, cy + 18, THEME['bg_panel'], 0.85)
            cv2.putText(img, f'Scroll locks in {remaining:.1f}s...', (8, cy + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_cyan'], 1)
        elif scroll_active:
            dir_s = 'UP  ^' if gesture == 'scroll_up' else 'DOWN  v'
            filled_rect(img, 0, cy, 200, cy + 18, THEME['bg_success'], 0.85)
            cv2.putText(img, f'Scrolling {dir_s}', (8, cy + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_green'], 2)

    # ── Legend pinned to bottom — guaranteed below arc strip ──────────────────
    legend = [
        ('Index up only      = MOVE',                    THEME['gest_move']),
        (f'Thumb+Mid <{LONG_PRESS_TIME}s   = LEFT CLICK',THEME['gest_left_click']),
        (f'Thumb+Mid >={LONG_PRESS_TIME}s  = DRAG',      THEME['gest_drag']),
        ('Thumb+Ring         = RIGHT CLICK',              THEME['gest_right_click']),
        ('Idx+Mid+Ring up    = SCROLL UP',                THEME['gest_scroll_up']),
        ('All fingers curled = SCROLL DOWN',              THEME['gest_scroll_down']),
        ('Fist + 1-4 fingers = GAME OPT (5s)',            THEME['accent_orange']),
    ]
    for i, (txt, c) in enumerate(legend):
        y = LEGEND_TOP + i * LEGEND_SP
        if y > ARC_BOT + 10:
            cv2.putText(img, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE + CAP
# ══════════════════════════════════════════════════════════════════════════════

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector     = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

def detection_worker():
    global _latest_result, _stop_thread
    while not _stop_thread:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.001)
            continue
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        with _result_lock:
            _latest_result = result

det_thread = threading.Thread(target=detection_worker, daemon=True)
det_thread.start()

# ══════════════════════════════════════════════════════════════════════════════
#  RESET
# ══════════════════════════════════════════════════════════════════════════════

def full_reset():
    global app_state, dist_ok_since, smooth_x, smooth_y
    global last_click_t, prev_scroll_y, scroll_entry_t, scroll_active
    global tm_pinch_start, tm_drag_active, intro_start_t, zone_start_t, guide_start_t
    global meta_hold, game_option_pending
    if tm_drag_active:
        pyautogui.mouseUp()
    app_state       = 'distance_check'
    dist_ok_since   = None
    smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2
    last_click_t    = 0.0
    prev_scroll_y   = None
    scroll_entry_t  = None
    scroll_active   = False
    tm_pinch_start  = None
    tm_drag_active  = False
    intro_start_t   = None
    zone_start_t    = None
    guide_start_t   = None
    meta_hold       = {k: None for k in meta_hold}
    game_option_pending = None

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

intro_start_t      = time.time()
zone_intro_start_t = None
zone_start_t       = None
guide_start_t      = None
thumb_hold_t       = None
zone_finger_hold   = {}
game_opt_hold_t    = None
game_opt_number    = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    with _frame_lock:
        _latest_frame = frame.copy()
    with _result_lock:
        result = _latest_result

    has_hand = bool(result and result.hand_landmarks)
    # Primary hand (index 0) for mouse control
    lms      = result.hand_landmarks[0] if has_hand else None
    # Second hand (index 1) for two-hand meta gestures
    lms2     = result.hand_landmarks[1] if (has_hand and len(result.hand_landmarks) > 1) else None
    display  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fh, fw   = display.shape[:2]
    now      = time.time()

    # ── INTRO ─────────────────────────────────────────────────────────────────
    if app_state == 'intro':
        elapsed = now - intro_start_t

        # skip only allowed after SKIP_LOCKOUT seconds
        skip_frac = 0.0
        if elapsed >= SKIP_LOCKOUT and lms and is_thumbs_up(lms):
            if thumb_hold_t is None:
                thumb_hold_t = now
            skip_frac = min((now - thumb_hold_t) / 1.0, 1.0)
            if skip_frac >= 1.0:
                elapsed = INTRO_DURATION
        else:
            if not (lms and is_thumbs_up(lms)):
                thumb_hold_t = None

        if lms:
            draw_hand(display, lms)
        draw_intro(display, elapsed, skip_frac)

        if elapsed >= INTRO_DURATION:
            app_state          = 'zone_intro'
            zone_intro_start_t = now
            thumb_hold_t       = None

    # ── ZONE INTRO (info only) ────────────────────────────────────────────────
    elif app_state == 'zone_intro':
        elapsed = now - zone_intro_start_t

        skip_frac = 0.0
        if elapsed >= SKIP_LOCKOUT and lms and is_thumbs_up(lms):
            if thumb_hold_t is None: thumb_hold_t = now
            skip_frac = min((now - thumb_hold_t) / 1.0, 1.0)
            if skip_frac >= 1.0: elapsed = ZONE_INTRO_DURATION
        else:
            if not (lms and is_thumbs_up(lms)):
                thumb_hold_t = None

        if lms: draw_hand(display, lms)
        draw_zone_intro(display, elapsed, skip_frac)

        if elapsed >= ZONE_INTRO_DURATION:
            app_state    = 'zone_pick'
            zone_start_t = now
            thumb_hold_t = None

    # ── ZONE PICK ─────────────────────────────────────────────────────────────
    elif app_state == 'zone_pick':
        elapsed = now - zone_start_t

        # detect finger count for zone choice
        detected_zone = None
        if lms:
            n = count_fingers_up(lms)
            if n == 1:   detected_zone = 'small'
            elif n == 2: detected_zone = 'medium'
            elif n >= 3: detected_zone = 'large'

        # hold to confirm (5 seconds)
        confirm_frac     = 0.0
        confirm_secs_left = 0.0
        if detected_zone:
            if detected_zone not in zone_finger_hold:
                zone_finger_hold = {detected_zone: now}
            held = now - zone_finger_hold[detected_zone]
            confirm_frac      = min(held / ZONE_CONFIRM_TIME, 1.0)
            confirm_secs_left = max(0.0, ZONE_CONFIRM_TIME - held)
            if confirm_frac >= 1.0:
                chosen_zone = detected_zone
                set_zone_from_choice(chosen_zone)
                app_state     = 'guide'
                guide_start_t = now
                thumb_hold_t  = None
                zone_finger_hold = {}
        else:
            zone_finger_hold = {}

        # skip via thumbs-up (locked out first 5s)
        skip_frac = 0.0
        if elapsed >= SKIP_LOCKOUT and lms and is_thumbs_up(lms):
            if thumb_hold_t is None:
                thumb_hold_t = now
            skip_frac = min((now - thumb_hold_t) / 1.0, 1.0)
            if skip_frac >= 1.0:
                set_zone_from_choice(chosen_zone)
                app_state    = 'guide'
                guide_start_t = now
                thumb_hold_t  = None
        else:
            if not (lms and is_thumbs_up(lms)):
                thumb_hold_t = None

        # auto-advance
        if elapsed >= ZONE_DURATION and app_state == 'zone_pick':
            set_zone_from_choice(chosen_zone)
            app_state    = 'guide'
            guide_start_t = now
            thumb_hold_t  = None

        if lms:
            draw_hand(display, lms)
        draw_zone_pick(display, elapsed, detected_zone or chosen_zone,
                       skip_frac, confirm_frac, confirm_secs_left)

    # ── GUIDE ─────────────────────────────────────────────────────────────────
    elif app_state == 'guide':
        elapsed = now - guide_start_t

        skip_frac = 0.0
        if elapsed >= SKIP_LOCKOUT and lms and is_thumbs_up(lms):
            if thumb_hold_t is None:
                thumb_hold_t = now
            skip_frac = min((now - thumb_hold_t) / 1.0, 1.0)
            if skip_frac >= 1.0:
                elapsed = GUIDE_DURATION
        else:
            if not (lms and is_thumbs_up(lms)):
                thumb_hold_t = None

        if lms:
            draw_hand(display, lms)
        draw_guide(display, elapsed, skip_frac)

        if elapsed >= GUIDE_DURATION:
            app_state     = 'distance_check'
            dist_ok_since = None
            thumb_hold_t  = None

    # ── DISTANCE CHECK ────────────────────────────────────────────────────────
    elif app_state == 'distance_check':
        dist     = hand_size(lms) if lms else 0.0
        in_range = lms and abs(dist - TARGET_DIST) <= DIST_TOL

        if in_range:
            if dist_ok_since is None:
                dist_ok_since = now
        else:
            dist_ok_since = None

        hold_frac = 0.0
        if dist_ok_since:
            hold_frac = min((now - dist_ok_since) / DIST_OK_HOLD, 1.0)
            if hold_frac >= 1.0:
                app_state     = 'running'
                dist_ok_since = None

        if lms:
            draw_hand(display, lms)
        draw_distance_ui(display, dist, has_hand, hold_frac)

    # ── RUNNING ───────────────────────────────────────────────────────────────
    elif app_state == 'running':
        gesture = 'idle'

        # draw calibrated zone box
        if range_min_x is not None:
            bx1 = int(range_min_x * fw); bx2 = int(range_max_x * fw)
            by1 = int(range_min_y * fh); by2 = int(range_max_y * fh)
            cv2.rectangle(display, (bx1, by1), (bx2, by2), THEME['accent_green'], 1)

        # ── meta-gesture detection ────────────────────────────────────────────
        meta_hold_fracs = {k: 0.0 for k in ('start', 'stop', 'recal', 'close', 'guide', 'game_opt')}
        triggered_meta  = None

        if lms:
            # PAUSE — one hand open palm (3s)
            if is_open_palm(lms):
                if meta_hold['stop'] is None: meta_hold['stop'] = now
                meta_hold_fracs['stop'] = min((now - meta_hold['stop']) / HOLD_META, 1.0)
                if meta_hold_fracs['stop'] >= 1.0:
                    triggered_meta = 'stop'
            else:
                meta_hold['stop'] = None

            # RECALIBRATE — one hand shaka (3s) -> zone_pick
            if is_shaka(lms):
                if meta_hold['recal'] is None: meta_hold['recal'] = now
                meta_hold_fracs['recal'] = min((now - meta_hold['recal']) / HOLD_META, 1.0)
                if meta_hold_fracs['recal'] >= 1.0:
                    triggered_meta = 'recal'
            else:
                meta_hold['recal'] = None

            # GUIDE — metal sign (index + pinky up) -> back to guide screen
            if is_metal_sign(lms):
                if meta_hold['guide'] is None: meta_hold['guide'] = now
                meta_hold_fracs['guide'] = min((now - meta_hold['guide']) / HOLD_META, 1.0)
                if meta_hold_fracs['guide'] >= 1.0:
                    triggered_meta = 'guide'
            else:
                meta_hold['guide'] = None

        # START — BOTH peace signs (3s) — distinct from fist, no conflict
        both_peace = lms and lms2 and is_peace_sign(lms) and is_peace_sign(lms2)
        if both_peace:
            if meta_hold['start'] is None: meta_hold['start'] = now
            meta_hold_fracs['start'] = min((now - meta_hold['start']) / HOLD_META, 1.0)
        else:
            meta_hold['start'] = None

        # CLOSE — BOTH fists (3s) — only when no game option is being detected
        game_opt_now = get_game_option(lms, lms2)
        both_fists   = lms and lms2 and is_fist(lms) and is_fist(lms2)
        if both_fists and game_opt_now is None:
            if meta_hold['close'] is None: meta_hold['close'] = now
            meta_hold_fracs['close'] = min((now - meta_hold['close']) / HOLD_CLOSE, 1.0)
            if meta_hold_fracs['close'] >= 1.0:
                triggered_meta = 'close'
        else:
            meta_hold['close'] = None

        # GAME OPTION — one fist + 1-4 fingers on other hand (5s)
        if game_opt_now is not None:
            if game_opt_now != game_opt_number:
                # switched option, reset timer
                game_opt_hold_t  = now
                game_opt_number  = game_opt_now
            else:
                if game_opt_hold_t is None: game_opt_hold_t = now
            held = now - game_opt_hold_t if game_opt_hold_t else 0
            meta_hold_fracs['game_opt'] = min(held / HOLD_GAME, 1.0)
            if meta_hold_fracs['game_opt'] >= 1.0:
                triggered_meta = f'game_{game_opt_number}'
        else:
            game_opt_hold_t = None
            game_opt_number = None

        if triggered_meta == 'stop':
            handle_pinch_release(now)
            app_state = 'stopped'
            for k in meta_hold: meta_hold[k] = None
        elif triggered_meta == 'recal':
            handle_pinch_release(now)
            app_state    = 'zone_pick'
            zone_start_t = now
            for k in meta_hold: meta_hold[k] = None
        elif triggered_meta == 'guide':
            handle_pinch_release(now)
            app_state     = 'guide'
            guide_start_t = now
            for k in meta_hold: meta_hold[k] = None
        elif triggered_meta and triggered_meta.startswith('game_'):
            opt = int(triggered_meta.split('_')[1])
            print(f'[GAME OPTION {opt} SELECTED]')
            game_opt_hold_t = None
            game_opt_number = None
            for k in meta_hold: meta_hold[k] = None
            # TODO: hook your game option callback here
        elif triggered_meta == 'close':
            handle_pinch_release(now)
            break  # exit loop -> cleanup

        # ── normal mouse gesture processing ───────────────────────────────────
        any_meta_active = any(v is not None for v in meta_hold.values()) or game_opt_now is not None

        if lms and triggered_meta is None and not any_meta_active:
            gesture = get_gesture(lms)
            dist    = hand_size(lms)
            dist_ok = abs(dist - TARGET_DIST) <= DIST_TOL * 2

            # ── distance drift message ────────────────────────────────────────
            dist_drift_msg = None
            if lms:
                drift = dist - TARGET_DIST
                if drift > DIST_TOL * 3:
                    dist_drift_msg = 'WARNING: Too close - move hand further away'
                elif drift > DIST_TOL * 2:
                    dist_drift_msg = 'Drifted: slightly too close to camera'
                elif drift < -DIST_TOL * 3:
                    dist_drift_msg = 'WARNING: Too far - bring hand closer'
                elif drift < -DIST_TOL * 2:
                    dist_drift_msg = 'Drifted: slightly too far from camera'

            # ── lighting check (low brightness = poor detection) ──────────────
            if not dist_drift_msg:
                gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness  = float(np.mean(gray))
                if brightness < 40:
                    dist_drift_msg = 'Poor lighting - turn on more light for better tracking'
                elif brightness < 65:
                    dist_drift_msg = 'Low lighting - tracking may be unreliable'

            target_x, target_y = map_cursor(lms[8].x, lms[8].y)
            smooth_x += (target_x - smooth_x) * (1 - SMOOTHING)
            smooth_y += (target_y - smooth_y) * (1 - SMOOTHING)

            if gesture == 'move':
                handle_pinch_release(now)
                pyautogui.moveTo(int(smooth_x), int(smooth_y))
                scroll_entry_t = None; scroll_active = False; prev_scroll_y = None

            elif gesture == 'tm_pinch':
                if tm_pinch_start is None: tm_pinch_start = now
                held = now - tm_pinch_start
                if not tm_drag_active and held >= LONG_PRESS_TIME:
                    pyautogui.mouseDown()
                    tm_drag_active = True
                if tm_drag_active:
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))
                scroll_entry_t = None; scroll_active = False; prev_scroll_y = None

            elif gesture == 'right_click':
                handle_pinch_release(now)
                if now - last_click_t > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_click_t = now
                scroll_entry_t = None; scroll_active = False; prev_scroll_y = None

            elif gesture in ('scroll_up', 'scroll_down'):
                handle_pinch_release(now)
                if scroll_entry_t is None:
                    scroll_entry_t = now; scroll_active = False
                elif not scroll_active and (now - scroll_entry_t) >= SCROLL_BUFFER:
                    scroll_active = True
                if scroll_active:
                    pyautogui.scroll(SCROLL_SPEED if gesture == 'scroll_up' else -SCROLL_SPEED)
                prev_scroll_y = None

            else:
                handle_pinch_release(now)
                scroll_entry_t = None; scroll_active = False; prev_scroll_y = None

            fx, fy    = int(lms[8].x * fw), int(lms[8].y * fh)
            dot_color = GESTURE_COLOR['drag'] if tm_drag_active else GESTURE_COLOR.get(gesture, THEME['text_primary'])
            cv2.circle(display, (fx, fy), 12, dot_color, 2)
            cv2.circle(display, (fx, fy),  4, dot_color, -1)
            draw_hand(display, lms)
            if lms2: draw_hand(display, lms2)
            draw_running_ui(display, gesture, dist_ok, dist_drift_msg,
                            scroll_entry_t, scroll_active,
                            tm_pinch_start, tm_drag_active,
                            meta_hold_fracs,
                            meta_hold_fracs.get('game_opt', 0.0),
                            game_opt_number)
        else:
            handle_pinch_release(now)
            scroll_entry_t = None; scroll_active = False; prev_scroll_y = None
            if lms:  draw_hand(display, lms)
            if lms2: draw_hand(display, lms2)
            draw_running_ui(display, 'idle', True, None,
                            None, False, None, False,
                            meta_hold_fracs,
                            meta_hold_fracs.get('game_opt', 0.0),
                            game_opt_number)

    # ── STOPPED (paused) ──────────────────────────────────────────────────────
    elif app_state == 'stopped':
        filled_rect(display, 0, 0, fw, fh, THEME['bg_dark'], 0.88)
        cv2.rectangle(display, (0, 0), (fw, 4), THEME['accent_red'], -1)
        text_centered(display, 'PAUSED', fh // 2 - 40, 1.2,
                      THEME['accent_red'], 3)
        text_centered(display, 'Both Peace Signs (hold 3s) to resume',
                      fh // 2 + 10, 0.55, THEME['text_secondary'], 1)
        text_centered(display, 'Both Fists (hold 3s) to close',
                      fh // 2 + 36, 0.5, THEME['text_dim'], 1)

        # resume via BOTH peace signs (3s) — no conflict with fist/close
        resume_frac = 0.0
        close_frac  = 0.0
        both_peace_pause = lms and lms2 and is_peace_sign(lms) and is_peace_sign(lms2)
        both_fists_pause = lms and lms2 and is_fist(lms) and is_fist(lms2)

        if both_peace_pause:
            if thumb_hold_t is None: thumb_hold_t = now
            resume_frac = min((now - thumb_hold_t) / HOLD_META, 1.0)
            hold_arc(display, fw // 2, fh // 2 + 80, 30, resume_frac,
                     THEME['accent_green'])
            if resume_frac >= 1.0:
                app_state    = 'running'
                thumb_hold_t = None
                for k in meta_hold: meta_hold[k] = None
        else:
            thumb_hold_t = None

        if both_fists_pause:
            if meta_hold['close'] is None: meta_hold['close'] = now
            close_frac = min((now - meta_hold['close']) / HOLD_CLOSE, 1.0)
            hold_arc(display, fw // 2 + 70, fh // 2 + 80, 30, close_frac,
                     THEME['accent_red'])
            if close_frac >= 1.0:
                break
        else:
            meta_hold['close'] = None

        if lms:  draw_hand(display, lms)
        if lms2: draw_hand(display, lms2)
        draw_top_pill(display, '  PAUSED', THEME['accent_red'])

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    output = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    cv2.imshow('HandMouse', output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in (ord('r'), ord('R')):
        full_reset()

# ══════════════════════════════════════════════════════════════════════════════
#  CLEANUP
# ══════════════════════════════════════════════════════════════════════════════
handle_pinch_release(time.time())
_stop_thread = True
det_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
detector.close()