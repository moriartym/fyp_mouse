import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import time
import threading
import math

try:
    import torch
    import torch.nn as nn
    import joblib
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] torch/joblib not found — Game Options 2 & 3 will be unavailable.")

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

THEME = {
    'bg_dark':       (15,  15,  20),
    'bg_panel':      (25,  28,  38),
    'bg_accent':     (30,  38,  55),
    'bg_warn':       (60,  20,  20),
    'bg_success':    (10,  50,  30),
    'text_primary':  (230, 230, 240),
    'text_secondary':(150, 155, 170),
    'text_title':    (255, 255, 255),
    'text_dim':      (90,  95, 110),
    'accent_green':  (0,   220,  90),
    'accent_cyan':   (0,   200, 255),
    'accent_yellow': (255, 200,   0),
    'accent_orange': (255, 140,  30),
    'accent_pink':   (255,  70, 180),
    'accent_blue':   (60,  140, 255),
    'accent_red':    (220,  40,  60),
    'accent_purple': (160,  80, 255),
    'gest_move':         (255, 200,   0),
    'gest_left_click':   (  0, 200, 255),
    'gest_right_click':  (255,  70, 180),
    'gest_drag':         (  0, 120, 255),
    'gest_scroll_up':    (180, 255,  80),
    'gest_scroll_down':  ( 80, 180, 255),
    'gest_idle':         (120, 120, 130),
    'hand_bone':     (0,   200, 110),
    'hand_joint':    (255, 255, 255),
    'hand_joint_bd': (0,   150,  80),
    'bar_track':     (55,  58,  70),
    'bar_zone':      (0,   90,  40),
    'bar_fill_ok':   (0,   210,  80),
    'bar_fill_warn': (0,   80,  255),
    'bar_fill_far':  (0,   190, 255),
    'border':        (55,  60,  80),
    'divider':       (40,  42,  58),
}

SMOOTHING        = 0.3
PINCH_THRESH     = 0.03
CLICK_COOLDOWN   = 0.5
SCROLL_SPEED     = 3
LONG_PRESS_TIME  = 0.5
SCROLL_BUFFER    = 0
TARGET_DIST      = 0.18
DIST_TOL         = 0.03
DIST_OK_HOLD     = 3.0
INTRO_DURATION   = 30.0
ZONE_DURATION    = 30.0
GUIDE_DURATION   = 30.0
SKIP_LOCKOUT     = 3.0
HOLD_META        = 3.0
HOLD_CLOSE       = 3.0
HOLD_GAME        = 5.0

SS_KEY_COOLDOWN       = 0.3
SS_SPACE_COOLDOWN     = 1.0
SS_CONFIDENCE_THRESH  = 0.75

RC_STEER_DEADZONE = 5
RC_STEER_MAX      = 40
RC_CONF_THRESH    = 0.6

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

ZONE_PRESETS = {
    'small':  0.25,
    'medium': 0.55,
    'large':  0.90,
}


class GestureNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(126, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,  64), nn.BatchNorm1d( 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear( 64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _load_nn(weights_path, encoder_path, tag):
    if not TORCH_AVAILABLE:
        return None, None
    try:
        le  = joblib.load(encoder_path)
        net = GestureNet(len(le.classes_))
        net.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
        net.eval()
        print(f"[NN:{tag}] Loaded — classes: {list(le.classes_)}")
        return net, le
    except FileNotFoundError as e:
        print(f"[NN:{tag}] File not found: {e}  — mode disabled.")
        return None, None
    except Exception as e:
        print(f"[NN:{tag}] Load error: {e}  — mode disabled.")
        return None, None


subway_model, subway_le = _load_nn(
    'gesture_model_subway.pt',
    'label_encoder_subway.pkl',
    'SUBWAY',
)

racing_model, racing_le = _load_nn(
    'gesture_model_racing.pt',
    'label_encoder_racing.pkl',
    'RACING',
)

app_state          = 'intro'
intro_start_t      = None
zone_intro_start_t = None
zone_start_t       = None
guide_start_t      = None
chosen_zone        = 'medium'
dist_ok_since      = None

range_min_x = range_max_x = None
range_min_y = range_max_y = None

last_click_t   = 0.0
prev_scroll_y  = None
scroll_entry_t = None
scroll_active  = False
tm_pinch_start = None
tm_drag_active = False

smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2

meta_hold = {
    'start':    None,
    'stop':     None,
    'close':    None,
    'recal':    None,
    'game_opt': None,
}
game_option_pending = None
game_opt_number     = None

active_game_mode = None

ss_current_zone  = 'neutral'
ss_last_key_t    = 0.0
ss_last_space_t  = 0.0
ss_space_pressed = False
ss_prev_row      = None

rc_held_keys      = set()
rc_prev_row_left  = None
rc_prev_row_right = None

_latest_frame  = None
_latest_result = None
_frame_lock    = threading.Lock()
_result_lock   = threading.Lock()
_stop_thread   = False


def _extract_features(lms, prev_row):
    wx, wy, wz = lms[0].x, lms[0].y, lms[0].z
    scale = max(
        math.sqrt((lms[9].x-wx)**2 + (lms[9].y-wy)**2 + (lms[9].z-wz)**2),
        1e-6)
    row = []
    for lm in lms:
        row.extend([(lm.x-wx)/scale, (lm.y-wy)/scale, (lm.z-wz)/scale])
    delta = [c-p for c, p in zip(row, prev_row)] if prev_row else [0.0] * 63
    return row + delta, row


def run_nn(lms, prev_row, model, le, conf_thresh):
    if model is None or le is None:
        return 'none', 0.0, prev_row
    features, new_prev = _extract_features(lms, prev_row)
    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        probs     = torch.softmax(model(x), dim=1)[0]
        conf, idx = probs.max(0)
    if conf.item() < conf_thresh:
        return 'none', conf.item(), new_prev
    return le.inverse_transform([idx.item()])[0], conf.item(), new_prev


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
    return sum(fingers_extended(lms).values())

def is_thumbs_up(lms):
    thumb_up = lms[4].y < lms[3].y < lms[2].y
    f = fingers_extended(lms)
    others_down = not (f['index'] or f['middle'] or f['ring'] or f['pinky'])
    return thumb_up and others_down

def is_open_palm(lms):
    return all(fingers_extended(lms).values())

def is_fist(lms):
    return not any(fingers_extended(lms).values())

def is_shaka(lms):
    f = fingers_extended(lms)
    thumb_ext = lms[4].y < lms[3].y
    return thumb_ext and f['pinky'] and not f['index'] and not f['middle'] and not f['ring']

def is_metal_sign(lms):
    f = fingers_extended(lms)
    return f['index'] and f['pinky'] and not f['middle'] and not f['ring']

def is_peace_sign(lms):
    f = fingers_extended(lms)
    return f['index'] and f['middle'] and not f['ring'] and not f['pinky']

def get_game_option(lms, lms2):
    if lms is None or lms2 is None:
        return None
    for fist_hand, finger_hand in [(lms, lms2), (lms2, lms)]:
        if is_fist(fist_hand):
            n = count_fingers_up(finger_hand)
            if 1 <= n <= 4:
                return n
    return None

def get_gesture_mouse(lms):
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

def get_steer_angle(lms_left, lms_right):
    return np.degrees(np.arctan2(
        lms_right[0].y - lms_left[0].y,
        lms_right[0].x - lms_left[0].x))

def split_hands(result):
    lms_left  = None
    lms_right = None
    for hand in result.hand_landmarks:
        if hand[0].x < 0.5:
            lms_left  = hand
        else:
            lms_right = hand
    return lms_left, lms_right

def rc_set_held_keys(desired):
    for k in list(rc_held_keys - desired):
        try: pyautogui.keyUp(k)
        except: pass
        rc_held_keys.discard(k)
    for k in list(desired - rc_held_keys):
        pyautogui.keyDown(k)
        rc_held_keys.add(k)

def rc_release_all():
    for k in list(rc_held_keys):
        try: pyautogui.keyUp(k)
        except: pass
    rc_held_keys.clear()


def draw_hand(img, lms, bone_color=None):
    if lms is None:
        return
    bc = bone_color or THEME['hand_bone']
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], bc, 2)
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
    if frac <= 0:
        return
    axes    = (r, r)
    start_a = -90
    end_a   = int(-90 + 360 * min(frac, 1.0))
    cv2.ellipse(img, (cx, cy), axes, 0, start_a, end_a, color, 4)


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
    cv2.rectangle(img, (0, 0), (w, 4), THEME['accent_cyan'], -1)
    slide = INTRO_SLIDES[0]
    text_centered(img, slide['title'],    h // 2 - 100, 1.1, THEME['text_title'], 2)
    text_centered(img, slide['subtitle'], h // 2 - 60,  0.6, THEME['text_secondary'], 1)
    cv2.line(img, (w // 2 - 120, h // 2 - 42), (w // 2 + 120, h // 2 - 42), THEME['divider'], 1)
    for i, step in enumerate(slide['steps']):
        text_centered(img, step, h // 2 - 18 + i * 30, 0.55, THEME['text_primary'], 1)
    cv2.line(img, (w // 2 - 120, h // 2 + 78), (w // 2 + 120, h // 2 + 78), THEME['divider'], 1)
    text_centered(img, slide['hint'], h // 2 + 100, 0.45, THEME['text_secondary'], 1)
    progress_bar(img, 10, h - 22, w - 20, 10, elapsed / INTRO_DURATION, THEME['accent_cyan'])
    if skip_frac > 0:
        hold_arc(img, w - 45, 45, 28, skip_frac, THEME['accent_green'])
        cv2.putText(img, 'SKIP', (w - 65, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_green'], 1)
    draw_top_pill(img, '  INTRO', THEME['accent_cyan'])


ZONE_INTRO_DURATION = 30.0

def draw_zone_intro(img, elapsed, skip_frac):
    h, w = img.shape[:2]
    filled_rect(img, 0, 0, w, h, THEME['bg_dark'], 0.92)
    cv2.rectangle(img, (0, 0), (w, 4), THEME['accent_purple'], -1)
    text_centered(img, 'Next: Choose Movement Zone', h // 2 - 100, 0.9, THEME['text_title'], 2)
    cv2.line(img, (w // 2 - 150, h // 2 - 62), (w // 2 + 150, h // 2 - 62), THEME['divider'], 1)
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
    cv2.line(img, (w // 2 - 150, h // 2 + 120), (w // 2 + 150, h // 2 + 120), THEME['divider'], 1)
    text_centered(img, 'Thumb Up to skip  -  Auto-advances in 30s', h // 2 + 140, 0.42, THEME['text_dim'], 1)
    progress_bar(img, 10, h - 22, w - 20, 10, elapsed / ZONE_INTRO_DURATION, THEME['accent_purple'])
    if skip_frac > 0:
        hold_arc(img, w - 45, 45, 28, skip_frac, THEME['accent_green'])
        cv2.putText(img, 'SKIP', (w - 65, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_green'], 1)
    draw_top_pill(img, 'NEXT: ZONE SETUP', THEME['accent_purple'])


ZONE_CONFIRM_TIME = 3.0

def draw_zone_pick(img, elapsed, chosen, skip_frac, confirm_frac, confirm_secs_left):
    h, w = img.shape[:2]
    filled_rect(img, 0, 0, w, h, THEME['bg_dark'], 0.92)
    cv2.rectangle(img, (0, 0), (w, 4), THEME['accent_purple'], -1)
    text_centered(img, 'Choose Movement Zone', h // 2 - 115, 0.9, THEME['text_title'], 2)
    text_centered(img, 'Hold finger gesture for 5 seconds to confirm', h // 2 - 78, 0.48, THEME['text_secondary'], 1)
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
        cv2.rectangle(img, (bx, top_y), (bx + box_w, top_y + box_h), bord_col, bord_thick)
        (gw, _), _ = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, gesture, (bx + (box_w - gw) // 2, top_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        (nw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(img, name, (bx + (box_w - nw) // 2, top_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(subdesc, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
        cv2.putText(img, subdesc, (bx + (box_w - sw) // 2, top_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.33, THEME['text_secondary'], 1)
        if active and confirm_frac > 0:
            progress_bar(img, bx, top_y + box_h + 4, box_w, 6, confirm_frac, col)
    if chosen and confirm_frac > 0:
        text_centered(img, f'Hold... {confirm_secs_left:.1f}s  ({chosen.upper()})', h // 2 + 78, 0.5, THEME['accent_green'], 1)
    progress_bar(img, 10, h - 22, w - 20, 10, elapsed / ZONE_DURATION, THEME['accent_purple'])
    if skip_frac > 0:
        hold_arc(img, w - 45, 45, 28, skip_frac, THEME['accent_green'])
        cv2.putText(img, 'SKIP', (w - 65, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_green'], 1)
    draw_top_pill(img, 'ZONE SETUP', THEME['accent_purple'])


def draw_guide(img, elapsed, skip_frac):
    h, w = img.shape[:2]
    filled_rect(img, 0, 0, w, h, THEME['bg_dark'], 0.92)
    cv2.rectangle(img, (0, 0), (w, 4), THEME['accent_green'], -1)
    text_centered(img, 'Gesture Guide', 28, 0.8, THEME['text_title'], 2)
    col1_x = 18
    y0     = 52
    mouse_gestures = [
        ('MOVE',        'Index finger up only',           THEME['gest_move']),
        ('LEFT CLICK',  'Thumb+Middle pinch (<0.5s)',      THEME['gest_left_click']),
        ('DRAG',        'Thumb+Middle pinch (hold 0.5s)', THEME['gest_drag']),
        ('RIGHT CLICK', 'Thumb+Ring pinch',                THEME['gest_right_click']),
        ('SCROLL UP',   'Index+Middle+Ring up',            THEME['gest_scroll_up']),
        ('SCROLL DOWN', 'All 4 fingers curled',            THEME['gest_scroll_down']),
    ]
    cv2.putText(img, 'MOUSE CONTROLS', (col1_x, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.42, THEME['text_dim'], 1)
    for i, (name, desc, col) in enumerate(mouse_gestures):
        cy = y0 + 18 + i * 28
        cv2.circle(img, (col1_x + 5, cy - 4), 4, col, -1)
        cv2.putText(img, name, (col1_x + 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(img, desc, (col1_x + 14, cy + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.33, THEME['text_secondary'], 1)
    mid_x = w // 2 - 10
    cv2.line(img, (mid_x, 46), (mid_x, h - 55), THEME['divider'], 1)
    col2_x = w // 2
    game_gestures = [
        ('GAME OPT 1', 'Fist + 1 finger (hold 5s) = MOUSE',  THEME['accent_cyan']),
        ('GAME OPT 2', 'Fist + 2 fingers (hold 5s) = SUBWAY', THEME['accent_green']),
        ('GAME OPT 3', 'Fist + 3 fingers (hold 5s) = RACING', THEME['accent_orange']),
        ('GAME OPT 4', 'Fist + 4 fingers (hold 5s) = free',   THEME['accent_pink']),
    ]
    cv2.putText(img, 'GAME OPTIONS', (col2_x, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.42, THEME['text_dim'], 1)
    for i, (name, desc, col) in enumerate(game_gestures):
        cy = y0 + 18 + i * 28
        cv2.circle(img, (col2_x + 5, cy - 4), 4, col, -1)
        cv2.putText(img, name, (col2_x + 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(img, desc, (col2_x + 14, cy + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.33, THEME['text_secondary'], 1)
    sep_y = y0 + 18 + len(game_gestures) * 28 + 8
    cv2.line(img, (col2_x, sep_y), (w - 10, sep_y), THEME['divider'], 1)
    meta_gestures = [
        ('START/RESUME', 'Both peace signs (3s)', THEME['accent_green']),
        ('PAUSE',        'Open palm (3s)',         THEME['accent_yellow']),
        ('RECALIBRATE',  'Shaka (3s) → zone',     THEME['accent_purple']),
        ('GUIDE',        'Metal sign (idx+pinky)', THEME['accent_cyan']),
        ('CLOSE APP',    'Both fists (3s)',        THEME['accent_red']),
    ]
    cv2.putText(img, 'META (OPT1 only)', (col2_x, sep_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, THEME['text_dim'], 1)
    for i, (name, desc, col) in enumerate(meta_gestures):
        cy = sep_y + 30 + i * 24
        cv2.circle(img, (col2_x + 5, cy - 4), 3, col, -1)
        cv2.putText(img, f'{name}: {desc}', (col2_x + 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)
    progress_bar(img, 10, h - 22, w - 20, 10, elapsed / GUIDE_DURATION, THEME['accent_green'])
    if skip_frac > 0:
        hold_arc(img, w - 45, 45, 28, skip_frac, THEME['accent_green'])
        cv2.putText(img, 'SKIP', (w - 65, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_green'], 1)
    draw_top_pill(img, 'GESTURE GUIDE', THEME['accent_green'])


def draw_distance_ui(img, dist, has_hand, hold_frac=0.0):
    h, w = img.shape[:2]
    filled_rect(img, 0, 0, w, 110, THEME['bg_dark'], 0.9)
    if not has_hand:
        text_centered(img, 'Show your hand to the camera', 55, 0.8, THEME['text_secondary'], 1)
        draw_top_pill(img, '  DISTANCE CHECK', THEME['accent_cyan'])
        return
    too_close = dist > TARGET_DIST + DIST_TOL
    too_far   = dist < TARGET_DIST - DIST_TOL
    if too_close:
        msg, col = 'Move FARTHER from camera',    THEME['bar_fill_warn']
    elif too_far:
        msg, col = 'Move CLOSER to camera',       THEME['bar_fill_far']
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
        text_centered(img, f'Hold still... {hold_frac * DIST_OK_HOLD:.1f}s', 100, 0.42, THEME['accent_green'], 1)
    draw_top_pill(img, '  DISTANCE CHECK', THEME['accent_cyan'])


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
    BAR_BOT     = 70
    ARC_BOT     = BAR_BOT + 52
    CONTENT_Y   = ARC_BOT + 4
    LEGEND_SP   = 19
    LEGEND_LINES= 7
    LEGEND_BOT  = h - 6
    LEGEND_TOP  = LEGEND_BOT - (LEGEND_LINES - 1) * LEGEND_SP

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
        hint  = '>> DRAG' if pct >= 1.0 else 'release=CLICK  hold=DRAG'
        cv2.putText(img, hint, (w - 180, BAR_BOT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, THEME['text_secondary'], 1)
    elif gesture == 'right_click': label, col = 'RIGHT CLICK',   GESTURE_COLOR['right_click']
    elif gesture == 'move':        label, col = 'MOVE',           GESTURE_COLOR['move']
    elif gesture == 'scroll_up':   label, col = 'SCROLL UP  ^',  GESTURE_COLOR['scroll_up']
    elif gesture == 'scroll_down': label, col = 'SCROLL DOWN v', GESTURE_COLOR['scroll_down']
    else:                          label, col = 'IDLE',           GESTURE_COLOR['idle']

    full_label = f'Gesture: {label}'
    (_, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    text_y = (BAR_BOT + th) // 2
    cv2.putText(img, full_label, (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)

    filled_rect(img, 0, BAR_BOT + 1, w, ARC_BOT, THEME['bg_panel'], 0.82)
    cv2.line(img, (0, ARC_BOT), (w, ARC_BOT), THEME['border'], 2)

    ARC_CY = BAR_BOT + 1 + (ARC_BOT - BAR_BOT - 1) // 2
    ARC_R  = 14
    arc_items = [
        ('PAUSE',  'stop',     THEME['accent_yellow'], 'Palm 3s'),
        ('RECAL',  'recal',    THEME['accent_purple'], 'Shaka 3s'),
        ('CLOSE',  'close',    THEME['accent_red'],    'Fists 3s'),
        ('GUIDE',  'guide',    THEME['accent_cyan'],   'Metal sign'),
        (f'OPT{game_opt_number or "?"}', 'game_opt', THEME['accent_orange'], f'Fist+{game_opt_number or "?"}F 5s'),
    ]
    spacing = w // len(arc_items)
    for i, (name, key, ic, hint) in enumerate(arc_items):
        cx   = i * spacing + spacing // 2
        frac = meta_hold_fracs.get(key, 0.0)
        cv2.circle(img, (cx, ARC_CY), ARC_R, THEME['bg_dark'], -1)
        cv2.circle(img, (cx, ARC_CY), ARC_R, ic if frac > 0 else THEME['border'], 1)
        hold_arc(img, cx, ARC_CY, ARC_R - 2, frac, ic)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.27, 1)
        cv2.putText(img, name, (cx - tw // 2, ARC_CY + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.27, ic, 1, cv2.LINE_AA)
        (hw2, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.24, 1)
        cv2.putText(img, hint, (cx - hw2 // 2, ARC_BOT - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.24, THEME['text_dim'], 1, cv2.LINE_AA)

    cy = CONTENT_Y
    if game_opt_frac > 0 and game_opt_number:
        opt_colors = [THEME['accent_cyan'], THEME['accent_green'], THEME['accent_orange'], THEME['accent_pink']]
        oc = opt_colors[(game_opt_number - 1) % 4]
        filled_rect(img, 0, cy, w, cy + 18, THEME['bg_accent'], 0.9)
        progress_bar(img, 0, cy, w, 18, game_opt_frac, oc)
        secs = max(0.0, HOLD_GAME * (1.0 - game_opt_frac))
        text_centered(img, f'GAME OPT {game_opt_number}  {secs:.1f}s', cy + 13, 0.38, THEME['text_title'], 1)
        cy += 20

    if dist_drift_msg:
        filled_rect(img, 0, cy, w, cy + 18, THEME['bg_warn'], 0.88)
        cv2.putText(img, dist_drift_msg, (8, cy + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_yellow'], 1, cv2.LINE_AA)
        cy += 20

    if gesture in ('scroll_up', 'scroll_down'):
        if scroll_entry_t and not scroll_active:
            remaining = max(0.0, SCROLL_BUFFER - (time.time() - scroll_entry_t))
            filled_rect(img, 0, cy, 260, cy + 18, THEME['bg_panel'], 0.85)
            cv2.putText(img, f'Scroll locks in {remaining:.1f}s...', (8, cy + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_cyan'], 1)
        elif scroll_active:
            dir_s = 'UP  ^' if gesture == 'scroll_up' else 'DOWN  v'
            filled_rect(img, 0, cy, 200, cy + 18, THEME['bg_success'], 0.85)
            cv2.putText(img, f'Scrolling {dir_s}', (8, cy + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME['accent_green'], 2)

    legend = [
        ('Index up only      = MOVE',                     THEME['gest_move']),
        (f'Thumb+Mid <{LONG_PRESS_TIME}s   = LEFT CLICK', THEME['gest_left_click']),
        (f'Thumb+Mid >={LONG_PRESS_TIME}s  = DRAG',       THEME['gest_drag']),
        ('Thumb+Ring         = RIGHT CLICK',               THEME['gest_right_click']),
        ('Idx+Mid+Ring up    = SCROLL UP',                 THEME['gest_scroll_up']),
        ('All fingers curled = SCROLL DOWN',               THEME['gest_scroll_down']),
        ('Fist + 1-4 fingers = GAME OPT (5s)',             THEME['accent_orange']),
    ]
    for i, (txt, c) in enumerate(legend):
        y = LEGEND_TOP + i * LEGEND_SP
        if y > ARC_BOT + 10:
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)


def draw_game_mode_badge(img, mode_num, mode_name, color, game_opt_frac, game_opt_number):
    h, w = img.shape[:2]
    badge_h = 64 if (game_opt_frac > 0 and game_opt_number) else 44
    filled_rect(img, 0, 0, w, badge_h, THEME['bg_dark'], 0.92)
    cv2.rectangle(img, (0, 0), (w, 3), color, -1)
    cv2.putText(img, f'MODE {mode_num}: {mode_name}  |  Fist+N fingers (5s) to swap',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    if game_opt_frac > 0 and game_opt_number:
        opt_colors = [THEME['accent_cyan'], THEME['accent_green'], THEME['accent_orange'], THEME['accent_pink']]
        oc   = opt_colors[(game_opt_number - 1) % 4]
        secs = max(0.0, HOLD_GAME * (1.0 - game_opt_frac))
        progress_bar(img, 0, 36, w, 6, game_opt_frac, oc)
        cv2.putText(img, f'Switching to OPT{game_opt_number} in {secs:.1f}s...',
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, oc, 1, cv2.LINE_AA)


def draw_ss_hud(img, gesture, conf, fw, fh, lms, game_opt_frac, game_opt_number):
    h, w = img.shape[:2]
    draw_game_mode_badge(img, 2, 'SUBWAY SURFER', THEME['accent_green'], game_opt_frac, game_opt_number)
    SS_COLORS  = {
        'up':      (180, 255,  80),
        'down':    ( 80, 180, 255),
        'left':    (255, 180,  80),
        'right':   (255,  80, 180),
        'space':   (  0, 220, 255),
        'neutral': (200, 200, 200),
        'none':    (100, 100, 100),
    }
    SS_ACTIONS = {
        'up':      'UP — arrow up',
        'down':    'DOWN — arrow down',
        'left':    'LEFT — arrow left',
        'right':   'RIGHT — arrow right',
        'space':   'SPACE — jump',
        'neutral': 'NEUTRAL — idle',
        'none':    'none — idle',
    }
    color  = SS_COLORS.get(gesture, (200, 200, 200))
    action = SS_ACTIONS.get(gesture, gesture)
    filled_rect(img, 0, 44, w, 90, THEME['bg_panel'], 0.85)
    cv2.putText(img, f'{action}  [{conf:.2f}]', (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    if lms:
        fx, fy = int(lms[8].x * fw), int(lms[8].y * fh)
        cv2.circle(img, (fx, fy), 12, color, 2)
        cv2.circle(img, (fx, fy),  4, color, -1)
    legend = [
        ('up      = arrow UP',    (180, 255,  80)),
        ('down    = arrow DOWN',  ( 80, 180, 255)),
        ('left    = arrow LEFT',  (255, 180,  80)),
        ('right   = arrow RIGHT', (255,  80, 180)),
        ('space   = SPACE',       (  0, 220, 255)),
        ('neutral/none = idle',   (160, 160, 160)),
    ]
    for i, (txt, col) in enumerate(legend):
        cv2.putText(img, txt, (10, h - 12 - (len(legend) - 1 - i) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)


def draw_rc_hud(img, angle, steer_dir, accel, brake, fw, fh,
                lms_left, lms_right, gest_l, conf_l, gest_r, conf_r,
                game_opt_frac, game_opt_number):
    h, w = img.shape[:2]
    draw_game_mode_badge(img, 3, 'CAR RACING', THEME['accent_orange'], game_opt_frac, game_opt_number)
    gl_color = (0, 220, 80) if gest_l == 'thumbs' else (180, 180, 180)
    gr_color = (0, 220, 80) if gest_r == 'thumbs' else (180, 180, 180)
    cv2.putText(img, f'L: {gest_l} [{conf_l:.2f}]', (10,  h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gl_color, 1)
    cv2.putText(img, f'R: {gest_r} [{conf_r:.2f}]', (200, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gr_color, 1)
    filled_rect(img, 0, 44, w, 88, THEME['bg_dark'], 0.88)
    sc = {'left': (255, 180, 80), 'right': (255, 80, 180), 'none': (180, 180, 180)}[steer_dir]
    sl = {'left': 'LEFT  <', 'right': 'RIGHT  >', 'none': 'STRAIGHT'}[steer_dir]
    cv2.putText(img, f'Steer: {sl}  ({angle:+.1f}deg)', (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc, 2)
    cv2.rectangle(img, (w - 165, 48), (w - 88, 82), (0, 180, 60) if accel else (50, 50, 50), -1)
    cv2.putText(img, 'ACCEL ^', (w - 162, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.rectangle(img, (w - 82,  48), (w - 4,  82), (0,  80, 220) if brake else (50, 50, 50), -1)
    cv2.putText(img, 'BRAKE v', (w - 79,  72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if lms_left and lms_right:
        lx = int(lms_left[0].x  * fw); ly = int(lms_left[0].y  * fh)
        rx = int(lms_right[0].x * fw); ry = int(lms_right[0].y * fh)
        cv2.line(img, (lx, ly), (rx, ry), sc, 3)
        cv2.circle(img, (lx, ly), 10, (255, 180,  80), -1)
        cv2.circle(img, (rx, ry), 10, (255,  80, 180), -1)
    cx, cy, r = w // 2, h - 95, 52
    cv2.circle(img, (cx, cy), r, (50, 50, 50), 3)
    for deg in [-RC_STEER_DEADZONE, RC_STEER_DEADZONE]:
        rad = np.radians(deg)
        cv2.line(img,
                 (int(cx + (r - 8) * np.sin(rad)), int(cy - (r - 8) * np.cos(rad))),
                 (int(cx + (r + 8) * np.sin(rad)), int(cy + (r + 8) * np.cos(rad))),
                 (100, 100, 100), 2)
    rad = np.radians(np.clip(angle, -RC_STEER_MAX, RC_STEER_MAX))
    cv2.line(img, (cx, cy), (int(cx + r * np.sin(rad)), int(cy - r * np.cos(rad))), sc, 3)
    cv2.circle(img, (cx, cy), 6, sc, -1)
    cv2.putText(img, 'L', (cx - r - 20, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180,  80), 2)
    cv2.putText(img, 'R', (cx + r + 6,  cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,  80, 180), 2)
    legend = [
        ('Right hand thumbs = ACCEL',   (  0, 220,  80)),
        ('Left  hand thumbs = BRAKE',   (  0, 120, 255)),
        ('Tilt hands = STEER L / R',    (255, 180,  80)),
        ('R = recalibrate (keyboard)',   (100, 100, 100)),
    ]
    for i, (txt, col) in enumerate(legend):
        cv2.putText(img, txt, (10, h - 50 - (len(legend) - 1 - i) * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1)


def tick_game_opt(lms, lms2, now, hold_t, cur_num):
    opt_now = get_game_option(lms, lms2)
    if opt_now is not None:
        if opt_now != cur_num:
            hold_t  = now
            cur_num = opt_now
        elif hold_t is None:
            hold_t = now
        held = now - hold_t
        frac = min(held / HOLD_GAME, 1.0)
        return hold_t, cur_num, frac, (opt_now if frac >= 1.0 else None)
    return None, None, 0.0, None


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


def full_reset():
    global app_state, dist_ok_since, smooth_x, smooth_y
    global last_click_t, prev_scroll_y, scroll_entry_t, scroll_active
    global tm_pinch_start, tm_drag_active, intro_start_t, zone_start_t, guide_start_t
    global meta_hold, game_option_pending, active_game_mode
    global ss_current_zone, ss_last_key_t, ss_last_space_t, ss_space_pressed, ss_prev_row
    global rc_prev_row_left, rc_prev_row_right
    if tm_drag_active:
        pyautogui.mouseUp()
    rc_release_all()
    active_game_mode   = None
    app_state          = 'distance_check'
    dist_ok_since      = None
    smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2
    last_click_t       = 0.0
    prev_scroll_y      = None
    scroll_entry_t     = None
    scroll_active      = False
    tm_pinch_start     = None
    tm_drag_active     = False
    intro_start_t      = None
    zone_start_t       = None
    guide_start_t      = None
    meta_hold          = {k: None for k in meta_hold}
    game_option_pending = None
    ss_current_zone    = 'neutral'
    ss_last_key_t      = 0.0
    ss_last_space_t    = 0.0
    ss_space_pressed   = False
    ss_prev_row        = None
    rc_prev_row_left   = None
    rc_prev_row_right  = None


def activate_game_mode(opt):
    global active_game_mode
    global ss_current_zone, ss_last_key_t, ss_last_space_t, ss_space_pressed, ss_prev_row
    global rc_prev_row_left, rc_prev_row_right
    handle_pinch_release(time.time())
    rc_release_all()
    if opt == 1:
        active_game_mode = None
        print("[MODE] Switched to OPT1: Mouse Mode")
    elif opt == 2:
        active_game_mode = 2
        ss_current_zone   = 'neutral'
        ss_last_key_t     = 0.0
        ss_last_space_t   = 0.0
        ss_space_pressed  = False
        ss_prev_row       = None
        print("[MODE] Switched to OPT2: Subway Surfer")
        if subway_model is None:
            print("[WARN] gesture_model_subway.pt / label_encoder_subway.pkl not found.")
    elif opt == 3:
        active_game_mode  = 3
        rc_prev_row_left  = None
        rc_prev_row_right = None
        print("[MODE] Switched to OPT3: Car Racing")
        if racing_model is None:
            print("[WARN] gesture_model_racing.pt / label_encoder_racing.pkl not found.")
    elif opt == 4:
        active_game_mode = 4
        print("[MODE] Switched to OPT4: free slot")


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
    lms      = result.hand_landmarks[0] if has_hand else None
    lms2     = result.hand_landmarks[1] if (has_hand and len(result.hand_landmarks) > 1) else None
    display  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fh, fw   = display.shape[:2]
    now      = time.time()

    if app_state == 'intro':
        elapsed = now - intro_start_t
        skip_frac = 0.0
        if elapsed >= SKIP_LOCKOUT and lms and is_thumbs_up(lms):
            if thumb_hold_t is None: thumb_hold_t = now
            skip_frac = min((now - thumb_hold_t) / 1.0, 1.0)
            if skip_frac >= 1.0: elapsed = INTRO_DURATION
        else:
            if not (lms and is_thumbs_up(lms)): thumb_hold_t = None
        if lms: draw_hand(display, lms)
        draw_intro(display, elapsed, skip_frac)
        if elapsed >= INTRO_DURATION:
            app_state = 'zone_intro'; zone_intro_start_t = now; thumb_hold_t = None

    elif app_state == 'zone_intro':
        elapsed = now - zone_intro_start_t
        skip_frac = 0.0
        if elapsed >= SKIP_LOCKOUT and lms and is_thumbs_up(lms):
            if thumb_hold_t is None: thumb_hold_t = now
            skip_frac = min((now - thumb_hold_t) / 1.0, 1.0)
            if skip_frac >= 1.0: elapsed = ZONE_INTRO_DURATION
        else:
            if not (lms and is_thumbs_up(lms)): thumb_hold_t = None
        if lms: draw_hand(display, lms)
        draw_zone_intro(display, elapsed, skip_frac)
        if elapsed >= ZONE_INTRO_DURATION:
            app_state = 'zone_pick'; zone_start_t = now; thumb_hold_t = None

    elif app_state == 'zone_pick':
        elapsed = now - zone_start_t
        detected_zone = None
        if lms:
            n = count_fingers_up(lms)
            if n == 1:   detected_zone = 'small'
            elif n == 2: detected_zone = 'medium'
            elif n >= 3: detected_zone = 'large'
        confirm_frac = 0.0; confirm_secs_left = 0.0
        if detected_zone:
            if detected_zone not in zone_finger_hold:
                zone_finger_hold = {detected_zone: now}
            held = now - zone_finger_hold[detected_zone]
            confirm_frac      = min(held / ZONE_CONFIRM_TIME, 1.0)
            confirm_secs_left = max(0.0, ZONE_CONFIRM_TIME - held)
            if confirm_frac >= 1.0:
                chosen_zone = detected_zone
                set_zone_from_choice(chosen_zone)
                app_state = 'guide'; guide_start_t = now
                thumb_hold_t = None; zone_finger_hold = {}
        else:
            zone_finger_hold = {}
        skip_frac = 0.0
        if elapsed >= SKIP_LOCKOUT and lms and is_thumbs_up(lms):
            if thumb_hold_t is None: thumb_hold_t = now
            skip_frac = min((now - thumb_hold_t) / 1.0, 1.0)
            if skip_frac >= 1.0:
                set_zone_from_choice(chosen_zone)
                app_state = 'guide'; guide_start_t = now; thumb_hold_t = None
        else:
            if not (lms and is_thumbs_up(lms)): thumb_hold_t = None
        if elapsed >= ZONE_DURATION and app_state == 'zone_pick':
            set_zone_from_choice(chosen_zone)
            app_state = 'guide'; guide_start_t = now; thumb_hold_t = None
        if lms: draw_hand(display, lms)
        draw_zone_pick(display, elapsed, detected_zone or chosen_zone,
                       skip_frac, confirm_frac, confirm_secs_left)

    elif app_state == 'guide':
        elapsed = now - guide_start_t
        skip_frac = 0.0
        if elapsed >= SKIP_LOCKOUT and lms and is_thumbs_up(lms):
            if thumb_hold_t is None: thumb_hold_t = now
            skip_frac = min((now - thumb_hold_t) / 1.0, 1.0)
            if skip_frac >= 1.0: elapsed = GUIDE_DURATION
        else:
            if not (lms and is_thumbs_up(lms)): thumb_hold_t = None
        if lms: draw_hand(display, lms)
        draw_guide(display, elapsed, skip_frac)
        if elapsed >= GUIDE_DURATION:
            app_state = 'distance_check'; dist_ok_since = None; thumb_hold_t = None

    elif app_state == 'distance_check':
        dist     = hand_size(lms) if lms else 0.0
        in_range = lms and abs(dist - TARGET_DIST) <= DIST_TOL
        if in_range:
            if dist_ok_since is None: dist_ok_since = now
        else:
            dist_ok_since = None
        hold_frac = 0.0
        if dist_ok_since:
            hold_frac = min((now - dist_ok_since) / DIST_OK_HOLD, 1.0)
            if hold_frac >= 1.0:
                app_state = 'running'; dist_ok_since = None
        if lms: draw_hand(display, lms)
        draw_distance_ui(display, dist, has_hand, hold_frac)

    elif app_state == 'running':
        game_opt_hold_t, game_opt_number, game_opt_frac, triggered_opt = \
            tick_game_opt(lms, lms2, now, game_opt_hold_t, game_opt_number)

        if triggered_opt is not None:
            print(f'[GAME OPTION {triggered_opt} SELECTED]')
            activate_game_mode(triggered_opt)
            game_opt_hold_t = None
            game_opt_number = None
            game_opt_frac   = 0.0

        if active_game_mode is None:
            gesture = 'idle'
            if range_min_x is not None:
                bx1 = int(range_min_x * fw); bx2 = int(range_max_x * fw)
                by1 = int(range_min_y * fh); by2 = int(range_max_y * fh)
                cv2.rectangle(display, (bx1, by1), (bx2, by2), THEME['accent_green'], 1)

            meta_hold_fracs = {k: 0.0 for k in ('start', 'stop', 'recal', 'close', 'guide', 'game_opt')}
            triggered_meta  = None

            if lms:
                if is_open_palm(lms):
                    if meta_hold['stop'] is None: meta_hold['stop'] = now
                    meta_hold_fracs['stop'] = min((now - meta_hold['stop']) / HOLD_META, 1.0)
                    if meta_hold_fracs['stop'] >= 1.0:
                        triggered_meta = 'stop'
                else:
                    meta_hold['stop'] = None

                if is_shaka(lms):
                    if meta_hold['recal'] is None: meta_hold['recal'] = now
                    meta_hold_fracs['recal'] = min((now - meta_hold['recal']) / HOLD_META, 1.0)
                    if meta_hold_fracs['recal'] >= 1.0:
                        triggered_meta = 'recal'
                else:
                    meta_hold['recal'] = None

                if is_metal_sign(lms):
                    if meta_hold['guide'] is None: meta_hold['guide'] = now
                    meta_hold_fracs['guide'] = min((now - meta_hold['guide']) / HOLD_META, 1.0)
                    if meta_hold_fracs['guide'] >= 1.0:
                        triggered_meta = 'guide'
                else:
                    meta_hold['guide'] = None

            both_peace = lms and lms2 and is_peace_sign(lms) and is_peace_sign(lms2)
            if both_peace:
                if meta_hold['start'] is None: meta_hold['start'] = now
                meta_hold_fracs['start'] = min((now - meta_hold['start']) / HOLD_META, 1.0)
            else:
                meta_hold['start'] = None

            game_opt_now_mouse = get_game_option(lms, lms2)
            both_fists = lms and lms2 and is_fist(lms) and is_fist(lms2)
            if both_fists and game_opt_now_mouse is None:
                if meta_hold['close'] is None: meta_hold['close'] = now
                meta_hold_fracs['close'] = min((now - meta_hold['close']) / HOLD_CLOSE, 1.0)
                if meta_hold_fracs['close'] >= 1.0:
                    triggered_meta = 'close'
            else:
                meta_hold['close'] = None

            meta_hold_fracs['game_opt'] = game_opt_frac

            if triggered_meta == 'stop':
                handle_pinch_release(now)
                app_state = 'stopped'
                for k in meta_hold: meta_hold[k] = None
            elif triggered_meta == 'recal':
                handle_pinch_release(now)
                app_state = 'zone_pick'; zone_start_t = now
                for k in meta_hold: meta_hold[k] = None
            elif triggered_meta == 'guide':
                handle_pinch_release(now)
                app_state = 'guide'; guide_start_t = now
                for k in meta_hold: meta_hold[k] = None
            elif triggered_meta == 'close':
                handle_pinch_release(now)
                break

            any_meta_active = any(v is not None for v in meta_hold.values()) or game_opt_frac > 0

            if lms and triggered_meta is None and not any_meta_active:
                gesture = get_gesture_mouse(lms)
                dist    = hand_size(lms)
                dist_ok = abs(dist - TARGET_DIST) <= DIST_TOL * 2

                dist_drift_msg = None
                drift = dist - TARGET_DIST
                if drift > DIST_TOL * 3:
                    dist_drift_msg = 'WARNING: Too close - move hand further away'
                elif drift > DIST_TOL * 2:
                    dist_drift_msg = 'Drifted: slightly too close to camera'
                elif drift < -DIST_TOL * 3:
                    dist_drift_msg = 'WARNING: Too far - bring hand closer'
                elif drift < -DIST_TOL * 2:
                    dist_drift_msg = 'Drifted: slightly too far from camera'

                if not dist_drift_msg:
                    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness = float(np.mean(gray))
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
                                meta_hold_fracs, game_opt_frac, game_opt_number)
            else:
                handle_pinch_release(now)
                scroll_entry_t = None; scroll_active = False; prev_scroll_y = None
                if lms:  draw_hand(display, lms)
                if lms2: draw_hand(display, lms2)
                draw_running_ui(display, 'idle', True, None,
                                None, False, None, False,
                                meta_hold_fracs, game_opt_frac, game_opt_number)

        elif active_game_mode == 2:
            if subway_model is None:
                text_centered(display,
                    'Missing: gesture_model_subway.pt + label_encoder_subway.pkl',
                    fh // 2, 0.5, THEME['accent_red'], 2)
            else:
                gesture_ss = 'none'; conf_ss = 0.0
                if lms:
                    gesture_ss, conf_ss, ss_prev_row = run_nn(
                        lms, ss_prev_row, subway_model, subway_le, SS_CONFIDENCE_THRESH)
                    if gesture_ss == 'space':
                        if not ss_space_pressed and (now - ss_last_space_t) > SS_SPACE_COOLDOWN:
                            pyautogui.press('space')
                            ss_space_pressed = True
                            ss_last_space_t  = now
                            print("[SUBWAY] SPACE")
                        ss_current_zone = 'neutral'
                    else:
                        ss_space_pressed = False
                        if gesture_ss in ('up', 'down', 'left', 'right'):
                            if gesture_ss != ss_current_zone and (now - ss_last_key_t) > SS_KEY_COOLDOWN:
                                pyautogui.press(gesture_ss)
                                ss_last_key_t   = now
                                ss_current_zone = gesture_ss
                                print(f"[SUBWAY] {gesture_ss.upper()}")
                        else:
                            ss_current_zone = 'neutral'
                else:
                    ss_prev_row = None
                if lms:  draw_hand(display, lms)
                if lms2: draw_hand(display, lms2)
                draw_ss_hud(display, gesture_ss, conf_ss, fw, fh, lms, game_opt_frac, game_opt_number)

        elif active_game_mode == 3:
            if racing_model is None:
                text_centered(display,
                    'Missing: gesture_model_racing.pt + label_encoder_racing.pkl',
                    fh // 2, 0.5, THEME['accent_red'], 2)
            else:
                lms_left = lms_right = None
                if result and result.hand_landmarks:
                    lms_left, lms_right = split_hands(result)
                if lms_left  is None: rc_prev_row_left  = None
                if lms_right is None: rc_prev_row_right = None

                desired   = set()
                angle     = 0.0
                steer_dir = 'none'
                accel     = False
                brake     = False
                gest_l    = 'none'
                gest_r    = 'none'
                conf_l    = 0.0
                conf_r    = 0.0

                if lms_left:
                    gest_l, conf_l, rc_prev_row_left  = run_nn(
                        lms_left, rc_prev_row_left, racing_model, racing_le, RC_CONF_THRESH)
                if lms_right:
                    gest_r, conf_r, rc_prev_row_right = run_nn(
                        lms_right, rc_prev_row_right, racing_model, racing_le, RC_CONF_THRESH)

                accel = gest_r == 'thumbs'
                brake = gest_l == 'thumbs'

                if lms_left and lms_right:
                    angle = get_steer_angle(lms_left, lms_right)
                    if angle < -RC_STEER_DEADZONE:   steer_dir = 'left'
                    elif angle > RC_STEER_DEADZONE:  steer_dir = 'right'

                if steer_dir == 'left':  desired.add('left')
                if steer_dir == 'right': desired.add('right')
                if accel:                desired.add('up')
                if brake:                desired.add('down')
                rc_set_held_keys(desired)

                if lms_left:  draw_hand(display, lms_left,  (255, 180,  80))
                if lms_right: draw_hand(display, lms_right, (255,  80, 180))

                if lms_left is None or lms_right is None:
                    missing = 'LEFT' if lms_left is None else 'RIGHT'
                    cv2.rectangle(display, (0, 44), (fw, 90), (40, 20, 20), -1)
                    cv2.putText(display, f'Lost {missing} hand!',
                                (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

                draw_rc_hud(display, angle, steer_dir, accel, brake, fw, fh,
                            lms_left, lms_right,
                            gest_l, conf_l, gest_r, conf_r,
                            game_opt_frac, game_opt_number)

        elif active_game_mode == 4:
            draw_game_mode_badge(display, 4, 'OPT4 — not yet assigned',
                                 THEME['accent_pink'], game_opt_frac, game_opt_number)
            text_centered(display,
                          'Game Option 4 is a free slot — add your logic here.',
                          fh // 2, 0.6, THEME['text_secondary'], 1)
            if lms:  draw_hand(display, lms)
            if lms2: draw_hand(display, lms2)

    elif app_state == 'stopped':
        filled_rect(display, 0, 0, fw, fh, THEME['bg_dark'], 0.88)
        cv2.rectangle(display, (0, 0), (fw, 4), THEME['accent_red'], -1)
        text_centered(display, 'PAUSED', fh // 2 - 40, 1.2, THEME['accent_red'], 3)
        text_centered(display, 'Both Peace Signs (hold 3s) to resume', fh // 2 + 10, 0.55, THEME['text_secondary'], 1)
        text_centered(display, 'Both Fists (hold 3s) to close', fh // 2 + 36, 0.5, THEME['text_dim'], 1)

        resume_frac = 0.0
        close_frac  = 0.0
        both_peace_pause = lms and lms2 and is_peace_sign(lms) and is_peace_sign(lms2)
        both_fists_pause = lms and lms2 and is_fist(lms) and is_fist(lms2)

        if both_peace_pause:
            if thumb_hold_t is None: thumb_hold_t = now
            resume_frac = min((now - thumb_hold_t) / HOLD_META, 1.0)
            hold_arc(display, fw // 2, fh // 2 + 80, 30, resume_frac, THEME['accent_green'])
            if resume_frac >= 1.0:
                app_state = 'running'; thumb_hold_t = None
                for k in meta_hold: meta_hold[k] = None
        else:
            thumb_hold_t = None

        if both_fists_pause:
            if meta_hold['close'] is None: meta_hold['close'] = now
            close_frac = min((now - meta_hold['close']) / HOLD_CLOSE, 1.0)
            hold_arc(display, fw // 2 + 70, fh // 2 + 80, 30, close_frac, THEME['accent_red'])
            if close_frac >= 1.0:
                break
        else:
            meta_hold['close'] = None

        if lms:  draw_hand(display, lms)
        if lms2: draw_hand(display, lms2)
        draw_top_pill(display, '  PAUSED', THEME['accent_red'])

    output = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    cv2.imshow('HandMouse', output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in (ord('r'), ord('R')):
        full_reset()

handle_pinch_release(time.time())
rc_release_all()
_stop_thread = True
det_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
detector.close()