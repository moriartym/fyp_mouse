import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import time
import threading
import platform

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# ─── Tuning Constants ────────────────────────────────────────────────────────
SMOOTHING        = 0.3
PINCH_THRESH     = 0.03
CLICK_COOLDOWN   = 0.5
SCROLL_SPEED     = 3
LONG_PRESS_TIME  = 0.5
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ─── Distance Calibration ────────────────────────────────────────────────────
TARGET_DIST    = 0.18
DIST_TOL       = 0.03
DIST_OK_HOLD   = 1.0
CALIB_DURATION = 5.0
range_min_x = range_max_x = None
range_min_y = range_max_y = None

# ─── App States ──────────────────────────────────────────────────────────────
app_state     = 'distance_check'
dist_ok_since = None
calib_start_t = None
calib_pts_x   = []
calib_pts_y   = []
calib_trail   = []
calib_done_t  = None

# ─── Mode 
current_mode       = 'mouse'
mode_switch_done_t = None

# ─── Peace-sign Mode Switch 
PEACE_SWITCH_HOLD  = 1.5
peace_switch_timer = None

# ─── Mouse Gesture State 
last_click_t   = 0.0
prev_scroll_y  = None
scroll_entry_t = None
SCROLL_BUFFER  = 0
scroll_active  = False
tm_pinch_start = None
tm_drag_active = False

# ─── Game State 
#
#  ╔══════════════════════════════════════════════════════════════════╗
#  ║  HOW TO CONFIGURE GESTURES — QUICK REFERENCE                   ║
#  ╠══════════════════════════════════════════════════════════════════╣
#  ║                                                                  ║
#  ║  GAME MODE 1  (gesture-based)                                    ║
#  ║  ───────────────────────────────────────────────                 ║
#  ║  1. get_game_gesture(lms)        ← ~line 220                    ║
#  ║     Edit finger conditions here to change hand-shape→action.    ║
#  ║     Returns: 'neutral' 'slide' 'left' 'right' 'jump'            ║
#  ║              'peace' (= jump or mode-switch)  'idle'            ║
#  ║                                                                  ║
#  ║  2. GAME_KEY_MAP                 ← ~line 275                    ║
#  ║     Maps action name → keyboard key string.                     ║
#  ║     Change values here to remap keys.                           ║
#  ║                                                                  ║
#  ║  3. Priority block in game1 loop ← ~line 850                   ║
#  ║     When both hands are present, controls which hand wins.      ║
#  ║                                                                  ║
#  ║  GAME MODE 2  (zone-based, from original Code 1)                 ║
#  ║  ───────────────────────────────────────────────                 ║
#  ║  1. NEUTRAL_ZONE_SIZE            ← ~line 285                   ║
#  ║     Fraction of calibrated range that is the dead-centre.       ║
#  ║                                                                  ║
#  ║  2. get_zone(tip_x, tip_y)       ← ~line 300                   ║
#  ║     Spatial mapping to zone string. Adjust thresholds here.     ║
#  ║                                                                  ║
#  ║  3. Gesture guards in game2 loop ← ~line 890                   ║
#  ║     Controls which hand shapes activate zone-nav vs             ║
#  ║     jump/slide overrides.                                        ║
#  ║                                                                  ║
#  ║  MODE-SWITCH PRESET                                              ║
#  ║    ✌✌  Peace sign on BOTH hands, held 1.5 s                    ║
#  ║    Cycles:  mouse → game1 → game2 → mouse → …                  ║
#  ╚══════════════════════════════════════════════════════════════════╝
#
GAME_COOLDOWN      = 0.4
last_game_action_t = 0.0
last_game_action   = ''

# ── Key map 
GAME_KEY_LEFT  = 'left'
GAME_KEY_RIGHT = 'right'
GAME_KEY_JUMP  = 'up'
GAME_KEY_SLIDE = 'down'

GAME_KEY_MAP = {
    'left':  GAME_KEY_LEFT,
    'right': GAME_KEY_RIGHT,
    'jump':  GAME_KEY_JUMP,
    'slide': GAME_KEY_SLIDE,
    'up':    GAME_KEY_JUMP,    
    'down':  GAME_KEY_SLIDE,   
}

# ── Zone-mode constants 
NEUTRAL_ZONE_SIZE = 0.28    # dead-centre fraction of calibrated range
ZONE_KEY_COOLDOWN = 0.35    # seconds between zone key presses

zone_current_zone = 'neutral'
zone_last_key_t   = 0.0

# ─── Threading 
_latest_frame      = None
_latest_result     = None
_frame_lock        = threading.Lock()
_result_lock       = threading.Lock()
_detection_running = False
_stop_thread       = False
smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2

WINDOW_NAME = 'Gesture Controller'


def set_window_always_on_top():
    system = platform.system()
    if system == 'Windows':
        try:
            import ctypes
            HWND_TOPMOST   = -1
            SWP_NOMOVE     = 0x0002
            SWP_NOSIZE     = 0x0001
            SWP_SHOWWINDOW = 0x0040
            hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_NAME)
            if hwnd:
                ctypes.windll.user32.SetWindowPos(
                    hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
                )
                return True
        except Exception as e:
            print(f"[WARN] Could not set always-on-top (Windows): {e}")
    elif system == 'Darwin':
        try:
            import subprocess
            script = 'tell application "System Events" to set frontmost of every process whose name is "Python" to true'
            subprocess.Popen(['osascript', '-e', script],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            print(f"[WARN] Could not set always-on-top (macOS): {e}")
    else:
        try:
            import subprocess
            subprocess.Popen(['wmctrl', '-r', WINDOW_NAME, '-b', 'add,above'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            print(f"[WARN] Could not set always-on-top (Linux — install wmctrl): {e}")
    return False

_aot_applied = False


# Helper Utilities

def hand_size(lms):
    return np.hypot(lms[0].x - lms[9].x, lms[0].y - lms[9].y)

def tip_dist(lms, a, b):
    return np.hypot(lms[a].x - lms[b].x, lms[a].y - lms[b].y)

def is_finger_extended(lms, tip, pip):
    tip_d = np.hypot(lms[tip].x - lms[0].x, lms[tip].y - lms[0].y)
    pip_d = np.hypot(lms[pip].x - lms[0].x, lms[pip].y - lms[0].y)
    return tip_d > pip_d

def is_thumb_extended(lms):
    """
    Thumb extended = tip clearly far from index-finger MCP (landmark 5).
    More reliable than wrist-distance for thumb detection.
    """
    return np.hypot(lms[4].x - lms[5].x, lms[4].y - lms[5].y) > 0.08

def fingers_extended(lms):
    return {
        'thumb':  is_thumb_extended(lms),
        'index':  is_finger_extended(lms, 8,  6),
        'middle': is_finger_extended(lms, 12, 10),
        'ring':   is_finger_extended(lms, 16, 14),
        'pinky':  is_finger_extended(lms, 20, 18),
    }

def wrist_x(lms):
    return lms[0].x

def map_cursor(tip_x, tip_y):
    span_x = range_max_x - range_min_x
    span_y = range_max_y - range_min_y
    rx = 0.5 if span_x < 0.01 else np.clip((tip_x - range_min_x) / span_x, 0, 1)
    ry = 0.5 if span_y < 0.01 else np.clip((tip_y - range_min_y) / span_y, 0, 1)
    return rx * SCREEN_W, ry * SCREEN_H

def draw_hand(img, lms, color_override=None):
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    bone_color = color_override if color_override else (0, 200, 100)
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], bone_color, 2)
    for pt in pts:
        cv2.circle(img, pt, 5, (255, 255, 255), -1)
        cv2.circle(img, pt, 5, bone_color, 2)

def handle_pinch_release(now):
    global tm_pinch_start, tm_drag_active, last_click_t
    if tm_pinch_start is None:
        return
    held = now - tm_pinch_start
    if tm_drag_active:
        pyautogui.mouseUp()
        tm_drag_active = False
        print("[DRAG END → DROP]")
    elif held < LONG_PRESS_TIME and (now - last_click_t > CLICK_COOLDOWN):
        pyautogui.click()
        last_click_t = now
        print("[LEFT CLICK]")
    tm_pinch_start = None


# Dual-Hand Detection

def get_sorted_hands(result):
    if not result or not result.hand_landmarks:
        return None, None
    hands = result.hand_landmarks
    if len(hands) == 1:
        lms = hands[0]
        return (lms, None) if wrist_x(lms) < 0.5 else (None, lms)
    sorted_hands = sorted(hands, key=lambda h: wrist_x(h))
    return sorted_hands[0], sorted_hands[1]


# Peace Sign Helper

def is_peace_sign(lms):
    """Index + middle extended, ring + pinky curled. Thumb ignored."""
    f = fingers_extended(lms)
    return f['index'] and f['middle'] and not f['ring'] and not f['pinky']


# Mouse Gestures

def get_mouse_gesture(lms):
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

def get_game_gesture(lms):
    f = fingers_extended(lms)

    index_up  = f['index']
    middle_up = f['middle']
    ring_up   = f['ring']
    pinky_up  = f['pinky']
    thumb_out = f['thumb']


    if index_up and middle_up and not ring_up and not pinky_up:
        return 'peace'

    # neutral
    if index_up and middle_up and ring_up and pinky_up:
        return 'neutral'

    # thumb out + pinky up, index + middle + ring ALL curled
    if thumb_out and pinky_up and not index_up and not middle_up and not ring_up:
        return 'slide'

    # Index only → left / right
    # middle/ring/pinky must be curled; stricter 0.06 horizontal threshold
    if index_up and not middle_up and not ring_up and not pinky_up:
        dx = lms[8].x - lms[0].x   # positive = tip right of wrist
        if dx < -0.06:
            return 'left'
        elif dx > 0.06:
            return 'right'
        return 'idle'   

    return 'idle'

# mode 2
def get_zone(tip_x, tip_y):
    span_x = range_max_x - range_min_x
    span_y = range_max_y - range_min_y
    rx = np.clip((tip_x - range_min_x) / span_x, 0, 1) if span_x > 0.01 else 0.5
    ry = np.clip((tip_y - range_min_y) / span_y, 0, 1) if span_y > 0.01 else 0.5

    half = NEUTRAL_ZONE_SIZE / 2
    if (0.5 - half) <= rx <= (0.5 + half) and (0.5 - half) <= ry <= (0.5 + half):
        return 'neutral'
    dx = rx - 0.5
    dy = ry - 0.5
    if abs(dx) >= abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'

def draw_zone_overlay(img, fw, fh, active_zone=None):
    if range_min_x is None:
        return
    ZONE_COLOR = {
        'up':    (180, 255,  80),
        'down':  ( 80, 180, 255),
        'left':  (255, 180,  80),
        'right': (255,  80, 180),
    }
    bx1 = int(range_min_x * fw);  bx2 = int(range_max_x * fw)
    by1 = int(range_min_y * fh);  by2 = int(range_max_y * fh)
    rw  = bx2 - bx1;              rh  = by2 - by1

    half = NEUTRAL_ZONE_SIZE / 2
    nxl  = bx1 + int((0.5 - half) * rw)
    nxr  = bx1 + int((0.5 + half) * rw)
    nyt  = by1 + int((0.5 - half) * rh)
    nyb  = by1 + int((0.5 + half) * rh)

    zone_rects = {
        'up':    ((bx1, by1), (bx2, nyt)),
        'down':  ((bx1, nyb), (bx2, by2)),
        'left':  ((bx1, nyt), (nxl, nyb)),
        'right': ((nxr, nyt), (bx2, nyb)),
    }
    overlay = img.copy()
    for zname, (p1, p2) in zone_rects.items():
        cv2.rectangle(overlay, p1, p2, ZONE_COLOR[zname], -1)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    for zname, (p1, p2) in zone_rects.items():
        thick = 2 if zname == active_zone else 1
        cv2.rectangle(img, p1, p2, ZONE_COLOR[zname], thick)
    mid_x = (bx1 + bx2) // 2;  mid_y = (by1 + by2) // 2
    cv2.putText(img, "UP",    (mid_x - 15, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ZONE_COLOR['up'],    2)
    cv2.putText(img, "DOWN",  (mid_x - 25, by2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ZONE_COLOR['down'],  2)
    cv2.putText(img, "LEFT",  (bx1 +  5,  mid_y),     cv2.FONT_HERSHEY_SIMPLEX, 0.55, ZONE_COLOR['left'],  2)
    cv2.putText(img, "RIGHT", (bx2 - 65,  mid_y),     cv2.FONT_HERSHEY_SIMPLEX, 0.55, ZONE_COLOR['right'], 2)
    cv2.rectangle(img, (nxl, nyt), (nxr, nyb), (255, 255, 255), 2)
    cv2.putText(img, "NEUTRAL", (nxl + 4, (nyt + nyb) // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def fire_game_action(action):
    """Game mode 1 action with cooldown."""
    global last_game_action_t, last_game_action
    now = time.time()
    if action in ('idle', 'neutral', 'peace'):
        return
    if now - last_game_action_t < GAME_COOLDOWN:
        return
    if action == last_game_action and now - last_game_action_t < GAME_COOLDOWN * 2:
        return
    key = GAME_KEY_MAP.get(action)
    if key:
        pyautogui.press(key)
        last_game_action_t = now
        last_game_action   = action
        print(f"[GAME1] {action.upper()} → '{key}'")

def fire_zone_action(zone):
    """Game mode 2 zone-based action with its own cooldown."""
    global zone_current_zone, zone_last_key_t, last_game_action_t, last_game_action
    now = time.time()
    if zone == 'neutral':
        zone_current_zone = 'neutral'
        return
    if zone != zone_current_zone:
        if now - zone_last_key_t > ZONE_KEY_COOLDOWN:
            key = GAME_KEY_MAP.get(zone)
            if key:
                pyautogui.press(key)
                zone_last_key_t    = now
                zone_current_zone  = zone
                last_game_action_t = now
                last_game_action   = zone
                print(f"[GAME2 ZONE] {zone.upper()} → '{key}'")

# ui draw
GESTURE_COLOR = {
    'move':        (255, 200,   0),
    'left_click':  (  0, 200, 255),
    'right_click': (255,  80, 200),
    'drag':        (  0, 120, 255),
    'scroll_up':   (180, 255,  80),
    'scroll_down': ( 80, 180, 255),
    'idle':        (180, 180, 180),
    'neutral':     (200, 200, 200),
    'navigate':    (200, 200, 200),
    'left':        ( 80, 200, 255),
    'right':       (255, 200,  80),
    'jump':        ( 80, 255, 150),
    'up':          ( 80, 255, 150),
    'slide':       (255,  80,  80),
    'down':        (255,  80,  80),
    'peace':       (  0, 220, 255),
}

HAND_COLOR_LEFT  = (255, 140,  50)
HAND_COLOR_RIGHT = ( 50, 180, 255)

MODE_LABEL = {
    'mouse': 'MOUSE MODE',
    'game1': 'GAME MODE 1  (Gesture)',
    'game2': 'GAME MODE 2  (Zone)',
}
MODE_COLOR = {
    'mouse': (255, 200,   0),
    'game1': ( 80, 255, 150),
    'game2': (255, 140,  50),
}

def _label_hand(img, lms, label, color, fw, fh):
    wx = int(lms[0].x * fw)
    wy = int(lms[0].y * fh)
    cv2.putText(img, label, (wx - 10, wy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_header(img, mode):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 40), (20, 20, 40), -1)
    cv2.putText(img, MODE_LABEL[mode], (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, MODE_COLOR[mode], 2)
    hint = "  |  [M] cycle  [R] recal  [ESC] quit"
    cv2.putText(img, hint, (10 + 280, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

def draw_distance_ui(img, dist, has_hand):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 80), (20, 20, 20), -1)
    if not has_hand:
        cv2.putText(img, "Show your hand to the camera", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
        return
    too_close = dist > TARGET_DIST + DIST_TOL
    too_far   = dist < TARGET_DIST - DIST_TOL
    if too_close:
        msg, color = "Move FARTHER from camera", (0, 80, 255)
    elif too_far:
        msg, color = "Move CLOSER to camera",    (0, 200, 255)
    else:
        msg, color = "Distance OK!  Hold still...", (0, 220, 80)
    cv2.putText(img, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    bar_x, bar_y, bar_w, bar_h = 10, 90, w - 20, 18
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(np.clip(dist / (TARGET_DIST * 2), 0, 1) * bar_w)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    lo = int((TARGET_DIST - DIST_TOL) / (TARGET_DIST * 2) * bar_w) + bar_x
    hi = int((TARGET_DIST + DIST_TOL) / (TARGET_DIST * 2) * bar_w) + bar_x
    cv2.rectangle(img, (lo, bar_y), (hi, bar_y + bar_h), (0, 255, 100), 2)

def draw_calibration_ui(img, elapsed, total, trail, has_hand, cx, cy):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 88), (20, 20, 20), -1)
    cv2.putText(img, "CALIBRATION  Raise index finger & sweep your arm around",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.putText(img, "Cover ALL corners of your comfortable reach area",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
    bar_x, bar_y, bar_w, bar_h = 10, 70, w - 20, 12
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    frac = min(elapsed / total, 1.0)
    fill_color = (0, int(200 + 55 * frac), int(255 * (1 - frac)))
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(frac * bar_w), bar_y + bar_h), fill_color, -1)
    remaining = max(0.0, total - elapsed)
    cdt = f"{remaining:.1f}s"
    fs, th = 3.0, 5
    (tw, tth), _ = cv2.getTextSize(cdt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    px, py = (w - tw) // 2, h // 2 + tth // 2
    cv2.putText(img, cdt, (px + 3, py + 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2)
    cv2.putText(img, cdt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (0, int(255 * frac), int(255 * (1 - frac))), th)
    if len(cx) >= 2:
        bx1 = int(min(cx) * w);  bx2 = int(max(cx) * w)
        by1 = int(min(cy) * h);  by2 = int(max(cy) * h)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 180), 2)
        cv2.putText(img, "recorded range", (bx1, max(by1 - 8, 95)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        color = (int(50 + 180 * alpha), int(220 * alpha), int(255 * alpha))
        cv2.line(img, trail[i - 1], trail[i], color, 2)
    if not has_hand:
        cv2.rectangle(img, (0, h - 40), (w, h), (0, 0, 180), -1)
        cv2.putText(img, "No hand detected — keep your hand visible!",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

def draw_calib_done_ui(img, remaining_s):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    for msg, y, scale, color in [
        ("Calibration Done!", h // 2 - 30, 1.4, (0, 255, 120)),
        (f"Starting in {remaining_s:.1f}s", h // 2 + 40, 0.9, (200, 200, 200)),
    ]:
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(img, msg, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def draw_peace_switch_progress(img, progress, next_mode):
    h, w = img.shape[:2]
    bar_w = 320
    bx = (w - bar_w) // 2
    by = h // 2 + 30
    cv2.rectangle(img, (bx - 2, by - 2), (bx + bar_w + 2, by + 26), (0, 220, 255), 2)
    fill = int(progress * bar_w)
    cv2.rectangle(img, (bx, by), (bx + fill, by + 22), (0, 220, 255), -1)
    label = f"  Hold to switch to {MODE_LABEL[next_mode]}..."
    (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(img, label, ((w - lw) // 2, by - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)

def draw_mode_switched_splash(img, new_mode):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    text  = MODE_LABEL[new_mode]
    color = MODE_COLOR[new_mode]
    sub   = "Peace both hands (1.5s) = cycle to next mode"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    (sw, _), _ = cv2.getTextSize(sub,  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, text, ((w - tw) // 2, h // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(img, sub,  ((w - sw) // 2, h // 2 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

def draw_mouse_ui(img, gesture, dist_ok, scroll_entry_t, scroll_active,
                  tm_pinch_start, tm_drag_active):
    h, w = img.shape[:2]
    draw_header(img, 'mouse')
    if tm_drag_active:
        label = "DRAG & DROP  [HOLDING]";  color = GESTURE_COLOR['drag']
    elif gesture == 'tm_pinch' and tm_pinch_start is not None:
        held  = time.time() - tm_pinch_start
        pct   = min(held / LONG_PRESS_TIME, 1.0)
        color = GESTURE_COLOR['left_click']
        label = f"PINCH  {held:.2f}s"
        cv2.rectangle(img, (0, 42), (380, 60), (40, 40, 40), -1)
        fill_w    = int(pct * 380)
        bar_color = (0, int(200 * (1 - pct)), int(255 * pct))
        cv2.rectangle(img, (0, 42), (fill_w, 60), bar_color, -1)
        hint = ">> DRAG MODE" if pct >= 1.0 else f"release=CLICK  hold {LONG_PRESS_TIME}s=DRAG"
        cv2.putText(img, hint, (10, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    elif gesture == 'right_click':
        label = "RIGHT CLICK";  color = GESTURE_COLOR['right_click']
    elif gesture == 'move':
        label = "MOVE";         color = GESTURE_COLOR['move']
    elif gesture in ('scroll_up', 'scroll_down'):
        label = gesture.upper(); color = GESTURE_COLOR[gesture]
    else:
        label = "IDLE";         color = GESTURE_COLOR['idle']
    cv2.rectangle(img, (0, 42), (w, 68), (30, 30, 30), -1)
    cv2.putText(img, f"Gesture: {label}", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    legend = [
        ("Index up               = MOVE",                    GESTURE_COLOR['move']),
        (f"Thumb+Mid <{LONG_PRESS_TIME}s  = LEFT CLICK",     GESTURE_COLOR['left_click']),
        (f"Thumb+Mid >={LONG_PRESS_TIME}s = DRAG & DROP",    GESTURE_COLOR['drag']),
        ("Thumb+Ring             = RIGHT CLICK",              GESTURE_COLOR['right_click']),
        ("Index+Mid+Ring up      = SCROLL UP",                GESTURE_COLOR['scroll_up']),
        ("All 4 fingers curled   = SCROLL DOWN",              GESTURE_COLOR['scroll_down']),
        ("Peace both hands 1.5s  = GAME MODE 1",             GESTURE_COLOR['peace']),
    ]
    for i, (text, col) in enumerate(legend):
        cv2.putText(img, text, (10, h - 12 - (len(legend) - 1 - i) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1)

def _draw_action_flash(img, action, fw, fh, y_centre=None):
    """Large centred action label that flashes on screen."""
    if not action or action in ('idle', 'neutral', 'peace', 'navigate', ''):
        return
    h, w = img.shape[:2]
    cy = y_centre if y_centre else h // 2
    lc = GESTURE_COLOR.get(action, (255, 255, 255))
    fs = 2.5
    (tw, th), _ = cv2.getTextSize(action.upper(), cv2.FONT_HERSHEY_SIMPLEX, fs, 4)
    cx = (w - tw) // 2
    cv2.putText(img, action.upper(), (cx + 3, cy + 3),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 6)
    cv2.putText(img, action.upper(), (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, fs, lc, 4)

def draw_game1_ui(img, left_g, right_g, last_action, peace_progress, next_mode):
    h, w = img.shape[:2]
    draw_header(img, 'game1')
    _draw_action_flash(img, last_action, w, h)

    for g, side, x_start in [(left_g, 'L', 0), (right_g, 'R', w - 200)]:
        if g and g != 'idle':
            gc = GESTURE_COLOR.get(g, (255, 255, 255))
            cv2.rectangle(img, (x_start, 42), (x_start + 200, 68), (30, 30, 30), -1)
            cv2.putText(img, f"{side}: {g.upper()}", (x_start + 8, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, gc, 2)

    legend = [
        ("Open palm              = NEUTRAL (no action)",  GESTURE_COLOR['neutral']),
        ("Shaka (thumb+pinky)    = SLIDE (down)",         GESTURE_COLOR['slide']),
        ("Index left             = MOVE LEFT",            GESTURE_COLOR['left']),
        ("Index right            = MOVE RIGHT",           GESTURE_COLOR['right']),
        ("Index+Middle up        = JUMP (up)",            GESTURE_COLOR['jump']),
        (f"Peace both hands 1.5s  = {MODE_LABEL[next_mode]}", GESTURE_COLOR['peace']),
    ]
    for i, (text, col) in enumerate(legend):
        cv2.putText(img, text, (10, h - 12 - (len(legend) - 1 - i) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1)

    if peace_progress is not None and peace_progress > 0:
        draw_peace_switch_progress(img, peace_progress, next_mode)

def draw_game2_ui(img, zone, last_action, hand_g, peace_progress, next_mode, fw, fh, ctrl_lms):
    h, w = img.shape[:2]
    draw_header(img, 'game2')

    ZONE_COLOR = {
        'neutral':  (200, 200, 200),
        'up':       (180, 255,  80),
        'down':     ( 80, 180, 255),
        'left':     (255, 180,  80),
        'right':    (255,  80, 180),
    }
    arrow_map = {'up': '^', 'down': 'v', 'left': '<', 'right': '>', 'neutral': 'O'}
    zc = ZONE_COLOR.get(zone, (200, 200, 200))

    cv2.rectangle(img, (0, 42), (w, 68), (25, 25, 25), -1)
    cv2.putText(img, f"Zone: {zone.upper()}  {arrow_map.get(zone,'?')}  |  Hand: {hand_g.upper()}",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zc, 2)

    draw_zone_overlay(img, fw, fh, active_zone=zone if zone != 'neutral' else None)

    if ctrl_lms is not None:
        fx = int(ctrl_lms[8].x * fw)
        fy = int(ctrl_lms[8].y * fh)
        cv2.circle(img, (fx, fy), 12, zc, 2)
        cv2.circle(img, (fx, fy),  4, zc, -1)

    _draw_action_flash(img, last_action, fw, fh, y_centre=h // 2 + 60)

    legend = [
        ("Open/any hand   = navigate zones (index tip)",  GESTURE_COLOR['neutral']),
        ("Index+Middle up = JUMP (overrides zone)",       GESTURE_COLOR['jump']),
        ("Shaka           = SLIDE (overrides zone)",      GESTURE_COLOR['slide']),
        ("Fist            = idle / suppress zone",        GESTURE_COLOR['idle']),
        (f"Peace both hands 1.5s = {MODE_LABEL[next_mode]}", GESTURE_COLOR['peace']),
    ]
    for i, (text, col) in enumerate(legend):
        cv2.putText(img, text, (10, h - 12 - (len(legend) - 1 - i) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1)

    if peace_progress is not None and peace_progress > 0:
        draw_peace_switch_progress(img, peace_progress, next_mode)


# MP Init

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector     = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

# Background Detection Thread

def detection_worker():
    global _latest_result, _detection_running, _stop_thread
    while not _stop_thread:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.001)
            continue
        _detection_running = True
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = detector.detect(mp_image)
        with _result_lock:
            _latest_result = result
        _detection_running = False

_det_thread = threading.Thread(target=detection_worker, daemon=True)
_det_thread.start()


# Mode Cycle

MODE_CYCLE = ['mouse', 'game1', 'game2']

def next_mode_after(mode):
    return MODE_CYCLE[(MODE_CYCLE.index(mode) + 1) % len(MODE_CYCLE)]

# ═══════════════════════════════════════════════════════════════════════════════
# Reset
# ═══════════════════════════════════════════════════════════════════════════════
def reset():
    global app_state, dist_ok_since, calib_start_t, calib_pts_x, calib_pts_y
    global calib_trail, range_min_x, range_max_x, range_min_y, range_max_y
    global smooth_x, smooth_y, calib_done_t, prev_scroll_y, last_click_t
    global scroll_entry_t, scroll_active, tm_pinch_start, tm_drag_active
    global peace_switch_timer, last_game_action, last_game_action_t
    global zone_current_zone, zone_last_key_t, _aot_applied
    if tm_drag_active:
        pyautogui.mouseUp()
    app_state          = 'distance_check'
    dist_ok_since      = None
    calib_start_t      = None
    calib_pts_x        = []
    calib_pts_y        = []
    calib_trail        = []
    calib_done_t       = None
    range_min_x        = range_max_x = None
    range_min_y        = range_max_y = None
    smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2
    prev_scroll_y      = None
    last_click_t       = 0.0
    scroll_entry_t     = None
    scroll_active      = False
    tm_pinch_start     = None
    tm_drag_active     = False
    peace_switch_timer = None
    last_game_action   = ''
    last_game_action_t = 0.0
    zone_current_zone  = 'neutral'
    zone_last_key_t    = 0.0
    _aot_applied       = False

# ═══════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════════
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    with _frame_lock:
        _latest_frame = frame.copy()
    with _result_lock:
        result = _latest_result

    left_lms, right_lms = get_sorted_hands(result)
    has_any  = left_lms is not None or right_lms is not None
    has_both = left_lms is not None and right_lms is not None
    primary_lms = right_lms if right_lms is not None else left_lms

    display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fh, fw  = display.shape[:2]

    # ── Peace-sign mode-cycle detection ───────────────────────────────────────
    peace_progress = None
    _next_mode     = next_mode_after(current_mode)

    if app_state == 'running':
        now = time.time()
        both_peace = (has_both and
                      is_peace_sign(left_lms) and
                      is_peace_sign(right_lms))
        if both_peace:
            if peace_switch_timer is None:
                peace_switch_timer = now
            progress = min((now - peace_switch_timer) / PEACE_SWITCH_HOLD, 1.0)
            peace_progress = progress
            if progress >= 1.0 and mode_switch_done_t is None:
                current_mode       = _next_mode
                _next_mode         = next_mode_after(current_mode)
                mode_switch_done_t = now
                peace_switch_timer = None
                handle_pinch_release(now)
                scroll_entry_t    = None;  scroll_active  = False
                tm_pinch_start    = None;  tm_drag_active = False
                zone_current_zone = 'neutral'
                print(f"[MODE] → {current_mode.upper()}")
        else:
            peace_switch_timer = None

    # ── Mode-switched splash ───────────────────────────────────────────────────
    if mode_switch_done_t is not None:
        elapsed_splash = time.time() - mode_switch_done_t
        if elapsed_splash < 1.5:
            if left_lms:  draw_hand(display, left_lms,  HAND_COLOR_LEFT)
            if right_lms: draw_hand(display, right_lms, HAND_COLOR_RIGHT)
            draw_mode_switched_splash(display, current_mode)
            output = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            cv2.imshow(WINDOW_NAME, output)
            if not _aot_applied:
                _aot_applied = set_window_always_on_top()
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord('r'), ord('R')):
                reset()
            continue
        else:
            mode_switch_done_t = None

    # ══════════════════════════════════════════════════════════════════════════
    # Distance Check
    # ══════════════════════════════════════════════════════════════════════════
    if app_state == 'distance_check':
        dist     = hand_size(primary_lms) if primary_lms else 0.0
        in_range = primary_lms and abs(dist - TARGET_DIST) <= DIST_TOL
        if in_range:
            if dist_ok_since is None:
                dist_ok_since = time.time()
            elif time.time() - dist_ok_since >= DIST_OK_HOLD:
                app_state     = 'calibrating'
                calib_start_t = time.time()
                dist_ok_since = None
        else:
            dist_ok_since = None
        if primary_lms:
            draw_hand(display, primary_lms)
        draw_distance_ui(display, dist, has_any)

    # ══════════════════════════════════════════════════════════════════════════
    # Calibration
    # ══════════════════════════════════════════════════════════════════════════
    elif app_state == 'calibrating':
        elapsed = time.time() - calib_start_t
        if primary_lms:
            draw_hand(display, primary_lms)
            if is_finger_extended(primary_lms, 8, 6):
                tx, ty = primary_lms[8].x, primary_lms[8].y
                calib_pts_x.append(tx)
                calib_pts_y.append(ty)
                calib_trail.append((int(tx * fw), int(ty * fh)))
                if len(calib_trail) > 300:
                    calib_trail.pop(0)
            if abs(hand_size(primary_lms) - TARGET_DIST) > DIST_TOL * 2:
                cv2.putText(display, "Distance drifted — adjust!",
                            (10, fh - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
        draw_calibration_ui(display, elapsed, CALIB_DURATION,
                            calib_trail, has_any, calib_pts_x, calib_pts_y)
        if elapsed >= CALIB_DURATION:
            if len(calib_pts_x) < 10:
                calib_start_t = time.time()
                calib_pts_x = [];  calib_pts_y = [];  calib_trail = []
            else:
                range_min_x = min(calib_pts_x);  range_max_x = max(calib_pts_x)
                range_min_y = min(calib_pts_y);  range_max_y = max(calib_pts_y)
                app_state    = 'calib_done'
                calib_done_t = time.time()
                print(f"[Calib done] x:[{range_min_x:.3f},{range_max_x:.3f}] "
                      f"y:[{range_min_y:.3f},{range_max_y:.3f}] pts:{len(calib_pts_x)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Calib Done
    # ══════════════════════════════════════════════════════════════════════════
    elif app_state == 'calib_done':
        elapsed_done = time.time() - calib_done_t
        remaining    = max(0.0, 2.0 - elapsed_done)
        if primary_lms:
            draw_hand(display, primary_lms)
        draw_calib_done_ui(display, remaining)
        if elapsed_done >= 2.0:
            app_state = 'running'

    # ══════════════════════════════════════════════════════════════════════════
    # Running — Mouse Mode
    # ══════════════════════════════════════════════════════════════════════════
    elif app_state == 'running' and current_mode == 'mouse':
        gesture  = 'idle'
        ctrl_lms = right_lms if right_lms is not None else left_lms
        if range_min_x is not None:
            cv2.rectangle(display,
                          (int(range_min_x * fw), int(range_min_y * fh)),
                          (int(range_max_x * fw), int(range_max_y * fh)),
                          (0, 255, 180), 2)
        if left_lms:
            draw_hand(display, left_lms, HAND_COLOR_LEFT)
            _label_hand(display, left_lms, "L", HAND_COLOR_LEFT, fw, fh)
        if right_lms:
            draw_hand(display, right_lms, HAND_COLOR_RIGHT)
            _label_hand(display, right_lms, "R", HAND_COLOR_RIGHT, fw, fh)
        if ctrl_lms:
            gesture = get_mouse_gesture(ctrl_lms)
            dist    = hand_size(ctrl_lms)
            dist_ok = abs(dist - TARGET_DIST) <= DIST_TOL * 2
            now     = time.time()
            target_x, target_y = map_cursor(ctrl_lms[8].x, ctrl_lms[8].y)
            smooth_x += (target_x - smooth_x) * (1 - SMOOTHING)
            smooth_y += (target_y - smooth_y) * (1 - SMOOTHING)
            if gesture == 'move':
                handle_pinch_release(now)
                pyautogui.moveTo(int(smooth_x), int(smooth_y))
                scroll_entry_t = None;  scroll_active = False;  prev_scroll_y = None
            elif gesture == 'tm_pinch':
                if tm_pinch_start is None:
                    tm_pinch_start = now
                held = now - tm_pinch_start
                if not tm_drag_active and held >= LONG_PRESS_TIME:
                    pyautogui.mouseDown();  tm_drag_active = True;  print("[DRAG START]")
                if tm_drag_active:
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))
                scroll_entry_t = None;  scroll_active = False;  prev_scroll_y = None
            elif gesture == 'right_click':
                handle_pinch_release(now)
                if now - last_click_t > CLICK_COOLDOWN:
                    pyautogui.rightClick();  last_click_t = now;  print("[RIGHT CLICK]")
                scroll_entry_t = None;  scroll_active = False;  prev_scroll_y = None
            elif gesture in ('scroll_up', 'scroll_down'):
                handle_pinch_release(now)
                if scroll_entry_t is None:
                    scroll_entry_t = now;  scroll_active = False
                elif not scroll_active and (now - scroll_entry_t) >= SCROLL_BUFFER:
                    scroll_active = True
                if scroll_active:
                    pyautogui.scroll(SCROLL_SPEED if gesture == 'scroll_up' else -SCROLL_SPEED)
                prev_scroll_y = None
            else:
                handle_pinch_release(now)
                scroll_entry_t = None;  scroll_active = False;  prev_scroll_y = None
            fx, fy    = int(ctrl_lms[8].x * fw), int(ctrl_lms[8].y * fh)
            dot_color = GESTURE_COLOR['drag'] if tm_drag_active else GESTURE_COLOR.get(gesture, (255, 255, 255))
            cv2.circle(display, (fx, fy), 12, dot_color, 2)
            cv2.circle(display, (fx, fy),  4, dot_color, -1)
            draw_mouse_ui(display, gesture, dist_ok, scroll_entry_t, scroll_active,
                          tm_pinch_start, tm_drag_active)
        else:
            handle_pinch_release(time.time())
            scroll_entry_t = None;  scroll_active = False;  prev_scroll_y = None
            draw_mouse_ui(display, 'idle', True, None, False, None, False)
        if peace_progress is not None and peace_progress > 0:
            draw_peace_switch_progress(display, peace_progress, _next_mode)

    # ══════════════════════════════════════════════════════════════════════════
    # Running — Game Mode 1  (Gesture-based)
    # ══════════════════════════════════════════════════════════════════════════
    elif app_state == 'running' and current_mode == 'game1':
        now     = time.time()
        left_g  = 'idle'
        right_g = 'idle'

        if left_lms:
            draw_hand(display, left_lms, HAND_COLOR_LEFT)
            _label_hand(display, left_lms, "L", HAND_COLOR_LEFT, fw, fh)
        if right_lms:
            draw_hand(display, right_lms, HAND_COLOR_RIGHT)
            _label_hand(display, right_lms, "R", HAND_COLOR_RIGHT, fw, fh)

        both_peace_now = (has_both and
                          is_peace_sign(left_lms) and
                          is_peace_sign(right_lms))

        if not both_peace_now:
            # ── [STEP 3] Priority — edit here if you want a different hand to win
            if has_both:
                left_g  = get_game_gesture(left_lms)
                right_g = get_game_gesture(right_lms)
                # Single-hand peace = jump; mode-switch only fires when BOTH show peace
                if left_g  == 'peace': left_g  = 'jump'
                if right_g == 'peace': right_g = 'jump'
                # Priority: left/right direction > jump/slide vertical
                if right_g in ('left', 'right'):
                    fire_game_action(right_g)
                elif left_g in ('left', 'right'):
                    fire_game_action(left_g)
                elif right_g in ('jump', 'slide'):
                    fire_game_action(right_g)
                elif left_g in ('jump', 'slide'):
                    fire_game_action(left_g)
            elif left_lms:
                left_g = get_game_gesture(left_lms)
                if left_g == 'peace': left_g = 'jump'
                fire_game_action(left_g)
            elif right_lms:
                right_g = get_game_gesture(right_lms)
                if right_g == 'peace': right_g = 'jump'
                fire_game_action(right_g)

        display_action = last_game_action if (now - last_game_action_t < 0.5) else ''
        draw_game1_ui(display, left_g, right_g, display_action, peace_progress, _next_mode)

    # ══════════════════════════════════════════════════════════════════════════
    # Running — Game Mode 2  (Zone-based, from original Code 1)
    #
    #  Index fingertip position inside calibrated bounding box → zone → key.
    #
    #  ── [ZONE STEP 3] Gesture guards (edit here) ─────────────────────────────
    #    peace/jump gesture  → fire jump, suppress zone nav
    #    shaka slide gesture → fire slide, suppress zone nav
    #    fist                → idle, suppress zone nav
    #    everything else     → zone nav active (index tip controls direction)
    # ══════════════════════════════════════════════════════════════════════════
    elif app_state == 'running' and current_mode == 'game2':
        now = time.time()

        ctrl_lms = right_lms if right_lms is not None else left_lms
        zone     = 'neutral'
        hand_g   = 'idle'

        if left_lms:
            draw_hand(display, left_lms, HAND_COLOR_LEFT)
            _label_hand(display, left_lms, "L", HAND_COLOR_LEFT, fw, fh)
        if right_lms:
            draw_hand(display, right_lms, HAND_COLOR_RIGHT)
            _label_hand(display, right_lms, "R", HAND_COLOR_RIGHT, fw, fh)

        both_peace_now = (has_both and
                          is_peace_sign(left_lms) and
                          is_peace_sign(right_lms))

        if not both_peace_now and ctrl_lms is not None and range_min_x is not None:
            f = fingers_extended(ctrl_lms)
            idx_up    = f['index']
            mid_up    = f['middle']
            ring_up   = f['ring']
            pinky_up  = f['pinky']
            thumb_out = f['thumb']

            is_fist         = not idx_up and not mid_up and not ring_up and not pinky_up and not thumb_out
            is_shaka        = thumb_out and pinky_up and not idx_up and not mid_up and not ring_up
            is_peace_single = idx_up and mid_up and not ring_up and not pinky_up

            if is_peace_single:
                hand_g = 'jump'
                fire_zone_action('up')
                zone_current_zone = 'neutral'
            elif is_shaka:
                hand_g = 'slide'
                fire_zone_action('down')
                zone_current_zone = 'neutral'
            elif is_fist:
                hand_g = 'idle'
                zone_current_zone = 'neutral'
            else:
                hand_g = 'navigate'
                zone   = get_zone(ctrl_lms[8].x, ctrl_lms[8].y)
                fire_zone_action(zone)

        display_action = last_game_action if (now - last_game_action_t < 0.5) else ''
        draw_game2_ui(display, zone, display_action, hand_g,
                      peace_progress, _next_mode, fw, fh, ctrl_lms)

    # ── Output ────────────────────────────────────────────────────────────────
    output = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    cv2.imshow(WINDOW_NAME, output)

    if not _aot_applied:
        _aot_applied = set_window_always_on_top()

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        if tm_drag_active:
            pyautogui.mouseUp()
        break
    elif key in (ord('r'), ord('R')):
        reset()
        print("[INFO] Recalibration triggered.")
    elif key in (ord('m'), ord('M')):
        current_mode       = next_mode_after(current_mode)
        mode_switch_done_t = time.time()
        print(f"[INFO] Manual cycle → {current_mode.upper()}")

# ─── Cleanup ──────────────────────────────────────────────────────────────────
_stop_thread = True
_det_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
detector.close()
