import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import time
import threading
import torch
import torch.nn as nn
import joblib
import math

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# ─── Tuning ──────────────────────────────────────────────────────────────────
TARGET_DIST       = 0.18
DIST_TOL          = 0.03
DIST_OK_HOLD      = 1.0
KEY_COOLDOWN      = 0.15
CONFIDENCE_THRESH = 0.75

# ─── RIGHT HAND CURSOR (index fingertip direct mapping) ───────────────────────
DIRECT_X_MIN = 0.10
DIRECT_X_MAX = 0.90
DIRECT_Y_MIN = 0.10
DIRECT_Y_MAX = 0.90

CURSOR_SMOOTHING = 0.20

# ─── PEACE GESTURE ESC ───────────────────────────────────────────────────────
PEACE_HOLD_DURATION = 0.8

# ─── PINCH / GESTURE THRESHOLDS ───────────────────────────────────────────────
PINCH_THRESH   = 0.3
RELEASE_THRESH = 0.35

# ─── Key Mappings ─────────────────────────────────────────────────────────────
MOVEMENT_KEY_MAP = {
    'forward':       'w',
    'forward_right': ('w', 'd'),
    'right':         'd',
    'back_right':    ('s', 'd'),
    'back':          's',
    'back_left':     ('s', 'a'),
    'left':          'a',
    'forward_left':  ('w', 'a'),
    'crouch':        'ctrl',
    'jump':          'space',
    'idle':          None,
}

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ─── NN Model Definition (LEFT hand only) ────────────────────────────────────
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ─── Load LEFT hand model (movement) ─────────────────────────────────────────
le_movement    = joblib.load('label_encoder_movement.pkl')
movement_model = GestureNet(126, len(le_movement.classes_))
movement_model.load_state_dict(torch.load('gesture_movement.pt', map_location='cpu'))
movement_model.eval()
print(f"[NN-LEFT]  Movement classes : {list(le_movement.classes_)}")

# ─── App State ────────────────────────────────────────────────────────────────
app_state     = 'distance_check'
dist_ok_since = None

# Cursor state (right hand, direct mode)
smooth_x, smooth_y = float(SCREEN_W // 2), float(SCREEN_H // 2)

# Movement state (left hand, NN)
prev_row_left    = None
current_movement = 'idle'
held_move_keys   = set()
last_move_t      = 0.0

# Gun state (right hand, pure geometry)
fire_held         = False
scope_held        = False
last_gun_t        = 0.0
current_gun_label = 'none'  # for display only

# Peace ESC state
peace_both_since = None
esc_fired        = False
esc_flash_until  = 0.0

# Threading
_latest_frame  = None
_latest_result = None
_frame_lock    = threading.Lock()
_result_lock   = threading.Lock()
_stop_thread   = False


# ─── Helpers ─────────────────────────────────────────────────────────────────

def hand_size(lms):
    return np.hypot(lms[0].x - lms[9].x, lms[0].y - lms[9].y)


def palm_width(lms):
    return math.hypot(lms[5].x - lms[17].x, lms[5].y - lms[17].y)


def tip_dist(lms, a, b):
    pw = max(palm_width(lms), 1e-6)
    d  = math.hypot(lms[a].x - lms[b].x, lms[a].y - lms[b].y)
    return d / pw


def is_finger_extended(lms, tip, pip):
    tip_d = np.hypot(lms[tip].x - lms[0].x, lms[tip].y - lms[0].y)
    pip_d = np.hypot(lms[pip].x - lms[0].x, lms[pip].y - lms[0].y)
    return tip_d > pip_d


def is_peace_gesture(lms):
    index_up   = is_finger_extended(lms, 8,  6)
    middle_up  = is_finger_extended(lms, 12, 10)
    ring_down  = not is_finger_extended(lms, 16, 14)
    pinky_down = not is_finger_extended(lms, 20, 18)
    return index_up and middle_up and ring_down and pinky_down


def is_reload_gesture(lms):
    index_up       = is_finger_extended(lms, 8,  6)
    middle_up      = is_finger_extended(lms, 12, 10)
    ring_up        = is_finger_extended(lms, 16, 14)
    thumb_pinky    = tip_dist(lms, 4, 20) < PINCH_THRESH
    return index_up and middle_up and ring_up and thumb_pinky


# ─── Right-hand gesture detection (geometry only) ─────────────────────────────

def detect_right_gesture(lms):
    """
    Returns one of: 'scope_fire', 'fire', 'scope', 'reload', 'none'

    Gesture definitions:
      scope_fire  → thumb + middle + ring all pinched together (3-way pinch)
                    Holds BOTH left and right mouse buttons simultaneously.
      fire        → thumb ↔ middle finger pinch (ring NOT pinched)
      scope       → thumb ↔ ring finger pinch (middle NOT pinched)
                    Holds RMB while held, releases on release.
      reload      → index + middle + ring extended
      none        → neutral hand
    """
    if is_reload_gesture(lms):
        return 'reload'

    # scope_fire: thumb close to BOTH middle AND ring
    thumb_middle_dist = tip_dist(lms, 4, 12)  # thumb ↔ middle tip
    thumb_ring_dist   = tip_dist(lms, 4, 16)  # thumb ↔ ring tip
    if thumb_middle_dist < PINCH_THRESH and thumb_ring_dist < PINCH_THRESH:
        return 'scope_fire'

    # fire: thumb ↔ middle pinch, ring NOT pinched
    fire_pinch  = tip_dist(lms, 4, 12) < PINCH_THRESH

    # scope: thumb ↔ ring pinch, middle NOT pinched (guard against scope_fire)
    scope_pinch = (tip_dist(lms, 4, 16) < PINCH_THRESH and
                   tip_dist(lms, 4, 12) >= RELEASE_THRESH)

    if scope_pinch:
        return 'scope'
    if fire_pinch:
        return 'fire'
    return 'none'


# ─── Feature extraction & prediction (LEFT hand NN) ──────────────────────────

def extract_features(lms, prev_row):
    wrist_x, wrist_y, wrist_z = lms[0].x, lms[0].y, lms[0].z
    scale = math.sqrt((lms[9].x - wrist_x)**2 +
                      (lms[9].y - wrist_y)**2 +
                      (lms[9].z - wrist_z)**2)
    scale = max(scale, 1e-6)
    row = []
    for lm in lms:
        row.extend([(lm.x - wrist_x) / scale,
                    (lm.y - wrist_y) / scale,
                    (lm.z - wrist_z) / scale])
    delta = [cur - p for cur, p in zip(row, prev_row)] if prev_row else [0.0] * 63
    return row + delta, row


def predict_movement(lms, prev_row):
    features, new_prev = extract_features(lms, prev_row)
    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        probs     = torch.softmax(movement_model(x), dim=1)[0]
        conf, idx = probs.max(0)
    if conf.item() < CONFIDENCE_THRESH:
        return 'idle', conf.item(), new_prev
    return le_movement.inverse_transform([idx.item()])[0], conf.item(), new_prev


# ─── Cursor movement (right hand, direct index-tip mapping) ──────────────────

def move_cursor_direct(lms):
    global smooth_x, smooth_y
    tip = lms[8]
    nx = (tip.x - DIRECT_X_MIN) / max(DIRECT_X_MAX - DIRECT_X_MIN, 1e-6)
    ny = (tip.y - DIRECT_Y_MIN) / max(DIRECT_Y_MAX - DIRECT_Y_MIN, 1e-6)
    nx = float(np.clip(nx, 0.0, 1.0))
    ny = float(np.clip(ny, 0.0, 1.0))
    target_x = nx * SCREEN_W
    target_y = ny * SCREEN_H
    smooth_x += (target_x - smooth_x) * (1.0 - CURSOR_SMOOTHING)
    smooth_y += (target_y - smooth_y) * (1.0 - CURSOR_SMOOTHING)
    pyautogui.moveTo(int(smooth_x), int(smooth_y))


# ─── Draw helpers ─────────────────────────────────────────────────────────────

def draw_hand(img, lms, color=(0, 200, 100)):
    if lms is None:
        return
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(img, pt, 5, (255, 255, 255), -1)
        cv2.circle(img, pt, 5, color, 2)


def draw_index_tip(img, lms, color=(0, 255, 80)):
    if lms is None:
        return
    h, w = img.shape[:2]
    tip = lms[8]
    tx, ty = int(tip.x * w), int(tip.y * h)
    cv2.circle(img, (tx, ty), 14, color, 2)
    cv2.circle(img, (tx, ty),  4, color, -1)
    cv2.putText(img, "AIM", (tx + 16, ty - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


# ─── Key helpers ──────────────────────────────────────────────────────────────

def _to_key_list(k):
    if k is None:
        return []
    return list(k) if isinstance(k, tuple) else [k]


def apply_movement_keys(gesture):
    global held_move_keys
    desired = set(_to_key_list(MOVEMENT_KEY_MAP.get(gesture)))
    for k in held_move_keys - desired:
        pyautogui.keyUp(k)
    for k in desired - held_move_keys:
        pyautogui.keyDown(k)
    held_move_keys = desired


def release_all_movement_keys():
    global held_move_keys
    for k in held_move_keys:
        pyautogui.keyUp(k)
    held_move_keys = set()


def release_all_mouse_buttons():
    global fire_held, scope_held
    if fire_held:
        pyautogui.mouseUp(button='left')
        fire_held = False
    if scope_held:
        pyautogui.mouseUp(button='right')
        scope_held = False


# ─── Gun action application ───────────────────────────────────────────────────

def apply_gun_action(gesture):
    """
    scope_fire → hold BOTH left and right mouse buttons simultaneously
    fire       → hold left mouse button (release right if held)
    scope      → hold right mouse button while gesture is active,
                 release right when gesture is released (NO toggle logic)
    reload     → tap R once on gesture entry
    none       → release all mouse buttons
    """
    global fire_held, scope_held, last_gun_t, current_gun_label
    now  = time.time()
    prev = current_gun_label

    if gesture == 'scope_fire':
        # Hold both LMB and RMB
        if not fire_held:
            pyautogui.mouseDown(button='left')
            fire_held = True
            print("[GUN] SCOPE_FIRE → mouseDown left")
        if not scope_held:
            pyautogui.mouseDown(button='right')
            scope_held = True
            print("[GUN] SCOPE_FIRE → mouseDown right")

    elif gesture == 'fire':
        # Hold LMB, release RMB if held
        if scope_held:
            pyautogui.mouseUp(button='right')
            scope_held = False
            print("[GUN] FIRE → released right mouse")
        if not fire_held:
            pyautogui.mouseDown(button='left')
            fire_held = True
            print("[GUN] FIRE → mouseDown left")

    elif gesture == 'scope':
        # Hold RMB while gesture is active, release LMB if held
        if fire_held:
            pyautogui.mouseUp(button='left')
            fire_held = False
            print("[GUN] SCOPE → released left mouse")
        if not scope_held:
            pyautogui.mouseDown(button='right')
            scope_held = True
            print("[GUN] SCOPE → mouseDown right (scoping in)")

    elif gesture == 'reload':
        release_all_mouse_buttons()
        if prev != 'reload':
            if (now - last_gun_t) > KEY_COOLDOWN:
                pyautogui.press('r')
                last_gun_t = now
                print("[GUN] RELOAD → tap R")

    else:  # 'none'
        # Release everything when hand returns to neutral
        if fire_held or scope_held:
            if fire_held:
                pyautogui.mouseUp(button='left')
                fire_held = False
                print("[GUN] NONE → released left mouse")
            if scope_held:
                pyautogui.mouseUp(button='right')
                scope_held = False
                print("[GUN] NONE → released right mouse (scoped out)")

    current_gun_label = gesture


# ─── Peace ESC ────────────────────────────────────────────────────────────────

def check_peace_esc(lms_left, lms_right):
    global peace_both_since, esc_fired
    both = (lms_left  is not None and is_peace_gesture(lms_left) and
            lms_right is not None and is_peace_gesture(lms_right))
    if both:
        if peace_both_since is None:
            peace_both_since = time.time()
            esc_fired = False
        elif not esc_fired and (time.time() - peace_both_since) >= PEACE_HOLD_DURATION:
            release_all_movement_keys()
            release_all_mouse_buttons()
            pyautogui.press('escape')
            esc_fired = True
            print("[ESC] Double peace → ESC fired")
            return True
    else:
        peace_both_since = None
        esc_fired = False
    return False


# ─── MediaPipe ────────────────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options      = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)


# ─── Background detection thread ─────────────────────────────────────────────

def detection_worker():
    global _latest_result
    while not _stop_thread:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.001)
            continue
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with _result_lock:
            _latest_result = detector.detect(mp_img)

_det_thread = threading.Thread(target=detection_worker, daemon=True)
_det_thread.start()


# ─── Split hands ──────────────────────────────────────────────────────────────

def split_hands(result):
    lms_left = lms_right = None
    if not result or not result.hand_landmarks:
        return lms_left, lms_right
    for lms, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name
        if label == 'Right':
            lms_left  = lms
        else:
            lms_right = lms
    return lms_left, lms_right


# ─── Reset ────────────────────────────────────────────────────────────────────

def full_reset():
    global app_state, dist_ok_since
    global smooth_x, smooth_y
    global current_movement, prev_row_left
    global current_gun_label
    global peace_both_since, esc_fired
    release_all_movement_keys()
    release_all_mouse_buttons()
    app_state         = 'distance_check'
    dist_ok_since     = None
    smooth_x          = float(SCREEN_W // 2)
    smooth_y          = float(SCREEN_H // 2)
    current_movement  = 'idle'
    current_gun_label = 'none'
    prev_row_left     = None
    peace_both_since  = None
    esc_fired         = False
    print("[INFO] Reset to distance check.")


# ─── Distance UI ─────────────────────────────────────────────────────────────

def draw_distance_ui(img, dist, has_hand, ok_since):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 80), (20, 20, 20), -1)
    if not has_hand:
        cv2.putText(img, "Show at least one hand to the camera",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
        return
    too_close = dist > TARGET_DIST + DIST_TOL
    too_far   = dist < TARGET_DIST - DIST_TOL
    if too_close:
        msg, color = "Move FARTHER from camera",    (0,  80, 255)
    elif too_far:
        msg, color = "Move CLOSER to camera",       (0, 200, 255)
    else:
        msg, color = "Distance OK!  Hold still...", (0, 220,  80)
    cv2.putText(img, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    bar_x, bar_y, bar_w, bar_h = 10, 90, w - 20, 18
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(np.clip(dist / (TARGET_DIST * 2), 0, 1) * bar_w)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    lo = int((TARGET_DIST - DIST_TOL) / (TARGET_DIST * 2) * bar_w) + bar_x
    hi = int((TARGET_DIST + DIST_TOL) / (TARGET_DIST * 2) * bar_w) + bar_x
    cv2.rectangle(img, (lo, bar_y), (hi, bar_y + bar_h), (0, 255, 100), 2)
    if ok_since is not None:
        elapsed = time.time() - ok_since
        frac    = min(elapsed / DIST_OK_HOLD, 1.0)
        cv2.rectangle(img, (10, 116), (10 + int(frac * (w - 20)), 130), (0, 220, 80), -1)
        cv2.putText(img, f"Hold... {DIST_OK_HOLD - elapsed:.1f}s",
                    (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)


# ─── Main loop ────────────────────────────────────────────────────────────────

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    with _frame_lock:
        _latest_frame = frame.copy()
    with _result_lock:
        result = _latest_result

    lms_left, lms_right = split_hands(result)
    display = frame.copy()
    fh, fw  = display.shape[:2]

    # ── Distance Check ────────────────────────────────────────────────────────
    if app_state == 'distance_check':
        ref_lms  = lms_left or lms_right
        has_hand = ref_lms is not None
        dist     = hand_size(ref_lms) if has_hand else 0.0
        in_range = has_hand and abs(dist - TARGET_DIST) <= DIST_TOL

        if in_range:
            if dist_ok_since is None:
                dist_ok_since = time.time()
            elif time.time() - dist_ok_since >= DIST_OK_HOLD:
                app_state     = 'running'
                dist_ok_since = None
                pyautogui.moveTo(SCREEN_W // 2, SCREEN_H // 2)
                print("[INFO] Distance OK — controller active.")
        else:
            dist_ok_since = None

        draw_hand(display, lms_left,  (0, 200, 100))
        draw_hand(display, lms_right, (0, 100, 220))
        draw_distance_ui(display, dist, has_hand, dist_ok_since)

    # ── Running ───────────────────────────────────────────────────────────────
    elif app_state == 'running':
        mv_gest  = 'idle'
        mv_conf  = 0.0
        gun_gest = 'none'

        # --- Peace ESC ---
        esc_triggered = check_peace_esc(lms_left, lms_right)
        if esc_triggered:
            esc_flash_until = time.time() + 0.5

        peace_progress = 0.0
        if peace_both_since is not None:
            peace_progress = min((time.time() - peace_both_since) / PEACE_HOLD_DURATION, 1.0)

        esc_flash = time.time() < esc_flash_until

        # --- LEFT hand → movement (NN) ---
        if lms_left:
            mv_gest, mv_conf, prev_row_left = predict_movement(lms_left, prev_row_left)
            if mv_gest != current_movement:
                apply_movement_keys(mv_gest)
                print(f"[MOVE] {current_movement} → {mv_gest}")
                current_movement = mv_gest
        else:
            if held_move_keys:
                release_all_movement_keys()
                current_movement = 'idle'
            prev_row_left = None

        # --- RIGHT hand → aim (index tip) + gun gestures ---
        if lms_right:
            move_cursor_direct(lms_right)
            gun_gest = detect_right_gesture(lms_right)
            apply_gun_action(gun_gest)
        else:
            release_all_mouse_buttons()
            current_gun_label = 'none'
            gun_gest = 'none'

        # --- Draw ---
        draw_hand(display, lms_left,  (80, 220, 80))
        draw_hand(display, lms_right, (80, 140, 255))
        draw_index_tip(display, lms_right)

        x1 = int(DIRECT_X_MIN * fw); x2 = int(DIRECT_X_MAX * fw)
        y1 = int(DIRECT_Y_MIN * fh); y2 = int(DIRECT_Y_MAX * fh)
        cv2.rectangle(display, (x1, y1), (x2, y2), (60, 60, 180), 1)

        # HUD
        mv_color = (100, 255, 100) if mv_gest != 'idle' else (140, 140, 140)
        GUN_COLORS = {
            'scope_fire': (0,  200, 255),
            'fire':       (0,   80, 255),
            'scope':      (255, 80, 200),
            'reload':     (255, 200,  0),
            'none':       (140, 140, 140),
        }
        gun_color = GUN_COLORS.get(gun_gest, (140, 140, 140))

        cv2.rectangle(display, (0, 0), (fw // 2, 46), (20, 40, 20), -1)
        cv2.putText(display, f"L (NN): {mv_gest} [{mv_conf:.2f}]",
                    (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mv_color, 2)

        cv2.rectangle(display, (fw // 2, 0), (fw, 46), (40, 20, 20), -1)
        scope_tag = " [SCOPED]" if scope_held else ""
        fire_tag  = " [FIRING]" if fire_held  else ""
        cv2.putText(display, f"R (geo): {gun_gest}{scope_tag}{fire_tag}",
                    (fw // 2 + 8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gun_color, 2)

        # Legend
        legend = [
            ("INDEX TIP         → aim cursor (direct)",    (0, 220, 255)),
            ("THUMB+MID         → hold fire (LMB)",        (0,  80, 255)),
            ("THUMB+RING        → hold scope (RMB)",       (255, 80, 200)),
            ("THUMB+MID+RING   → scope + fire (LMB+RMB)", (0, 200, 255)),
            ("IDX+MID+RING+THUMB↔PINKY → reload (R)",     (255, 200, 0)),
            ("Both PEACE        → ESC",                    (200, 200, 60)),
            ("Q / ESC → quit",                              (100, 100, 100)),
        ]
        for i, (txt, col) in enumerate(legend):
            y = fh - 12 - (len(legend) - 1 - i) * 17
            cv2.putText(display, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)

        # Peace progress bar
        if peace_progress > 0:
            cv2.rectangle(display, (0, fh - 6), (int(peace_progress * fw), fh), (60, 255, 120), -1)
            cv2.putText(display, "Hold PEACE for ESC...",
                        (fw // 2 - 90, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 120), 1)

        # ESC flash
        if esc_flash:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
            cv2.putText(display, "ESC", (fw // 2 - 60, fh // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)

    cv2.imshow('Dual-Hand Game Controller', display)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q'), ord('Q')):
        break


# ─── Cleanup ─────────────────────────────────────────────────────────────────
release_all_movement_keys()
release_all_mouse_buttons()
_stop_thread = True
_det_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
detector.close()
print("[INFO] Exited cleanly.")