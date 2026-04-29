"""
index_cursor.py
───────────────
Right hand only. Index fingertip (landmark 8) = mouse cursor.
No ML model needed — pure landmark mapping.

Requirements:
    pip install mediapipe opencv-python pyautogui

Controls:
    Q / ESC  → quit
    C        → recenter (locks current fingertip pos as screen center,
                switching from direct-map to offset mode — see CURSOR_MODE below)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import time
import threading

# ─── Safety / speed ──────────────────────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# ─── Tuning ──────────────────────────────────────────────────────────────────

# "direct"   → fingertip position in camera frame maps 1:1 to screen.
#              Move finger to top-left corner → cursor goes top-left.
# "velocity" → tilt hand from a neutral center to move cursor (joystick feel).
#              Press C while running to lock current pos as center.
CURSOR_MODE = "direct"   # "direct" | "velocity"

# ── Direct-mode settings ──────────────────────────────────────────────────────
# Crop the camera region that maps to the full screen.
# 0.0–1.0 in normalized camera coords. Shrink region = faster across screen.
DIRECT_X_MIN = 0.10   # left  edge
DIRECT_X_MAX = 0.90   # right edge
DIRECT_Y_MIN = 0.10   # top   edge
DIRECT_Y_MAX = 0.90   # bottom edge

# ── Velocity-mode settings ────────────────────────────────────────────────────
SENSITIVITY_X = 10.0   # pixels moved per frame per unit tilt
SENSITIVITY_Y = 10.0
DEAD_ZONE     = 0.02   # ignore tiny tilts

# ── Shared ───────────────────────────────────────────────────────────────────
SMOOTHING = 0.25   # 0 = no smoothing, closer to 1 = very laggy/smooth

# MediaPipe detection settings
MIN_DETECTION_CONF = 0.5
MIN_PRESENCE_CONF  = 0.5
MIN_TRACKING_CONF  = 0.5

# ─── State ───────────────────────────────────────────────────────────────────
smooth_x, smooth_y   = float(SCREEN_W // 2), float(SCREEN_H // 2)
neutral_x, neutral_y = None, None          # velocity-mode center
center_flash_until   = 0.0

# ─── Threading ───────────────────────────────────────────────────────────────
_latest_frame  = None
_latest_result = None
_frame_lock    = threading.Lock()
_result_lock   = threading.Lock()
_stop_thread   = False

# ─── Hand skeleton connections ────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]


# ─── Draw skeleton ────────────────────────────────────────────────────────────
def draw_hand(img, lms, color=(0, 180, 255)):
    if lms is None:
        return
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for i, pt in enumerate(pts):
        dot_color = (0, 255, 80) if i == 8 else (255, 255, 255)
        cv2.circle(img, pt, 6 if i == 8 else 4, dot_color, -1)
        cv2.circle(img, pt, 6 if i == 8 else 4, color, 2)


# ─── Cursor movement ─────────────────────────────────────────────────────────
def move_cursor_direct(lms):
    """Map index fingertip position in camera → screen position."""
    global smooth_x, smooth_y
    tip = lms[8]
    # Remap from camera crop region to full screen
    nx = (tip.x - DIRECT_X_MIN) / max(DIRECT_X_MAX - DIRECT_X_MIN, 1e-6)
    ny = (tip.y - DIRECT_Y_MIN) / max(DIRECT_Y_MAX - DIRECT_Y_MIN, 1e-6)
    nx = float(np.clip(nx, 0.0, 1.0))
    ny = float(np.clip(ny, 0.0, 1.0))

    target_x = nx * SCREEN_W
    target_y = ny * SCREEN_H

    smooth_x = smooth_x + (target_x - smooth_x) * (1.0 - SMOOTHING)
    smooth_y = smooth_y + (target_y - smooth_y) * (1.0 - SMOOTHING)

    pyautogui.moveTo(int(smooth_x), int(smooth_y))


def move_cursor_velocity(lms):
    """Joystick: tilt from neutral → cursor velocity."""
    global smooth_x, smooth_y, neutral_x, neutral_y

    tip = lms[8]
    ax, ay = tip.x, tip.y

    if neutral_x is None:
        neutral_x, neutral_y = ax, ay
        pyautogui.moveTo(SCREEN_W // 2, SCREEN_H // 2)
        smooth_x, smooth_y = float(SCREEN_W // 2), float(SCREEN_H // 2)
        print(f"[CENTER] Auto-locked ({ax:.3f}, {ay:.3f})")
        return

    dx = ax - neutral_x
    dy = ay - neutral_y

    if abs(dx) < DEAD_ZONE: dx = 0.0
    if abs(dy) < DEAD_ZONE: dy = 0.0

    smooth_x += dx * SCREEN_W  * SENSITIVITY_X
    smooth_y += dy * SCREEN_H  * SENSITIVITY_Y  # positive = down (matches camera)

    smooth_x = float(np.clip(smooth_x, 0, SCREEN_W - 1))
    smooth_y = float(np.clip(smooth_y, 0, SCREEN_H - 1))

    pyautogui.moveTo(int(smooth_x), int(smooth_y))


def set_center(lms):
    global neutral_x, neutral_y, center_flash_until
    tip = lms[8]
    neutral_x, neutral_y = tip.x, tip.y
    center_flash_until = time.time() + 0.6
    print(f"[CENTER] Locked ({neutral_x:.3f}, {neutral_y:.3f})")


# ─── Get the RIGHT hand from result ──────────────────────────────────────────
def get_right_hand(result):
    """
    MediaPipe labels from the camera's perspective.
    Camera-'Right' = user's LEFT hand (mirrored).
    We flip the frame, so camera-'Left' label = user's RIGHT hand.
    """
    if not result or not result.hand_landmarks:
        return None
    for lms, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name
        if label == "Left":      # after flip, "Left" = user's right hand
            return lms
    return None


# ─── Detection worker ─────────────────────────────────────────────────────────
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


# ─── MediaPipe setup ─────────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options      = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,                          # only need 1 hand
    min_hand_detection_confidence=MIN_DETECTION_CONF,
    min_hand_presence_confidence=MIN_PRESENCE_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

_det_thread = threading.Thread(target=detection_worker, daemon=True)
_det_thread.start()

print(f"[INFO] Running in CURSOR_MODE = '{CURSOR_MODE}'")
print("[INFO] Press C to lock center (velocity mode) | Q/ESC to quit")

# ─── Main loop ────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)   # mirror so movement feels natural

    with _frame_lock:
        _latest_frame = frame.copy()
    with _result_lock:
        result = _latest_result

    lms_right = get_right_hand(result)
    display   = frame.copy()
    fh, fw    = display.shape[:2]

    # ── Move cursor ───────────────────────────────────────────────────────────
    if lms_right is not None:
        if CURSOR_MODE == "direct":
            move_cursor_direct(lms_right)
        else:
            move_cursor_velocity(lms_right)

    # ── Draw ──────────────────────────────────────────────────────────────────
    draw_hand(display, lms_right)

    # Overlay: mode + status
    mode_color = (0, 220, 120) if lms_right else (0, 80, 200)
    cv2.rectangle(display, (0, 0), (fw, 52), (15, 15, 15), -1)
    status = "RIGHT HAND DETECTED" if lms_right else "Show RIGHT hand..."
    cv2.putText(display, f"Mode: {CURSOR_MODE.upper()}  |  {status}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1)
    cv2.putText(display, "Index tip (LM8) = cursor   |  Q=quit  C=center(vel)",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # Highlight index fingertip
    if lms_right is not None:
        tip = lms_right[8]
        tx, ty = int(tip.x * fw), int(tip.y * fh)
        cv2.circle(display, (tx, ty), 14, (0, 255, 80), 2)
        cv2.circle(display, (tx, ty),  4, (0, 255, 80), -1)
        cv2.putText(display, "LM8", (tx + 16, ty - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1)

    # Center-locked flash
    if time.time() < center_flash_until:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 180, 60), -1)
        cv2.addWeighted(overlay, 0.18, display, 0.82, 0, display)
        cv2.putText(display, "CENTER LOCKED", (fw // 2 - 100, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 100), 3)

    # Direct-mode crop region guide
    if CURSOR_MODE == "direct":
        x1 = int(DIRECT_X_MIN * fw); x2 = int(DIRECT_X_MAX * fw)
        y1 = int(DIRECT_Y_MIN * fh); y2 = int(DIRECT_Y_MAX * fh)
        cv2.rectangle(display, (x1, y1), (x2, y2), (80, 80, 255), 1)
        cv2.putText(display, "active zone", (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 255), 1)

    cv2.imshow("Index Finger Cursor  |  Right Hand", display)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q'), ord('Q')):
        break
    elif key in (ord('c'), ord('C')):
        if lms_right is not None:
            set_center(lms_right)
            if CURSOR_MODE == "direct":
                print("[INFO] C pressed — switch CURSOR_MODE to 'velocity' to use centering")
        else:
            print("[CENTER] No right hand visible")

# ─── Cleanup ─────────────────────────────────────────────────────────────────
_stop_thread = True
_det_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
detector.close()
print("[INFO] Exited cleanly.")