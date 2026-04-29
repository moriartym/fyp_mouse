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

# ─── Tuning ───────────────────────────────────────────────────────────────────
HAND_SIZE_MIN  = 70
HAND_SIZE_MAX  = 200
DIST_OK_HOLD   = 1.0
STEER_DEADZONE = 5
STEER_MAX      = 40

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

# ─── Landmark indices ─────────────────────────────────────────────────────────
# Finger tip and pip (knuckle) indices
#   Thumb:  tip=4, ip=3, mcp=2
#   Index:  tip=8, pip=6
#   Middle: tip=12, pip=10
#   Ring:   tip=16, pip=14
#   Pinky:  tip=20, pip=18

def is_thumbs_up(lms):
    """
    True only when thumb points UP and all other fingers are curled.
    Checks both Y axis (thumb raised) AND X axis (thumb away from palm),
    preventing sideways thumbs from triggering.
    """
    # ── 1. All four fingers must be TIGHTLY curled ────────────────────────
    # tip must be BELOW the MCP (base knuckle), not just the PIP.
    # This is a much stricter curl check than tip < pip.
    finger_tips = [8,  12, 16, 20]
    finger_mcps = [5,   9, 13, 17]
    for tip, mcp in zip(finger_tips, finger_mcps):
        if lms[tip].y < lms[mcp].y:   # finger is open
            return False

    # ── 2. Thumb tip must be clearly ABOVE the wrist ──────────────────────
    wrist = lms[0]
    thumb_tip = lms[4]
    thumb_mcp = lms[2]   # base of thumb
    thumb_ip  = lms[3]   # middle thumb joint

    if thumb_tip.y >= wrist.y - 0.05:   # not raised high enough
        return False

    # ── 3. Thumb must point UP, not sideways ──────────────────────────────
    # The tip→ip vector should be more vertical than horizontal.
    dx = abs(thumb_tip.x - thumb_ip.x)
    dy = abs(thumb_tip.y - thumb_ip.y)   # in image coords, up = smaller y
    if dy < dx:   # more horizontal than vertical → sideways thumb
        return False

    # ── 4. Thumb tip must be EXTENDED away from palm center ───────────────
    # Palm center ≈ landmark 9 (middle finger MCP).
    palm_x = lms[9].x
    dist_tip  = abs(thumb_tip.x - palm_x)
    dist_mcp  = abs(thumb_mcp.x - palm_x)
    if dist_tip < dist_mcp * 0.8:   # thumb folded inward
        return False

    return True

# ─── State ────────────────────────────────────────────────────────────────────
app_state     = 'distance_check'
dist_ok_since = None
held_keys     = set()

_frame_left  = _frame_right  = None
_result_left = _result_right = None
_lock_fl = threading.Lock(); _lock_fr = threading.Lock()
_lock_rl = threading.Lock(); _lock_rr = threading.Lock()
_stop_threads = False

# ─── Helpers ──────────────────────────────────────────────────────────────────

def hand_size_px(lms, w, h):
    x0, y0 = lms[0].x * w, lms[0].y * h
    x9, y9 = lms[9].x * w, lms[9].y * h
    return math.sqrt((x9 - x0)**2 + (y9 - y0)**2)

def get_steer_angle(lms_left, lms_right):
    lx = lms_left[0].x  * 0.5
    ly = lms_left[0].y
    rx = lms_right[0].x * 0.5 + 0.5
    ry = lms_right[0].y
    return np.degrees(np.arctan2(ry - ly, rx - lx))

def draw_hand_on_half(img, lms, x_offset, color=(0, 200, 100)):
    if lms is None:
        return
    h, w = img.shape[:2]
    hw = w // 2
    pts = [(int(lm.x * hw + x_offset), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(img, pt, 5, (255, 255, 255), -1)
        cv2.circle(img, pt, 5, color, 2)

def set_held_keys(desired):
    for k in list(held_keys - desired):
        try: pyautogui.keyUp(k)
        except: pass
        held_keys.discard(k)
    for k in list(desired - held_keys):
        pyautogui.keyDown(k)
        held_keys.add(k)

def release_all():
    for k in list(held_keys):
        try: pyautogui.keyUp(k)
        except: pass
    held_keys.clear()

# ─── MediaPipe ────────────────────────────────────────────────────────────────

def make_detector():
    opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=1)
    return vision.HandLandmarker.create_from_options(opts)

detector_left  = make_detector()
detector_right = make_detector()

def worker_left():
    global _result_left
    while not _stop_threads:
        with _lock_fl: frame = _frame_left
        if frame is None: time.sleep(0.001); continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with _lock_rl:
            _result_left = detector_left.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

def worker_right():
    global _result_right
    while not _stop_threads:
        with _lock_fr: frame = _frame_right
        if frame is None: time.sleep(0.001); continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with _lock_rr:
            _result_right = detector_right.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

threading.Thread(target=worker_left,  daemon=True).start()
threading.Thread(target=worker_right, daemon=True).start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

# ─── Draw ─────────────────────────────────────────────────────────────────────

def draw_distance_ui(img, size_l, size_r, has_l, has_r):
    h, w = img.shape[:2]
    hw   = w // 2
    cv2.rectangle(img, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.line(img, (hw, 0), (hw, 90), (60, 60, 60), 1)

    for side, size, has, x0 in [
        ('LEFT',  size_l, has_l, 10),
        ('RIGHT', size_r, has_r, hw + 10)
    ]:
        if not has:
            cv2.putText(img, f"{side}: show hand", (x0, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
            continue
        too_close = size > HAND_SIZE_MAX
        too_far   = size < HAND_SIZE_MIN
        if too_close:   msg, color = f"{side}: farther", (0,  80, 255)
        elif too_far:   msg, color = f"{side}: closer",  (0, 200, 255)
        else:           msg, color = f"{side}: OK!",     (0, 220,  80)
        cv2.putText(img, msg, (x0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        bar_max = hw - 20
        fill    = int(min(size / (HAND_SIZE_MAX * 1.2), 1.0) * bar_max)
        g_s     = int(HAND_SIZE_MIN / (HAND_SIZE_MAX * 1.2) * bar_max)
        g_e     = int(HAND_SIZE_MAX / (HAND_SIZE_MAX * 1.2) * bar_max)
        cv2.rectangle(img, (x0, 50), (x0 + bar_max, 68), (60, 60, 60), -1)
        cv2.rectangle(img, (x0 + g_s, 50), (x0 + g_e, 68), (0, 80, 0), -1)
        cv2.rectangle(img, (x0, 50), (x0 + fill, 68), color, -1)
        cv2.rectangle(img, (x0, 50), (x0 + bar_max, 68), (120, 120, 120), 1)

    cv2.putText(img, "Hold both hands at correct distance for 1s",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

def draw_running_ui(img, angle, steer_dir, accel, brake, fw, fh,
                    lms_left, lms_right, thumbs_l, thumbs_r):
    h, w = img.shape[:2]
    hw   = w // 2

    cv2.line(img, (hw, 0), (hw, h), (80, 80, 80), 1)
    cv2.putText(img, "LEFT",  (10,      h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180,  80), 1)
    cv2.putText(img, "RIGHT", (hw + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,  80, 180), 1)

    # thumbs-up indicator per hand
    gl_color = (0, 220, 80) if thumbs_l else (180, 180, 180)
    gr_color = (0, 220, 80) if thumbs_r else (180, 180, 180)
    gl_label = "L: THUMBS UP" if thumbs_l else "L: open hand"
    gr_label = "R: THUMBS UP" if thumbs_r else "R: open hand"
    cv2.putText(img, gl_label, (10,      h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gl_color, 1)
    cv2.putText(img, gr_label, (hw + 10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gr_color, 1)

    # top bar — steer
    cv2.rectangle(img, (0, 0), (w, 44), (25, 25, 25), -1)
    sc = {'left': (255, 180, 80), 'right': (255, 80, 180), 'none': (180, 180, 180)}[steer_dir]
    sl = {'left': 'LEFT  <', 'right': 'RIGHT  >', 'none': 'STRAIGHT'}[steer_dir]
    cv2.putText(img, f"Steer: {sl}  ({angle:+.1f}deg)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc, 2)

    # accel / brake indicators
    cv2.rectangle(img, (w - 165, 4), (w - 88, 38), (0, 180, 60) if accel else (50, 50, 50), -1)
    cv2.putText(img, "ACCEL ^", (w - 162, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.rectangle(img, (w - 82,  4), (w - 4,  38), (0,  80, 220) if brake else (50, 50, 50), -1)
    cv2.putText(img, "BRAKE v", (w - 79,  28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # wrist line
    if lms_left and lms_right:
        lx = int(lms_left[0].x  * hw);      ly = int(lms_left[0].y  * fh)
        rx = int(lms_right[0].x * hw + hw); ry = int(lms_right[0].y * fh)
        cv2.line(img, (lx, ly), (rx, ry), sc, 3)
        cv2.circle(img, (lx, ly), 10, (255, 180,  80), -1)
        cv2.circle(img, (rx, ry), 10, (255,  80, 180), -1)

    # steering gauge
    cx, cy, r = w // 2, h - 95, 52
    cv2.circle(img, (cx, cy), r, (50, 50, 50), 3)
    for deg in [-STEER_DEADZONE, STEER_DEADZONE]:
        rad = np.radians(deg)
        cv2.line(img,
                 (int(cx + (r - 8) * np.sin(rad)), int(cy - (r - 8) * np.cos(rad))),
                 (int(cx + (r + 8) * np.sin(rad)), int(cy + (r + 8) * np.cos(rad))),
                 (100, 100, 100), 2)
    rad = np.radians(np.clip(angle, -STEER_MAX, STEER_MAX))
    cv2.line(img, (cx, cy),
             (int(cx + r * np.sin(rad)), int(cy - r * np.cos(rad))), sc, 3)
    cv2.circle(img, (cx, cy), 6, sc, -1)
    cv2.putText(img, "L", (cx - r - 20, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180,  80), 2)
    cv2.putText(img, "R", (cx + r + 6,  cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,  80, 180), 2)

    # legend
    legend = [
        ("Right hand thumbs up = ACCEL", (  0, 220,  80)),
        ("Left  hand thumbs up = BRAKE", (  0, 120, 255)),
        ("Tilt hands = STEER L / R",     (255, 180,  80)),
        ("R = recalibrate",              (100, 100, 100)),
    ]
    for i, (txt, col) in enumerate(legend):
        cv2.putText(img, txt,
                    (10, h - 50 - (len(legend) - 1 - i) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1)

# ─── Main loop ────────────────────────────────────────────────────────────────

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    hw     = fw // 2

    with _lock_fl: _frame_left  = frame[:, :hw].copy()
    with _lock_fr: _frame_right = frame[:, hw:].copy()
    with _lock_rl: res_l = _result_left
    with _lock_rr: res_r = _result_right

    lms_left  = res_l.hand_landmarks[0] if (res_l and res_l.hand_landmarks)  else None
    lms_right = res_r.hand_landmarks[0] if (res_r and res_r.hand_landmarks)  else None

    display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now     = time.time()

    if app_state == 'distance_check':
        size_l  = hand_size_px(lms_left,  hw, fh) if lms_left  else 0.0
        size_r  = hand_size_px(lms_right, hw, fh) if lms_right else 0.0
        ok_l    = lms_left  is not None and HAND_SIZE_MIN <= size_l <= HAND_SIZE_MAX
        ok_r    = lms_right is not None and HAND_SIZE_MIN <= size_r <= HAND_SIZE_MAX
        both_ok = ok_l and ok_r

        if both_ok:
            if dist_ok_since is None: dist_ok_since = now
            elif now - dist_ok_since >= DIST_OK_HOLD:
                app_state = 'running'; dist_ok_since = None
        else:
            dist_ok_since = None

        draw_hand_on_half(display, lms_left,  0,  (255, 180,  80))
        draw_hand_on_half(display, lms_right, hw, (255,  80, 180))
        draw_distance_ui(display, size_l, size_r,
                         lms_left is not None, lms_right is not None)

        if both_ok and dist_ok_since:
            elapsed = now - dist_ok_since
            frac    = min(elapsed / DIST_OK_HOLD, 1.0)
            cv2.rectangle(display, (10, 92),
                          (10 + int(frac * (fw - 20)), 106), (0, 220, 80), -1)
            cv2.putText(display, f"Hold... {DIST_OK_HOLD - elapsed:.1f}s",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)

    elif app_state == 'running':
        desired   = set()
        angle     = 0.0
        steer_dir = 'none'
        accel     = False
        brake     = False

        # ── thumbs-up detection via landmarks ────────────────────────────────
        thumbs_r = is_thumbs_up(lms_right) if lms_right else False
        thumbs_l = is_thumbs_up(lms_left)  if lms_left  else False

        accel = thumbs_r   # right hand thumbs up → accelerate
        brake = thumbs_l   # left  hand thumbs up → brake

        # ── steering from wrist angle ─────────────────────────────────────────
        if lms_left and lms_right:
            angle = get_steer_angle(lms_left, lms_right)
            if angle < -STEER_DEADZONE:  steer_dir = 'left'
            elif angle > STEER_DEADZONE: steer_dir = 'right'

        if steer_dir == 'left':  desired.add('left')
        if steer_dir == 'right': desired.add('right')
        if accel:                desired.add('up')
        if brake:                desired.add('down')

        set_held_keys(desired)

        draw_hand_on_half(display, lms_left,  0,  (255, 180,  80))
        draw_hand_on_half(display, lms_right, hw, (255,  80, 180))

        if lms_left is None or lms_right is None:
            missing = "LEFT" if lms_left is None else "RIGHT"
            cv2.rectangle(display, (0, 0), (fw, 50), (40, 20, 20), -1)
            cv2.putText(display, f"Lost {missing} hand!",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        draw_running_ui(display, angle, steer_dir, accel, brake,
                        fw, fh, lms_left, lms_right, thumbs_l, thumbs_r)

    output = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    cv2.imshow('Car Racing Controller', output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in (ord('r'), ord('R')):
        release_all()
        app_state     = 'distance_check'
        dist_ok_since = None

# ─── Cleanup ──────────────────────────────────────────────────────────────────
release_all()
_stop_threads = True
cap.release()
cv2.destroyAllWindows()
detector_left.close()
detector_right.close()