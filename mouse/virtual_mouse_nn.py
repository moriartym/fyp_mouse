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
from collections import deque

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# Tuning Constants
SMOOTHING         = 0.12   # lower = smoother but more lag (was 0.3)
CLICK_COOLDOWN    = 0.5
SCROLL_SPEED      = 3
CONFIDENCE_THRESH = 0.75

# Cursor smoothing: average over last N positions to kill jitter
CURSOR_BUFFER_SIZE = 5

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Distance
TARGET_DIST  = 0.18
DIST_TOL     = 0.03
DIST_OK_HOLD = 1.0

# Range Calibration
CALIB_DURATION = 5.0

range_min_x = range_max_x = None
range_min_y = range_max_y = None

# App states
app_state     = 'distance_check'
dist_ok_since = None
calib_start_t = None
calib_pts_x   = []
calib_pts_y   = []
calib_trail   = []
calib_done_t  = None

# Gesture state
last_click_t      = 0.0
prev_scroll_y     = None
left_click_armed  = True
right_click_armed = True

# Scroll buffer state
scroll_entry_t = None
SCROLL_BUFFER  = 0
scroll_active  = False

# Drag & drop state
drag_drop_active = False

# Delta state
prev_row = None

# Threading state
_latest_frame      = None
_latest_result     = None
_frame_lock        = threading.Lock()
_result_lock       = threading.Lock()
_stop_thread       = False

# Cursor buffer for jitter smoothing
cursor_buf_x = deque(maxlen=CURSOR_BUFFER_SIZE)
cursor_buf_y = deque(maxlen=CURSOR_BUFFER_SIZE)

# ── NN Model ──────────────────────────────────────────────────────────────────

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

le            = joblib.load('label_encoder.pkl')
gesture_model = GestureNet(126, len(le.classes_))
gesture_model.load_state_dict(torch.load('gesture_model.pt', map_location='cpu'))
gesture_model.eval()
print(f"[NN] Loaded model  classes: {list(le.classes_)}")

# ── Helper Functions ───────────────────────────────────────────────────────────

def hand_size(lms):
    return np.hypot(lms[0].x - lms[9].x, lms[0].y - lms[9].y)

def normalize_landmarks(lms):
    global prev_row
    wrist_x = lms[0].x
    wrist_y = lms[0].y
    wrist_z = lms[0].z

    scale = math.sqrt((lms[9].x - wrist_x)**2 +
                      (lms[9].y - wrist_y)**2 +
                      (lms[9].z - wrist_z)**2)
    scale = max(scale, 1e-6)

    row = []
    for lm in lms:
        row.extend([(lm.x - wrist_x) / scale,
                    (lm.y - wrist_y) / scale,
                    (lm.z - wrist_z) / scale])

    if prev_row is not None:
        delta = [cur - p for cur, p in zip(row, prev_row)]
    else:
        delta = [0.0] * 63
    prev_row = row
    return row + delta  # 126 features

def get_gesture(lms):
    features = normalize_landmarks(lms)
    x        = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        probs     = torch.softmax(gesture_model(x), dim=1)[0]
        conf, idx = probs.max(0)
    if conf.item() < CONFIDENCE_THRESH:
        return 'idle', conf.item(), True
    return le.inverse_transform([idx.item()])[0], conf.item(), False

def map_cursor(tip_x, tip_y):
    span_x = range_max_x - range_min_x
    span_y = range_max_y - range_min_y
    rx = 0.5 if span_x < 0.01 else np.clip((tip_x - range_min_x) / span_x, 0, 1)
    ry = 0.5 if span_y < 0.01 else np.clip((tip_y - range_min_y) / span_y, 0, 1)
    return rx * SCREEN_W, ry * SCREEN_H

def smooth_cursor(raw_x, raw_y):
    """EMA + rolling average buffer — kills jitter while staying responsive."""
    cursor_buf_x.append(raw_x)
    cursor_buf_y.append(raw_y)
    return float(np.mean(cursor_buf_x)), float(np.mean(cursor_buf_y))

def draw_hand(img, lms):
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 200, 100), 2)
    for pt in pts:
        cv2.circle(img, pt, 5, (255, 255, 255), -1)
        cv2.circle(img, pt, 5, (0, 150, 80), 2)

def release_drag():
    global drag_drop_active
    if drag_drop_active:
        pyautogui.mouseUp()
        drag_drop_active = False
        print("[DRAG END → DROP]")

# ── Gesture Colors ─────────────────────────────────────────────────────────────

GESTURE_COLOR = {
    'move':            (255, 200,   0),
    'pre_left_click':  (  0, 200, 200),
    'left_click':      (  0, 200, 255),
    'pre_drag_drop':   (  0,  80, 200),
    'drag_drop':       (  0, 120, 255),
    'pre_right_click': (200,  50, 180),
    'right_click':     (255,  80, 200),
    'scroll_up':       (180, 255,  80),
    'scroll_down':     ( 80, 180, 255),
    'idle':            (180, 180, 180),
}

# ── Draw Functions ─────────────────────────────────────────────────────────────

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
        msg, color = "Move CLOSER to camera", (0, 200, 255)
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

def draw_calibration_ui(img, elapsed, total, trail, has_hand, calib_pts_x, calib_pts_y):
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
    countdown_text = f"{remaining:.1f}s"
    font_scale, thickness = 3.0, 5
    (tw, th), _ = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cx, cy = (w - tw) // 2, h // 2 + th // 2
    cv2.putText(img, countdown_text, (cx + 3, cy + 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
    green = int(255 * frac)
    cv2.putText(img, countdown_text, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, green, 255 - green), thickness)
    if len(calib_pts_x) >= 2:
        bx1 = int(min(calib_pts_x) * w);  bx2 = int(max(calib_pts_x) * w)
        by1 = int(min(calib_pts_y) * h);  by2 = int(max(calib_pts_y) * h)
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

def draw_running_ui(img, gesture, conf, dist_ok, scroll_entry_t, scroll_active, drag_drop_active):
    h, w = img.shape[:2]

    if drag_drop_active:
        label, color = "DRAG & DROP  [HOLDING]", GESTURE_COLOR['drag_drop']
    elif gesture == 'pre_left_click':
        label, color = "PRE LEFT CLICK",  GESTURE_COLOR['pre_left_click']
    elif gesture == 'left_click':
        label, color = "LEFT CLICK",      GESTURE_COLOR['left_click']
    elif gesture == 'pre_drag_drop':
        label, color = "PRE DRAG DROP",   GESTURE_COLOR['pre_drag_drop']
    elif gesture == 'drag_drop':
        label, color = "DRAG & DROP",     GESTURE_COLOR['drag_drop']
    elif gesture == 'pre_right_click':
        label, color = "PRE RIGHT CLICK", GESTURE_COLOR['pre_right_click']
    elif gesture == 'right_click':
        label, color = "RIGHT CLICK",     GESTURE_COLOR['right_click']
    elif gesture == 'move':
        label, color = "MOVE",            GESTURE_COLOR['move']
    elif gesture in ('scroll_up', 'scroll_down'):
        label, color = gesture.upper(),   GESTURE_COLOR[gesture]
    else:
        label, color = "IDLE",            GESTURE_COLOR['idle']

    cv2.rectangle(img, (0, 0), (460, 40), (30, 30, 30), -1)
    cv2.putText(img, f"Gesture: {label}  [{conf:.2f}]", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if gesture in ('scroll_up', 'scroll_down') and scroll_entry_t is not None and not scroll_active:
        elapsed   = time.time() - scroll_entry_t
        remaining = max(0.0, SCROLL_BUFFER - elapsed)
        cv2.rectangle(img, (0, 42), (320, 72), (40, 40, 20), -1)
        cv2.putText(img, f"Scroll locks in {remaining:.1f}s...", (10, 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    elif gesture in ('scroll_up', 'scroll_down') and scroll_active:
        direction = "UP  ▲" if gesture == 'scroll_up' else "DOWN  ▼"
        cv2.rectangle(img, (0, 42), (280, 72), (20, 50, 20), -1)
        cv2.putText(img, f"Scrolling {direction}", (10, 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2)

    legend = [
        ("Index up           = MOVE",         GESTURE_COLOR['move']),
        ("Pre left click      = STANDBY",      GESTURE_COLOR['pre_left_click']),
        ("Left click gesture  = LEFT CLICK",   GESTURE_COLOR['left_click']),
        ("Pre drag drop       = STANDBY",      GESTURE_COLOR['pre_drag_drop']),
        ("Drag gesture        = DRAG & DROP",  GESTURE_COLOR['drag_drop']),
        ("Pre right click     = STANDBY",      GESTURE_COLOR['pre_right_click']),
        ("Right click gesture = RIGHT CLICK",  GESTURE_COLOR['right_click']),
        ("Scroll up gesture   = SCROLL UP",    GESTURE_COLOR['scroll_up']),
        ("Scroll down gesture = SCROLL DOWN",  GESTURE_COLOR['scroll_down']),
        ("R = recalibrate",                    (100, 100, 100)),
    ]
    for i, (text, col) in enumerate(legend):
        cv2.putText(img, text, (10, h - 12 - (len(legend) - 1 - i) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1)

    if not dist_ok:
        cv2.rectangle(img, (0, h - 170), (w, h - 150), (0, 0, 180), -1)
        cv2.putText(img, "Distance changed — accuracy may drift",
                    (10, h - 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ── Init ───────────────────────────────────────────────────────────────────────

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector     = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2

# ── Detection thread ───────────────────────────────────────────────────────────

def detection_worker():
    global _latest_result, _stop_thread
    while not _stop_thread:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.001)
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = detector.detect(mp_image)
        with _result_lock:
            _latest_result = result

_det_thread = threading.Thread(target=detection_worker, daemon=True)
_det_thread.start()

# ── Reset ──────────────────────────────────────────────────────────────────────

def reset():
    global app_state, dist_ok_since, calib_start_t, calib_pts_x, calib_pts_y
    global calib_trail, range_min_x, range_max_x, range_min_y, range_max_y
    global smooth_x, smooth_y, calib_done_t, prev_scroll_y, last_click_t
    global scroll_entry_t, scroll_active, drag_drop_active
    global left_click_armed, right_click_armed, prev_row
    release_drag()
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
    left_click_armed   = True
    right_click_armed  = True
    prev_row           = None
    cursor_buf_x.clear()
    cursor_buf_y.clear()

# ── Main Loop ──────────────────────────────────────────────────────────────────

print("ESC to quit  |  R to recalibrate")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # feed latest frame to detection thread
    with _frame_lock:
        _latest_frame = frame.copy()

    # grab latest result (non-blocking)
    with _result_lock:
        result = _latest_result

    has_hand = bool(result and result.hand_landmarks)
    lms      = result.hand_landmarks[0] if has_hand else None
    display  = frame.copy()
    fh, fw   = display.shape[:2]

    if not has_hand:
        prev_row = None
        cursor_buf_x.clear()
        cursor_buf_y.clear()

    if app_state == 'distance_check':
        dist     = hand_size(lms) if lms else 0.0
        in_range = lms and abs(dist - TARGET_DIST) <= DIST_TOL
        if in_range:
            if dist_ok_since is None:
                dist_ok_since = time.time()
            elif time.time() - dist_ok_since >= DIST_OK_HOLD:
                app_state     = 'calibrating'
                calib_start_t = time.time()
                dist_ok_since = None
        else:
            dist_ok_since = None
        if lms:
            draw_hand(display, lms)
        draw_distance_ui(display, dist, has_hand)

    elif app_state == 'calibrating':
        elapsed = time.time() - calib_start_t
        if lms:
            draw_hand(display, lms)
            tx, ty = lms[8].x, lms[8].y
            calib_pts_x.append(tx)
            calib_pts_y.append(ty)
            trail_px = (int(tx * fw), int(ty * fh))
            calib_trail.append(trail_px)
            if len(calib_trail) > 300:
                calib_trail.pop(0)
            dist = hand_size(lms)
            if abs(dist - TARGET_DIST) > DIST_TOL * 2:
                cv2.putText(display, "Distance drifted — adjust!",
                            (10, fh - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
        draw_calibration_ui(display, elapsed, CALIB_DURATION,
                            calib_trail, has_hand, calib_pts_x, calib_pts_y)
        if elapsed >= CALIB_DURATION:
            if len(calib_pts_x) < 10:
                calib_start_t = time.time()
                calib_pts_x   = []
                calib_pts_y   = []
                calib_trail   = []
            else:
                range_min_x  = min(calib_pts_x)
                range_max_x  = max(calib_pts_x)
                range_min_y  = min(calib_pts_y)
                range_max_y  = max(calib_pts_y)
                app_state    = 'calib_done'
                calib_done_t = time.time()

    elif app_state == 'calib_done':
        elapsed_done = time.time() - calib_done_t
        remaining    = max(0.0, 2.0 - elapsed_done)
        if lms:
            draw_hand(display, lms)
        draw_calib_done_ui(display, remaining)
        if elapsed_done >= 2.0:
            app_state = 'running'

    elif app_state == 'running':
        gesture = 'idle'
        conf    = 0.0

        if range_min_x is not None:
            bx1 = int(range_min_x * fw);  bx2 = int(range_max_x * fw)
            by1 = int(range_min_y * fh);  by2 = int(range_max_y * fh)
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 180), 2)

        if lms:
            gesture, conf, low_conf = get_gesture(lms)
            dist    = hand_size(lms)
            dist_ok = abs(dist - TARGET_DIST) <= DIST_TOL * 2
            now     = time.time()

            # map raw tip to screen coords
            raw_x, raw_y = map_cursor(lms[8].x, lms[8].y)

            # rolling average buffer kills high-freq jitter
            avg_x, avg_y = smooth_cursor(raw_x, raw_y)

            # EMA on top of the buffer for extra smoothness
            smooth_x += (avg_x - smooth_x) * (1 - SMOOTHING)
            smooth_y += (avg_y - smooth_y) * (1 - SMOOTHING)

            confident_idle = (gesture == 'idle' and not low_conf)
            if not confident_idle:
                pyautogui.moveTo(int(smooth_x), int(smooth_y))

            if gesture == 'move':
                release_drag()
                left_click_armed  = True
                right_click_armed = True
                scroll_entry_t    = None
                scroll_active     = False
                prev_scroll_y     = None

            elif gesture in ('pre_left_click', 'pre_drag_drop', 'pre_right_click'):
                pass

            elif gesture == 'left_click':
                release_drag()
                if left_click_armed and now - last_click_t > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click_t     = now
                    left_click_armed = False
                    print("[LEFT CLICK]")
                scroll_entry_t = None
                scroll_active  = False
                prev_scroll_y  = None

            elif gesture == 'drag_drop':
                if not drag_drop_active:
                    pyautogui.mouseDown()
                    drag_drop_active = True
                    print("[DRAG START]")
                left_click_armed  = True
                right_click_armed = True
                scroll_entry_t    = None
                scroll_active     = False
                prev_scroll_y     = None

            elif gesture == 'right_click':
                release_drag()
                if right_click_armed and now - last_click_t > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_click_t      = now
                    right_click_armed = False
                    print("[RIGHT CLICK]")
                scroll_entry_t = None
                scroll_active  = False
                prev_scroll_y  = None

            elif gesture == 'scroll_up':
                release_drag()
                left_click_armed  = True
                right_click_armed = True
                if scroll_entry_t is None:
                    scroll_entry_t = now
                    scroll_active  = False
                elif not scroll_active and (now - scroll_entry_t) >= SCROLL_BUFFER:
                    scroll_active = True
                if scroll_active:
                    pyautogui.scroll(SCROLL_SPEED)
                prev_scroll_y = None

            elif gesture == 'scroll_down':
                release_drag()
                left_click_armed  = True
                right_click_armed = True
                if scroll_entry_t is None:
                    scroll_entry_t = now
                    scroll_active  = False
                elif not scroll_active and (now - scroll_entry_t) >= SCROLL_BUFFER:
                    scroll_active = True
                if scroll_active:
                    pyautogui.scroll(-SCROLL_SPEED)
                prev_scroll_y = None

            else:
                release_drag()
                left_click_armed  = True
                right_click_armed = True
                scroll_entry_t    = None
                scroll_active     = False
                prev_scroll_y     = None

            fx, fy    = int(lms[8].x * fw), int(lms[8].y * fh)
            dot_color = GESTURE_COLOR['drag_drop'] if drag_drop_active else GESTURE_COLOR.get(gesture, (255, 255, 255))
            cv2.circle(display, (fx, fy), 12, dot_color, 2)
            cv2.circle(display, (fx, fy),  4, dot_color, -1)

            draw_hand(display, lms)
            draw_running_ui(display, gesture, conf, dist_ok,
                            scroll_entry_t, scroll_active, drag_drop_active)
        else:
            release_drag()
            scroll_entry_t = None
            scroll_active  = False
            prev_scroll_y  = None
            draw_running_ui(display, 'idle', 0.0, True, None, False, False)

    cv2.imshow('Virtual Mouse', display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        release_drag()
        break
    elif key in (ord('r'), ord('R')):
        reset()
        print("[INFO] Recalibration triggered.")

_stop_thread = True
_det_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
detector.close()