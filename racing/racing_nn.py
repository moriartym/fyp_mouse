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

# ─── Tuning ───────────────────────────────────────────────────────────────────
HAND_SIZE_MIN     = 70
HAND_SIZE_MAX     = 200
DIST_OK_HOLD      = 1.0
STEER_DEADZONE    = 5
STEER_MAX         = 40
CONFIDENCE_THRESH = 0.6

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

# ─── NN Model ─────────────────────────────────────────────────────────────────
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
print(f"[NN] Loaded classes: {list(le.classes_)}")

# ─── State ────────────────────────────────────────────────────────────────────
app_state     = 'distance_check'
dist_ok_since = None
held_keys     = set()

# ── Single detector + single worker thread ────────────────────────────────────
_frame_full   = None
_result_full  = None
_lock_f       = threading.Lock()   # protects _frame_full
_lock_r       = threading.Lock()   # protects _result_full
_stop_threads = False

prev_row_left  = None
prev_row_right = None

# ─── Helpers ──────────────────────────────────────────────────────────────────

def hand_size_px(lms, w, h):
    x0, y0 = lms[0].x * w, lms[0].y * h
    x9, y9 = lms[9].x * w, lms[9].y * h
    return math.sqrt((x9 - x0)**2 + (y9 - y0)**2)

def get_steer_angle(lms_left, lms_right):
    lx = lms_left[0].x
    ly = lms_left[0].y
    rx = lms_right[0].x
    ry = lms_right[0].y
    return np.degrees(np.arctan2(ry - ly, rx - lx))

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

def get_gesture(lms, prev_row):
    features, new_prev = extract_features(lms, prev_row)
    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        probs     = torch.softmax(gesture_model(x), dim=1)[0]
        conf, idx = probs.max(0)
    if conf.item() < CONFIDENCE_THRESH:
        return 'none', conf.item(), new_prev
    return le.inverse_transform([idx.item()])[0], conf.item(), new_prev

def split_hands(result, fw):
    """Split detected hands into left-of-center and right-of-center by wrist x."""
    lms_left  = None
    lms_right = None
    for hand in result.hand_landmarks:
        wrist_x = hand[0].x  # normalized 0→1 across full frame
        if wrist_x < 0.5:
            lms_left  = hand
        else:
            lms_right = hand
    return lms_left, lms_right

def draw_hand(img, lms, color=(0, 200, 100)):
    """Draw hand on the full frame using normalized coords directly."""
    if lms is None:
        return
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
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

# ─── Single MediaPipe detector ────────────────────────────────────────────────

opts     = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=2)                          # detect up to 2 hands in one pass
detector = vision.HandLandmarker.create_from_options(opts)

def worker():
    global _result_full
    while not _stop_threads:
        with _lock_f: frame = _frame_full
        if frame is None:
            time.sleep(0.001)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        with _lock_r:
            _result_full = result

threading.Thread(target=worker, daemon=True).start()

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
                    lms_left, lms_right, gest_l, conf_l, gest_r, conf_r):
    h, w = img.shape[:2]
    hw   = w // 2

    cv2.line(img, (hw, 0), (hw, h), (80, 80, 80), 1)
    cv2.putText(img, "LEFT",  (10,      h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180,  80), 1)
    cv2.putText(img, "RIGHT", (hw + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,  80, 180), 1)

    gl_color = (0, 220, 80) if gest_l == 'thumbs' else (180, 180, 180)
    gr_color = (0, 220, 80) if gest_r == 'thumbs' else (180, 180, 180)
    cv2.putText(img, f"L: {gest_l} [{conf_l:.2f}]",
                (10,      h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gl_color, 1)
    cv2.putText(img, f"R: {gest_r} [{conf_r:.2f}]",
                (hw + 10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gr_color, 1)

    cv2.rectangle(img, (0, 0), (w, 44), (25, 25, 25), -1)
    sc = {'left': (255, 180, 80), 'right': (255, 80, 180), 'none': (180, 180, 180)}[steer_dir]
    sl = {'left': 'LEFT  <', 'right': 'RIGHT  >', 'none': 'STRAIGHT'}[steer_dir]
    cv2.putText(img, f"Steer: {sl}  ({angle:+.1f}deg)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc, 2)

    cv2.rectangle(img, (w - 165, 4), (w - 88, 38), (0, 180, 60) if accel else (50, 50, 50), -1)
    cv2.putText(img, "ACCEL ^", (w - 162, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.rectangle(img, (w - 82,  4), (w - 4,  38), (0,  80, 220) if brake else (50, 50, 50), -1)
    cv2.putText(img, "BRAKE v", (w - 79,  28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if lms_left and lms_right:
        lx = int(lms_left[0].x  * fw); ly = int(lms_left[0].y  * fh)
        rx = int(lms_right[0].x * fw); ry = int(lms_right[0].y * fh)
        cv2.line(img, (lx, ly), (rx, ry), sc, 3)
        cv2.circle(img, (lx, ly), 10, (255, 180,  80), -1)
        cv2.circle(img, (rx, ry), 10, (255,  80, 180), -1)

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

    legend = [
        ("Right hand thumbs = ACCEL",  (  0, 220,  80)),
        ("Left  hand thumbs = BRAKE",  (  0, 120, 255)),
        ("Tilt hands = STEER L / R",   (255, 180,  80)),
        ("R = recalibrate",            (100, 100, 100)),
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

    # ── Feed full frame to the single worker ──────────────────────────────────
    with _lock_f: _frame_full = frame.copy()
    with _lock_r: res = _result_full

    # ── Split result by x position ────────────────────────────────────────────
    lms_left, lms_right = (None, None)
    if res and res.hand_landmarks:
        lms_left, lms_right = split_hands(res, fw)

    if lms_left  is None: prev_row_left  = None
    if lms_right is None: prev_row_right = None

    display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now     = time.time()

    if app_state == 'distance_check':
        # hand_size_px now uses full-frame width
        size_l  = hand_size_px(lms_left,  fw, fh) if lms_left  else 0.0
        size_r  = hand_size_px(lms_right, fw, fh) if lms_right else 0.0
        ok_l    = lms_left  is not None and HAND_SIZE_MIN <= size_l <= HAND_SIZE_MAX
        ok_r    = lms_right is not None and HAND_SIZE_MIN <= size_r <= HAND_SIZE_MAX
        both_ok = ok_l and ok_r

        if both_ok:
            if dist_ok_since is None: dist_ok_since = now
            elif now - dist_ok_since >= DIST_OK_HOLD:
                app_state = 'running'; dist_ok_since = None
        else:
            dist_ok_since = None

        draw_hand(display, lms_left,  (255, 180,  80))
        draw_hand(display, lms_right, (255,  80, 180))
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
        gest_l    = 'none'
        gest_r    = 'none'
        conf_l    = 0.0
        conf_r    = 0.0

        if lms_left:
            gest_l, conf_l, prev_row_left  = get_gesture(lms_left,  prev_row_left)
        if lms_right:
            gest_r, conf_r, prev_row_right = get_gesture(lms_right, prev_row_right)

        accel = gest_r == 'thumbs'
        brake = gest_l == 'thumbs'

        if lms_left and lms_right:
            angle = get_steer_angle(lms_left, lms_right)
            if angle < -STEER_DEADZONE:  steer_dir = 'left'
            elif angle > STEER_DEADZONE: steer_dir = 'right'

        if steer_dir == 'left':  desired.add('left')
        if steer_dir == 'right': desired.add('right')
        if accel:                desired.add('up')
        if brake:                desired.add('down')

        set_held_keys(desired)

        draw_hand(display, lms_left,  (255, 180,  80))
        draw_hand(display, lms_right, (255,  80, 180))

        if lms_left is None or lms_right is None:
            missing = "LEFT" if lms_left is None else "RIGHT"
            cv2.rectangle(display, (0, 0), (fw, 50), (40, 20, 20), -1)
            cv2.putText(display, f"Lost {missing} hand!",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        draw_running_ui(display, angle, steer_dir, accel, brake,
                        fw, fh, lms_left, lms_right,
                        gest_l, conf_l, gest_r, conf_r)

    output = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    cv2.imshow('Car Racing Controller', output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in (ord('r'), ord('R')):
        release_all()
        app_state     = 'distance_check'
        dist_ok_since = None
        prev_row_left = prev_row_right = None

# ─── Cleanup ──────────────────────────────────────────────────────────────────
release_all()
_stop_threads = True
cap.release()
cv2.destroyAllWindows()
detector.close()