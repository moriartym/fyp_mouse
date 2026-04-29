import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import time
import threading

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# Tuning Constant
SMOOTHING         = 0.3
PINCH_THRESH      = 0.03
CLICK_COOLDOWN    = 0.5
SCROLL_SPEED      = 3
LONG_PRESS_TIME   = 0.5

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Distant
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

# Gesture State
last_click_t   = 0.0
prev_scroll_y  = None

# Scroll buffer state
scroll_entry_t = None
SCROLL_BUFFER  = 0
scroll_active  = False

# Thumb + middle pinch state
tm_pinch_start = None
tm_drag_active = False

# --- Threading state ---
_latest_frame       = None
_latest_result      = None
_frame_lock         = threading.Lock()
_result_lock        = threading.Lock()
_detection_running  = False
_stop_thread        = False

# Helper Function

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

def map_cursor(tip_x, tip_y):
    span_x = range_max_x - range_min_x
    span_y = range_max_y - range_min_y
    rx = 0.5 if span_x < 0.01 else np.clip((tip_x - range_min_x) / span_x, 0, 1)
    ry = 0.5 if span_y < 0.01 else np.clip((tip_y - range_min_y) / span_y, 0, 1)
    return rx * SCREEN_W, ry * SCREEN_H

def draw_hand(img, lms):
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 200, 100), 2)
    for pt in pts:
        cv2.circle(img, pt, 5, (255, 255, 255), -1)
        cv2.circle(img, pt, 5, (0, 150, 80), 2)

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

# Draw Function

GESTURE_COLOR = {
    'move':        (255, 200,   0),
    'left_click':  (  0, 200, 255),
    'right_click': (255,  80, 200),
    'drag':        (  0, 120, 255),
    'scroll_up':   (180, 255,  80),
    'scroll_down': ( 80, 180, 255),
    'idle':        (180, 180, 180),
}

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

def draw_running_ui(img, gesture, dist_ok, scroll_entry_t, scroll_active,
                    tm_pinch_start, tm_drag_active):
    h, w = img.shape[:2]

    if tm_drag_active:
        label = "DRAG & DROP  [HOLDING]"
        color = GESTURE_COLOR['drag']
    elif gesture == 'tm_pinch' and tm_pinch_start is not None:
        held  = time.time() - tm_pinch_start
        pct   = min(held / LONG_PRESS_TIME, 1.0)
        color = GESTURE_COLOR['left_click']
        label = f"PINCH  {held:.2f}s"
        cv2.rectangle(img, (0, 42), (380, 60), (40, 40, 40), -1)
        fill_w    = int(pct * 380)
        bar_color = (0, int(200 * (1 - pct)), int(255 * pct))
        cv2.rectangle(img, (0, 42), (fill_w, 60), bar_color, -1)
        hint = ">> DRAG MODE" if pct >= 1.0 else f"release = CLICK  |  hold {LONG_PRESS_TIME}s = DRAG"
        cv2.putText(img, hint, (10, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    elif gesture == 'right_click':
        label = "RIGHT CLICK"
        color = GESTURE_COLOR['right_click']
    elif gesture == 'move':
        label = "MOVE"
        color = GESTURE_COLOR['move']
    elif gesture in ('scroll_up', 'scroll_down'):
        label = gesture.upper()
        color = GESTURE_COLOR[gesture]
    else:
        label = "IDLE"
        color = GESTURE_COLOR['idle']

    cv2.rectangle(img, (0, 0), (420, 40), (30, 30, 30), -1)
    cv2.putText(img, f"Gesture: {label}", (10, 28),
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
        ("Index up               = MOVE",                      GESTURE_COLOR['move']),
        (f"Thumb+Middle < {LONG_PRESS_TIME}s  = LEFT CLICK",   GESTURE_COLOR['left_click']),
        (f"Thumb+Middle >= {LONG_PRESS_TIME}s = DRAG & DROP",  GESTURE_COLOR['drag']),
        ("Thumb + Ring           = RIGHT CLICK",                GESTURE_COLOR['right_click']),
        ("Index+Mid+Ring up      = SCROLL UP",                  GESTURE_COLOR['scroll_up']),
        ("All 4 fingers curled   = SCROLL DOWN",                GESTURE_COLOR['scroll_down']),
        ("R = recalibrate",                                      (100, 100, 100)),
    ]
    for i, (text, col) in enumerate(legend):
        cv2.putText(img, text, (10, h - 12 - (len(legend) - 1 - i) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1)

    if not dist_ok:
        cv2.rectangle(img, (0, h - 170), (w, h - 150), (0, 0, 180), -1)
        cv2.putText(img, "Distance changed — accuracy may drift",
                    (10, h - 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Init

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector     = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# --- Latency improvements ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2

# Background detection thread 
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


def reset():
    global app_state, dist_ok_since, calib_start_t, calib_pts_x, calib_pts_y
    global calib_trail, range_min_x, range_max_x, range_min_y, range_max_y
    global smooth_x, smooth_y, calib_done_t, prev_scroll_y, last_click_t
    global scroll_entry_t, scroll_active, tm_pinch_start, tm_drag_active
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
    display  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fh, fw   = display.shape[:2]

    # Distance check
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

    # Calibration Start
    elif app_state == 'calibrating':
        elapsed = time.time() - calib_start_t
        if lms:
            draw_hand(display, lms)
            if is_finger_extended(lms, 8, 6):
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
                range_min_x = min(calib_pts_x)
                range_max_x = max(calib_pts_x)
                range_min_y = min(calib_pts_y)
                range_max_y = max(calib_pts_y)
                app_state    = 'calib_done'
                calib_done_t = time.time()
                print(f"[Calibration done]  "
                      f"x:[{range_min_x:.3f}, {range_max_x:.3f}]  "
                      f"y:[{range_min_y:.3f}, {range_max_y:.3f}]  "
                      f"pts:{len(calib_pts_x)}")

    # Calibration finish
    elif app_state == 'calib_done':
        elapsed_done = time.time() - calib_done_t
        remaining    = max(0.0, 2.0 - elapsed_done)
        if lms:
            draw_hand(display, lms)
        draw_calib_done_ui(display, remaining)
        if elapsed_done >= 2.0:
            app_state = 'running'

    # Running
    elif app_state == 'running':
        gesture = 'idle'

        if range_min_x is not None:
            bx1 = int(range_min_x * fw);  bx2 = int(range_max_x * fw)
            by1 = int(range_min_y * fh);  by2 = int(range_max_y * fh)
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 180), 2)

        if lms:
            gesture = get_gesture(lms)
            dist    = hand_size(lms)
            dist_ok = abs(dist - TARGET_DIST) <= DIST_TOL * 2
            now     = time.time()

            target_x, target_y = map_cursor(lms[8].x, lms[8].y)
            smooth_x += (target_x - smooth_x) * (1 - SMOOTHING)
            smooth_y += (target_y - smooth_y) * (1 - SMOOTHING)

            # Move
            if gesture == 'move':
                handle_pinch_release(now)
                pyautogui.moveTo(int(smooth_x), int(smooth_y))
                scroll_entry_t = None
                scroll_active  = False
                prev_scroll_y  = None

            # Left click + Drag & Drop
            elif gesture == 'tm_pinch':
                if tm_pinch_start is None:
                    tm_pinch_start = now

                held = now - tm_pinch_start

                if not tm_drag_active and held >= LONG_PRESS_TIME:
                    pyautogui.mouseDown()
                    tm_drag_active = True
                    print("[DRAG START]")

                if tm_drag_active:
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))

                scroll_entry_t = None
                scroll_active  = False
                prev_scroll_y  = None

            # Right Click
            elif gesture == 'right_click':
                handle_pinch_release(now)
                if now - last_click_t > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_click_t = now
                    print("[RIGHT CLICK]")
                scroll_entry_t = None
                scroll_active  = False
                prev_scroll_y  = None

            # Scroll Up
            elif gesture == 'scroll_up':
                handle_pinch_release(now)
                if scroll_entry_t is None:
                    scroll_entry_t = now
                    scroll_active  = False
                    print("[SCROLL UP] buffer started")
                elif not scroll_active and (now - scroll_entry_t) >= SCROLL_BUFFER:
                    scroll_active = True
                    print("[SCROLL UP] activated")
                if scroll_active:
                    pyautogui.scroll(SCROLL_SPEED)
                prev_scroll_y = None

            # Scroll Down
            elif gesture == 'scroll_down':
                handle_pinch_release(now)
                if scroll_entry_t is None:
                    scroll_entry_t = now
                    scroll_active  = False
                    print("[SCROLL DOWN] buffer started")
                elif not scroll_active and (now - scroll_entry_t) >= SCROLL_BUFFER:
                    scroll_active = True
                    print("[SCROLL DOWN] activated")
                if scroll_active:
                    pyautogui.scroll(-SCROLL_SPEED)
                prev_scroll_y = None

            # Idle
            else:
                handle_pinch_release(now)
                scroll_entry_t = None
                scroll_active  = False
                prev_scroll_y  = None

            # Fingertip dot
            fx, fy    = int(lms[8].x * fw), int(lms[8].y * fh)
            dot_color = GESTURE_COLOR['drag'] if tm_drag_active else GESTURE_COLOR.get(gesture, (255, 255, 255))
            cv2.circle(display, (fx, fy), 12, dot_color, 2)
            cv2.circle(display, (fx, fy),  4, dot_color, -1)

            draw_hand(display, lms)
            draw_running_ui(display, gesture, dist_ok,
                            scroll_entry_t, scroll_active,
                            tm_pinch_start, tm_drag_active)

        else:
            # Hand lost
            handle_pinch_release(time.time())
            scroll_entry_t = None
            scroll_active  = False
            prev_scroll_y  = None
            draw_running_ui(display, 'idle', True, None, False, None, False)

    # Output
    output = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    cv2.imshow('Virtual Mouse', output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        if tm_drag_active:
            pyautogui.mouseUp()
        break
    elif key in (ord('r'), ord('R')):
        reset()
        print("[INFO] Recalibration triggered.")

# Cleanup
_stop_thread = True
_det_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
detector.close()