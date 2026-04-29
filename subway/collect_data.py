import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv, os, shutil, time
import math

GESTURES = [
    'up',             # 0
    'down',   # 1
    'right',       # 2
    'left',    # 3
    'space',    # 4
    'neutral',        # 5
    'none',  # 6
]

OUTPUT_FILE     = 'gesture_data.csv'
VIDEO_DIR       = 'gesture_videos'
TARGET          = 500
MIN_RECORD_DIST = 0.015
HAND_SIZE_MIN   = 70
HAND_SIZE_MAX   = 200

os.makedirs(VIDEO_DIR, exist_ok=True)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector     = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

FRAME_W, FRAME_H = 640, 480
FPS = 20

current_label  = None
collecting     = False
prev_row       = None
last_saved_pos = None
confirm_delete = False
confirm_timer  = 0.0
CONFIRM_WINDOW = 3.0

video_writer = None
video_path   = None

HEADER = (
    [f'{ax}{i}' for i in range(21) for ax in ['x', 'y', 'z']] +
    [f'd{ax}{i}' for i in range(21) for ax in ['x', 'y', 'z']] +
    ['label']
)

counts = {g: 0 for g in GESTURES}
file_exists = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0
if file_exists:
    with open(OUTPUT_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['label'] in counts:
                counts[row['label']] += 1

print("0-9 select  |  SPACE record+video  |  D delete selected  |  ESC quit")
for i, g in enumerate(GESTURES):
    print(f"  {i} = {g}  ({counts[g]} samples)")

# ── helpers ────────────────────────────────────────────────────────────────────

def draw_landmarks(frame, lms, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 1, cv2.LINE_AA)
    for idx, (px, py) in enumerate(pts):
        r = 5 if idx in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, (px, py), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), r, (0, 150, 255),  1,  cv2.LINE_AA)

def wrist_dist(lms, last_pos):
    if last_pos is None:
        return float('inf')
    dx = lms[0].x - last_pos[0]
    dy = lms[0].y - last_pos[1]
    return math.sqrt(dx*dx + dy*dy)

def hand_size_px(lms, w, h):
    x0, y0 = lms[0].x * w, lms[0].y * h
    x9, y9 = lms[9].x * w, lms[9].y * h
    return math.sqrt((x9-x0)**2 + (y9-y0)**2)

def delete_gesture(label):
    if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
        return 0
    removed  = 0
    tmp_path = OUTPUT_FILE + '.tmp'
    with open(OUTPUT_FILE, 'r', newline='') as fin, \
         open(tmp_path,   'w', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['label'] == label:
                removed += 1
            else:
                writer.writerow(row)
    shutil.move(tmp_path, OUTPUT_FILE)
    return removed

def open_csv():
    exists = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0
    fh = open(OUTPUT_FILE, 'a', newline='')
    cw = csv.writer(fh)
    if not exists:
        cw.writerow(HEADER)
    return fh, cw

def start_video(label):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(VIDEO_DIR, f"{label}_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(path, fourcc, FPS, (FRAME_W, FRAME_H))
    return vw, path

def stop_video(vw, path):
    if vw is not None:
        vw.release()
        print(f"  Video saved → {path}")

# ── open CSV ───────────────────────────────────────────────────────────────────
csv_file, csv_writer = open_csv()

# ── main loop ──────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)
    h, w     = frame.shape[:2]
    now      = time.time()

    if confirm_delete and (now - confirm_timer) > CONFIRM_WINDOW:
        confirm_delete = False

    # ── landmark detection & feature extraction (before any drawing) ───────────
    features   = None
    in_range   = False
    moved      = False
    can_save   = False
    size       = 0

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]

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

        delta    = [cur - p for cur, p in zip(row, prev_row)] if prev_row else [0.0] * 63
        prev_row = row
        features = row + delta

        size     = hand_size_px(lms, w, h)
        in_range = HAND_SIZE_MIN <= size <= HAND_SIZE_MAX
        dist     = wrist_dist(lms, last_saved_pos)
        moved    = dist == float('inf') or dist >= MIN_RECORD_DIST
        can_save = in_range and moved

        # save data
        if collecting and current_label is not None and can_save:
            csv_writer.writerow(features + [GESTURES[current_label]])
            csv_file.flush()
            counts[GESTURES[current_label]] += 1
            last_saved_pos = (lms[0].x, lms[0].y)
    else:
        prev_row       = None
        last_saved_pos = None

    # ── write CLEAN frame to video (no UI, landmarks only) ────────────────────
    if collecting and video_writer is not None:
        clean = frame.copy()
        if result.hand_landmarks:
            draw_landmarks(clean, result.hand_landmarks[0], w, h)
        video_writer.write(clean)

    # ── now draw all UI on top for display only ────────────────────────────────
    display = frame.copy()

    # legend
    for i, g in enumerate(GESTURES):
        done   = counts[g] >= TARGET
        color  = (0, 255, 0) if done else (180, 180, 180)
        tag    = "OK" if done else f"{counts[g]}/{TARGET}"
        prefix = ">> " if current_label == i else "   "
        cv2.putText(display, f"{prefix}{i}:{g}  {tag}",
                    (w - 310, 30 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # confirm-delete banner
    if confirm_delete and current_label is not None:
        remaining = max(0.0, CONFIRM_WINDOW - (now - confirm_timer))
        msg = f"Press D again to DELETE all '{GESTURES[current_label]}' data  ({remaining:.1f}s)"
        cv2.rectangle(display, (0, h - 80), (w, h - 55), (0, 0, 160), -1)
        cv2.putText(display, msg, (10, h - 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        draw_landmarks(display, lms, w, h)

        # depth bar
        bar_max = 200
        bar_h   = 14
        fill    = int(min(size / (HAND_SIZE_MAX * 1.2), 1.0) * bar_max)
        cv2.rectangle(display, (10, h-50), (10+bar_max, h-50+bar_h), (40,40,40), -1)
        g_s = int(HAND_SIZE_MIN / (HAND_SIZE_MAX * 1.2) * bar_max)
        g_e = int(HAND_SIZE_MAX / (HAND_SIZE_MAX * 1.2) * bar_max)
        cv2.rectangle(display, (10+g_s, h-50), (10+g_e, h-50+bar_h), (0,80,0), -1)
        cv2.rectangle(display, (10, h-50), (10+fill, h-50+bar_h),
                      (0,220,80) if in_range else (0,80,220), -1)
        cv2.rectangle(display, (10, h-50), (10+bar_max, h-50+bar_h), (120,120,120), 1)

        if in_range:
            hint, hc = f"distance OK  (size {size:.0f}px)", (0,220,80)
        elif size < HAND_SIZE_MIN:
            hint, hc = f"move CLOSER  (size {size:.0f} < {HAND_SIZE_MIN}px)", (0,140,255)
        else:
            hint, hc = f"move FURTHER (size {size:.0f} > {HAND_SIZE_MAX}px)", (0,80,255)
        cv2.putText(display, hint, (10, h-56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, hc, 1)

        if collecting and current_label is not None:
            label = GESTURES[current_label]
            bar_w = int((counts[label] / TARGET) * 300)
            cv2.rectangle(display, (10, h-30), (310, h-10), (60,60,60), -1)
            cv2.rectangle(display, (10, h-30), (10+bar_w, h-10), (0,0,255), -1)
            gate_msg = ("  [bad distance]" if not in_range else
                        "  [move a bit]"   if not moved    else "")
            cv2.putText(display, f"RECORDING: {label}  [{counts[label]}/{TARGET}]{gate_msg}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
            # REC dot
            cv2.circle(display, (w - 20, 20), 8, (0, 0, 255), -1)
            cv2.putText(display, "REC", (w - 50, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            lname = GESTURES[current_label] if current_label is not None else 'none selected'
            cv2.putText(display, f"Ready | {lname}  (SPACE=record  D=delete)",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
    else:
        cv2.putText(display, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,80,255), 2)

    cv2.imshow('Collect Gestures', display)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        if collecting and video_writer is not None:
            stop_video(video_writer, video_path)
            video_writer = None
        break

    elif key == ord(' '):
        if current_label is None:
            print("Select a gesture first (0-9)")
        else:
            collecting     = not collecting
            prev_row       = None
            last_saved_pos = None
            confirm_delete = False
            if collecting:
                video_writer, video_path = start_video(GESTURES[current_label])
                print(f"Recording started → {video_path}")
            else:
                stop_video(video_writer, video_path)
                video_writer = None
                video_path   = None
                print(f"Recording stopped. '{GESTURES[current_label]}' now has {counts[GESTURES[current_label]]} samples.")

    elif key in (ord('d'), ord('D')):
        if current_label is None:
            print("Select a gesture first (0-9)")
        elif not confirm_delete:
            confirm_delete = True
            confirm_timer  = now
            if collecting:
                collecting = False
                stop_video(video_writer, video_path)
                video_writer = None
                video_path   = None
            print(f"Press D again within {CONFIRM_WINDOW:.0f}s to DELETE '{GESTURES[current_label]}'")
        else:
            label_to_del = GESTURES[current_label]
            collecting   = False
            if video_writer is not None:
                stop_video(video_writer, video_path)
                video_writer = None
                video_path   = None
            csv_file.flush()
            csv_file.close()
            removed = delete_gesture(label_to_del)
            counts[label_to_del] = 0
            confirm_delete = False
            print(f"Deleted {removed} rows for '{label_to_del}'")
            csv_file, csv_writer = open_csv()
            prev_row       = None
            last_saved_pos = None

    elif ord('0') <= key <= ord('6'):
        if collecting:
            collecting = False
            stop_video(video_writer, video_path)
            video_writer = None
            video_path   = None
        current_label  = key - ord('0')
        collecting     = False
        prev_row       = None
        last_saved_pos = None
        confirm_delete = False
        print(f"Selected: {GESTURES[current_label]}  ({counts[GESTURES[current_label]]} samples so far)")

# ── cleanup ────────────────────────────────────────────────────────────────────
if video_writer is not None:
    stop_video(video_writer, video_path)
csv_file.close()
cap.release()
cv2.destroyAllWindows()
detector.close()

print("\nFinal counts:")
for g, c in counts.items():
    status = "OK" if c >= TARGET else f"NEED {TARGET - c} more"
    print(f"  {g}: {c}  [{status}]")

print(f"\nVideos saved in: {VIDEO_DIR}/")