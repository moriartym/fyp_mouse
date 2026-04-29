import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import joblib
import math

CONFIDENCE_THRESH = 0.5

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ─── Model ────────────────────────────────────────────────────────────────────
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

le            = joblib.load('label_encoder_movement.pkl')
gesture_model = GestureNet(126, len(le.classes_))
gesture_model.load_state_dict(torch.load('gesture_movement.pt', map_location='cpu'))
gesture_model.eval()

print(f"Classes: {list(le.classes_)}")

prev_row = None

# ─── Helpers ──────────────────────────────────────────────────────────────────
def normalize_landmarks(lms):
    global prev_row
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
    return row + delta   # 126 features

def get_all_probs(lms):
    features = normalize_landmarks(lms)
    x        = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(gesture_model(x), dim=1)[0]
    return probs.numpy()

def draw_hand(img, lms):
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 200, 100), 2)
    for pt in pts:
        cv2.circle(img, pt, 5, (255, 255, 255), -1)
        cv2.circle(img, pt, 5, (0, 150, 80), 2)

# ─── MediaPipe ────────────────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector     = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

print("ESC to quit  |  watching gestures...")

# ─── Main loop ────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    h, w   = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)

    if result.hand_landmarks:
        lms   = result.hand_landmarks[0]
        probs = get_all_probs(lms)

        best_idx   = int(np.argmax(probs))
        best_label = le.classes_[best_idx]
        best_conf  = probs[best_idx]
        final      = best_label if best_conf >= CONFIDENCE_THRESH else 'idle'

        color = (0, 255, 0) if final != 'idle' else (180, 180, 180)
        cv2.putText(frame, f"{final.upper()}  [{best_conf:.2f}]",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        bar_x  = 10
        bar_h  = 22
        bar_mw = 200
        for i, (label, prob) in enumerate(zip(le.classes_, probs)):
            y      = 55 + i * (bar_h + 6)
            filled = int(prob * bar_mw)
            is_top = (i == best_idx)
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_mw, y + bar_h), (50, 50, 50), -1)
            bar_color = (0, 220, 80) if is_top else (80, 140, 200)
            cv2.rectangle(frame, (bar_x, y), (bar_x + filled, y + bar_h), bar_color, -1)
            cv2.putText(frame, f"{label}: {prob:.2f}",
                        (bar_x + bar_mw + 8, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        draw_hand(frame, lms)

    else:
        prev_row = None
        cv2.putText(frame, "No hand detected", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2)

    cv2.imshow('Movement Gesture Test', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ─── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
detector.close()