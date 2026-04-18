# Virtual Mouse 🖱️
Control your cursor with hand gestures using your webcam and MediaPipe.

---

## Setup

### Option A — Using Conda (recommended)

**1. Create a conda environment with Python 3.11**
```bash
conda create -n virtual-mouse python=3.11
conda activate virtual-mouse
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

### Option B — Using Python venv (no Conda)

**1. Make sure you have Python 3.11 installed**
Download from [python.org](https://www.python.org/downloads/) if needed.

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the MediaPipe hand landmark model**

Download `hand_landmarker.task` from the [MediaPipe releases page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and place it in the **same folder** as `virtual_mouse.py`.

Your folder should look like:
```
virtual-mouse/
├── virtual_mouse.py
├── hand_landmarker.task
└── requirements.txt
```

---

## How It Works
1. **Distance Check** — Hold your hand at arm's length until the bar turns green (1 second).
2. **Calibration (5s)** — Raise only your index finger and sweep across your full reach area to map your hand range to the screen.
3. **Running** — Your index fingertip drives the cursor. Gestures trigger mouse actions in real time. Press `R` to recalibrate anytime.

---

## Gestures
| Gesture | Action |
|---|---|
| Index finger up, others curled | Move cursor |
| Thumb + Middle pinch, release **< 0.5s** | Left click |
| Thumb + Middle pinch, hold **≥ 0.5s** | Drag & drop (release to drop) |
| Thumb + Ring pinch | Right click |
| Index + Middle + Ring up, pinky down | Scroll up |
| All 4 fingers curled (fist) | Scroll down |

> Scroll gestures have a 1-second buffer before activating to avoid accidental triggers.

---

## Run
```bash
python virtual_mouse.py
```

| Key | Action |
|---|---|
| `R` | Recalibrate |
| `Esc` | Quit |