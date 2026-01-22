import time
import csv
import os
import cv2
import gc
import random
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
CAMERA_ID = 0
IMG_SIZE = 640
SLOT_SEC = 1.0

# Bandit params
EPSILON = 0.15
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999
ALPHA = 0.15

LOG_CSV = "bandit_light_log.csv"

MODELS = [
    ("NCNN", "/home/dell/venv/best_ncnn_modelfp16"),
    ("INT8", "/home/dell/venv/best_saved_model/best_int8.tflite"),
    ("ONNX", "/home/dell/venv/bestfp32.onnx"),
    ("PT",   "/home/dell/venv/best.pt"),
]

# =========================
# Light metrics
# =========================
ANALYZE_W, ANALYZE_H = 160, 120
DARK_PIXEL_Y = 35
EMA_ALPHA = 0.2

MEAN_BINS = [140, 95, 60]
DARK_BINS = [0.20, 0.45, 0.65]

STABLE_HOLD_SEC = 1.0
LEVELS = ["BRIGHT", "NORMAL", "DIM", "DARK"]

def compute_light_metrics(frame):
    small = cv2.resize(frame, (ANALYZE_W, ANALYZE_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean_y = float(gray.mean())
    dark_ratio = float(np.mean(gray < DARK_PIXEL_Y))
    return mean_y, dark_ratio

def classify_light(mean_y, dark_ratio):
    if mean_y >= MEAN_BINS[0] and dark_ratio <= DARK_BINS[0]:
        return 0
    if mean_y >= MEAN_BINS[1] and dark_ratio <= DARK_BINS[1]:
        return 1
    if mean_y >= MEAN_BINS[2] and dark_ratio <= DARK_BINS[2]:
        return 2
    return 3

# =========================
# Epsilon-greedy bandit
# =========================
class EpsGreedyBandit:
    def __init__(self, action_names, epsilon=0.15, alpha=0.15):
        self.action_names = action_names
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.q = {a: 0.0 for a in action_names}

    def select(self):
        if random.random() < self.epsilon:
            return random.choice(self.action_names)
        return max(self.action_names, key=lambda a: self.q[a])

    def update(self, action, reward):
        self.q[action] = (1 - self.alpha) * self.q[action] + self.alpha * reward

# =========================
# Model cache
# =========================
loaded_models = {}

def get_model(name, path):
    if name in loaded_models:
        return loaded_models[name]
    print(f"[LOAD] {name}")
    model = YOLO(path, task="detect")
    loaded_models[name] = model
    return model

# =========================
# Stable light tracker
# =========================
class StableLightLevel:
    def __init__(self):
        self.ema_mean = None
        self.ema_dark = None
        self.active = None
        self.candidate = None
        self.candidate_since = None

    def update(self, frame):
        mean_y, dark_ratio = compute_light_metrics(frame)

        if self.ema_mean is None:
            self.ema_mean, self.ema_dark = mean_y, dark_ratio
        else:
            self.ema_mean = (1 - EMA_ALPHA) * self.ema_mean + EMA_ALPHA * mean_y
            self.ema_dark = (1 - EMA_ALPHA) * self.ema_dark + EMA_ALPHA * dark_ratio

        detected = classify_light(self.ema_mean, self.ema_dark)
        now = time.time()

        if self.active is None:
            self.active = detected
        elif detected != self.active:
            if self.candidate != detected:
                self.candidate = detected
                self.candidate_since = now
            elif now - self.candidate_since >= STABLE_HOLD_SEC:
                self.active = self.candidate
                self.candidate = None
        else:
            self.candidate = None

        return self.active, self.ema_mean, self.ema_dark

# =========================
# Main
# =========================
def main():
    global EPSILON

    action_map = {n: p for n, p in MODELS}
    action_names = [n for n, _ in MODELS]

    bandits = {lvl: EpsGreedyBandit(action_names, EPSILON, ALPHA) for lvl in LEVELS}

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("No camera frame")

    print("=== Warmup models ===")
    for n in action_names:
        model = get_model(n, action_map[n])
        model.predict(first_frame, imgsz=IMG_SIZE, verbose=False)
        time.sleep(0.2)

    light = StableLightLevel()
    level_idx, ema_mean, ema_dark = light.update(first_frame)
    slot_level = LEVELS[level_idx]
    action = bandits[slot_level].select()

    slot_start = time.time()
    slot_frames = 0
    infer_sum_ms = 0.0

    print("=== RUNNING (ESC to exit) ===")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Light update
        lvl_idx, ema_mean, ema_dark = light.update(frame)
        level_now = LEVELS[lvl_idx]

        # Inference
        model = get_model(action, action_map[action])
        t0 = time.time()
        results = model.predict(frame, imgsz=IMG_SIZE, conf=0.25, verbose=False)
        infer_ms = (time.time() - t0) * 1000.0

        slot_frames += 1
        infer_sum_ms += infer_ms

        # Draw boxes
        annotated = results[0].plot()
        cv2.putText(annotated, f"LEVEL: {level_now}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated, f"MODEL: {action}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated, f"meanY={ema_mean:.1f} dark={ema_dark*100:.1f}%",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Bandit Adaptive YOLO", annotated)

        # Slot end
        now = time.time()
        if now - slot_start >= SLOT_SEC:
            fps = slot_frames / (now - slot_start)
            reward = fps

            bandits[slot_level].update(action, reward)

            EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
            for b in bandits.values():
                b.epsilon = EPSILON

            slot_level = level_now
            action = bandits[slot_level].select()

            slot_start = now
            slot_frames = 0
            infer_sum_ms = 0.0
            gc.collect()

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
