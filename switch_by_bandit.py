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
# Light metrics (from your code)
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
        self.n = {a: 0 for a in action_names}

    def select(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.action_names)
        return max(self.action_names, key=lambda a: self.q[a])

    def update(self, action: str, reward: float):
        self.n[action] += 1
        self.q[action] = (1 - self.alpha) * self.q[action] + self.alpha * reward

# =========================
# Model cache + warmup
# =========================
loaded_models = {}  # name -> YOLO object

def get_model(name: str, path: str):
    if name in loaded_models:
        return loaded_models[name]
    print(f"[LOAD] {name}: {path}")
    m = YOLO(path, task="detect")
    loaded_models[name] = m
    return m

def warmup_model(model, frame):
    _ = model.predict(source=frame, imgsz=IMG_SIZE, conf=0.25, verbose=False)

# =========================
# Stable light level tracker
# =========================
class StableLightLevel:
    def __init__(self):
        self.ema_mean = None
        self.ema_dark = None
        self.active_level = None
        self.candidate_level = None
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

        if self.active_level is None:
            self.active_level = detected
        elif detected != self.active_level:
            if self.candidate_level != detected:
                self.candidate_level = detected
                self.candidate_since = now
            elif now - self.candidate_since >= STABLE_HOLD_SEC:
                self.active_level = self.candidate_level
                self.candidate_level = None
                self.candidate_since = None
        else:
            self.candidate_level = None
            self.candidate_since = None

        return self.active_level, self.ema_mean, self.ema_dark

# =========================
# Main
# =========================
def main():
    global EPSILON

    action_map = {n: p for n, p in MODELS}
    action_names = [n for n, _ in MODELS]

    # 4 bandits for 4 light levels
    bandits = {lvl: EpsGreedyBandit(action_names, epsilon=EPSILON, alpha=ALPHA)
               for lvl in LEVELS}

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Grab first frame for warmup
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("No frames from camera")

    print("=== Preload & warmup models (stagger) ===")
    for n in action_names:
        m = get_model(n, action_map[n])
        warmup_model(m, first_frame)
        time.sleep(0.2)
    print("=== Preload done ===\n")

    # CSV log
    new_file = not os.path.exists(LOG_CSV)
    fcsv = open(LOG_CSV, "a", newline="")
    writer = csv.writer(fcsv)
    if new_file:
        writer.writerow([
            "ts",
            "slot_level", "meanY_start", "darkRatio_start",
            "level_now", "meanY_now", "darkRatio_now",
            "action", "slot_frames", "slot_sec",
            "infer_avg_ms", "slot_fps", "reward",
            "epsilon"
        ])

    light = StableLightLevel()

    # init state/action from first_frame
    level0, mean0, dark0 = light.update(first_frame)
    slot_level = LEVELS[level0]
    action = bandits[slot_level].select()

    print("=== Bandit switching on CAMERA (1s slots) ===")
    print(f"Actions: {action_names}")
    print(f"Start level={slot_level} -> action={action}")
    print(f"Logging to: {LOG_CSV}")
    print("Press ESC to stop.\n")

    slot_start = time.time()
    slot_frames = 0
    infer_sum_ms = 0.0

    slot_mean_start = float(mean0)
    slot_dark_start = float(dark0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # update current light level (EMA + stable hold)
            lvl_idx, ema_mean, ema_dark = light.update(frame)
            level_now = LEVELS[lvl_idx]

            # run inference with current slot action
            model = get_model(action, action_map[action])
            t0 = time.time()
            _ = model.predict(source=frame, imgsz=IMG_SIZE, conf=0.25, verbose=False)
            infer_ms = (time.time() - t0) * 1000.0

            slot_frames += 1
            infer_sum_ms += infer_ms

            # slot boundary
            now = time.time()
            if now - slot_start >= SLOT_SEC:
                slot_time = now - slot_start
                slot_fps = slot_frames / slot_time if slot_time > 0 else 0.0
                infer_avg = infer_sum_ms / slot_frames if slot_frames > 0 else 0.0

                # reward (simple: maximize fps)
                reward = slot_fps

                # update bandit of the level where this SLOT was running
                bandits[slot_level].update(action, reward)

                # log
                writer.writerow([
                    int(now),
                    slot_level, f"{slot_mean_start:.2f}", f"{slot_dark_start:.4f}",
                    level_now, f"{ema_mean:.2f}", f"{ema_dark:.4f}",
                    action, slot_frames, f"{slot_time:.3f}",
                    f"{infer_avg:.2f}", f"{slot_fps:.2f}", f"{reward:.3f}",
                    f"{EPSILON:.3f}"
                ])
                fcsv.flush()

                q = bandits[slot_level].q
                q_str = " ".join([f"{k}={q[k]:.2f}" for k in action_names])
                print(f"[SLOT] slot_level={slot_level} -> action={action} "
                      f"fps={slot_fps:.2f} infer_avg={infer_avg:.1f}ms reward={reward:.3f} | Q(slot): {q_str}")

                # decay epsilon
                EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
                for b in bandits.values():
                    b.epsilon = EPSILON

                # choose next action based on current stable level
                slot_level = level_now
                slot_mean_start = float(ema_mean)
                slot_dark_start = float(ema_dark)
                action = bandits[slot_level].select()
                print(f"[NEXT] level={slot_level} epsilon={EPSILON:.3f} -> next_action={action}\n")

                # reset slot counters
                slot_start = time.time()
                slot_frames = 0
                infer_sum_ms = 0.0
                gc.collect()

            # Optional display (lightweight)
            # Press ESC to exit
            vis = frame.copy()
            cv2.putText(vis, f"LEVEL: {level_now}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(vis, f"ACTION: {action}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(vis, f"meanY={ema_mean:.1f} dark={ema_dark*100:.1f}%", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Bandit Adaptive by Light", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        fcsv.close()
        print("Done.")

if __name__ == "__main__":
    main()
