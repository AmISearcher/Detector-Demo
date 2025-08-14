#!/usr/bin/env python3
import degirum as dg
import cv2
import numpy as np
import time
import sys, termios, tty, select
from picamera2 import Picamera2, Preview

# --- NEW: norfair imports ---
from norfair import Detection, Tracker

# ===== Parameters =====
TILE_WIDTH, TILE_HEIGHT = 480, 480
OVERLAP_X, OVERLAP_Y = 0.2, 0.2
STRIDE_X = int(TILE_WIDTH * (1 - OVERLAP_X))
STRIDE_Y = int(TILE_HEIGHT * (1 - OVERLAP_Y))
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
REDETECT_INTERVAL = 3.0  # seconds (set 0.4 if you want more frequent detections)

# Norfair params (tune as needed)
DISTANCE_THRESHOLD = 35        # pixels; how close a detection must be to a track
HITS_TO_START = 1              # detections before a track is "confirmed"
INITIALIZATION_DELAY = 0
MAX_AGE = 10                   # frames to keep a track without new detections

# ===== Model load =====
model = dg.load_model(
    model_name="best",
    inference_host_address="@local",
    zoo_url="../models/model_cropped_big_hailo8",
    token="",
    overlay_color=(0, 255, 0)
)

# ===== Helpers =====
def fb_resolution():
    with open("/sys/class/graphics/fb0/virtual_size") as f:
        w, h = f.read().strip().split(",")
    return int(w), int(h)

def generate_tiles(frame):
    h, w, _ = frame.shape
    tiles, coords = [], []
    for y in range(0, h - TILE_HEIGHT + 1, STRIDE_Y):
        for x in range(0, w - TILE_WIDTH + 1, STRIDE_X):
            tiles.append(frame[y:y + TILE_HEIGHT, x:x + TILE_WIDTH])
            coords.append((x, y))
    return tiles, coords

def apply_nms(detections, iou_thresh=0.5):
    if not detections:
        return []
    boxes = np.array([d["bbox"] for d in detections])
    scores = np.array([d["score"] for d in detections])
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, iou_thresh)
    if isinstance(idx, np.ndarray):
        idx = idx.flatten().tolist()
    elif isinstance(idx, list) and idx and isinstance(idx[0], (list, tuple)):
        idx = [i[0] for i in idx]
    return [detections[i] for i in idx] if idx else []

def run_detection(frame_rgb):
    t0 = time.time()
    tiles, coords = generate_tiles(frame_rgb)
    results = model.predict_batch(tiles)

    detections = []
    for result, (dx, dy) in zip(results, coords):
        for det in result.results:
            if det["score"] >= CONF_THRESHOLD:
                x1, y1, x2, y2 = det["bbox"]
                detections.append({
                    "bbox": [x1 + dx, y1 + dy, x2 + dx, y2 + dy],
                    "score": float(det["score"]),
                })
    print("Detection time:", f"{time.time()-t0:.3f}s")
    return apply_nms(detections, IOU_THRESHOLD)

def make_overlay(h, w, boxes_text, fps_text):
    overlay = np.zeros((h, w, 4), dtype=np.uint8)  # BGRA
    for x1, y1, x2, y2, color, label in boxes_text:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (*color, 255), 2)
        cv2.putText(overlay, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (*color, 255), 2)
    cv2.putText(overlay, fps_text, (w - 200, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)
    return overlay

def stdin_key_available():
    return select.select([sys.stdin], [], [], 0)[0]

# --- Norfair distance ---
def euclidean_distance(detected_points, tracked_points):
    # each is shape (1,2): [[x,y]]
    return np.linalg.norm(detected_points - tracked_points)

# ===== Main =====
def main():
    CAP_W, CAP_H = 2592, 1944
    FB_W, FB_H = fb_resolution()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (CAP_W, CAP_H)}
    )
    picam2.configure(config)
    picam2.start_preview(Preview.DRM, x=0, y=0, width=FB_W, height=FB_H)
    picam2.start()

    # --- Norfair tracker ---
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=DISTANCE_THRESHOLD,
        hit_inertia_max=MAX_AGE,
        initialization_delay=INITIALIZATION_DELAY,
        points_hit_counter_max=HITS_TO_START,
    )

    # Remember last known box size per track id so we can draw rectangles (Norfair tracks points)
    track_box_sizes = {}  # track_id -> (w, h)

    last_detection_time = 0.0
    fps = 0.0

    old_attrs = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            loop_start = time.time()

            frame = picam2.capture_array()[:, :, :3]  # RGB
            boxes_for_overlay = []

            # Run detection at intervals; feed results to Norfair
            if (time.time() - last_detection_time) >= REDETECT_INTERVAL:
                dets = run_detection(frame)
                last_detection_time = time.time()

                # Convert to Norfair detections (use center point; keep size in data)
                nf_dets = []
                for d in dets:
                    x1, y1, x2, y2 = map(int, d["bbox"])
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w, h = x2 - x1, y2 - y1
                    nf_dets.append(
                        Detection(
                            points=np.array([[cx, cy]], dtype=np.float32),
                            scores=np.array([d["score"]], dtype=np.float32),
                            data={"size": (w, h)},
                        )
                    )

                tracks = tracker.update(detections=nf_dets)

                # Update size memory from each track's last detection (if available)
                for t in tracks:
                    if t.last_detection and "size" in (t.last_detection.data or {}):
                        track_box_sizes[t.id] = t.last_detection.data["size"]
            else:
                # No fresh detections: keep tracker alive (no updates)
                tracks = tracker.update(detections=[])

            # Draw current tracks as rectangles using last known size
            for t in tracks:
                # t.estimate shape (1,2)
                cx, cy = t.estimate[0]
                w, h = track_box_sizes.get(t.id, (100, 100))
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = x1 + int(w)
                y2 = y1 + int(h)
                boxes_for_overlay.append((x1, y1, x2, y2, (255, 0, 0), f"ID {t.id}"))

            # FPS
            inst_fps = 1.0 / max(1e-6, (time.time() - loop_start))
            fps = 0.9 * fps + 0.1 * inst_fps

            overlay = make_overlay(CAP_H, CAP_W, boxes_for_overlay, f"FPS: {fps:.2f}")
            picam2.set_overlay(overlay)

            if stdin_key_available():
                ch = sys.stdin.read(1)
                if ch.lower() == 'q':
                    print("Quit requested (q)")
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Interrupted (Ctrl+C).")
    finally:
        try:
            picam2.set_overlay(None)
        except Exception:
            pass
        picam2.stop_preview()
        picam2.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)

if __name__ == "__main__":
    main()
