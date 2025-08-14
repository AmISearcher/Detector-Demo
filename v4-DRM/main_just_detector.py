#!/usr/bin/env python3
import degirum as dg
import cv2
import numpy as np
import time
import sys, termios, tty, select
from picamera2 import Picamera2, Preview

# ===== Parameters =====
TILE_WIDTH, TILE_HEIGHT = 480, 480
OVERLAP_X, OVERLAP_Y = 0.2, 0.2
STRIDE_X = int(TILE_WIDTH * (1 - OVERLAP_X))
STRIDE_Y = int(TILE_HEIGHT * (1 - OVERLAP_Y))
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
REDETECT_INTERVAL = 0.4  # seconds between detection updates

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
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                               CONF_THRESHOLD, iou_thresh)
    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    elif isinstance(indices, list) and indices and isinstance(indices[0], (list, tuple)):
        indices = [i[0] for i in indices]
    return [detections[i] for i in indices] if indices else []

def run_detection(frame_rgb):
    t0 = time.time()
    tiles, coords = generate_tiles(frame_rgb)
    results = model.predict_batch(tiles)

    detections = []
    for result, (dx, dy) in zip(results, coords):
        for det in result.results:
            if det["score"] >= CONF_THRESHOLD:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
                detections.append(det)
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

# ===== Main =====
def main():
    CAP_W, CAP_H = 2592, 1944
    FB_W, FB_H = fb_resolution()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (CAP_W, CAP_H)}
    )
    picam2.configure(config)

    # Fullscreen DRM preview
    picam2.start_preview(Preview.DRM, x=0, y=0, width=FB_W, height=FB_H)
    picam2.start()

    last_detection_time = 0.0
    last_boxes = []
    fps = 0.0

    old_attrs = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            loop_start = time.time()

            # Capture frame
            frame = picam2.capture_array()[:, :, :3]

            # Run detection every REDETECT_INTERVAL
            if (time.time() - last_detection_time) >= REDETECT_INTERVAL:
                detections = run_detection(frame)
                last_detection_time = time.time()
                last_boxes = []
                if detections:
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det["bbox"])
                        last_boxes.append((x1, y1, x2, y2, (0, 255, 0), "Detected"))

            # FPS
            inst_fps = 1.0 / max(1e-6, (time.time() - loop_start))
            fps = 0.9 * fps + 0.1 * inst_fps

            overlay = make_overlay(CAP_H, CAP_W, last_boxes, f"FPS: {fps:.2f}")
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
