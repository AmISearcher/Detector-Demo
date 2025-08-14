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
REDETECT_INTERVAL = 0.4  # seconds

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
    print("Detection time:", f"{time.time()-t0:.3f}s  |  dets:", len(detections))
    return apply_nms(detections, IOU_THRESHOLD)

def make_overlay(h, w, boxes_text, fps_text):
    """
    Build **BGRA** overlay (IMPORTANT) matching the DISPLAYED stream size.
    """
    overlay = np.zeros((h, w, 4), dtype=np.uint8)  # BGRA

    # Always-visible crosshair to prove overlay path works
    cx, cy = w // 2, h // 2
    cv2.line(overlay, (cx - 80, cy), (cx + 80, cy), (255, 255, 255, 160), 3)
    cv2.line(overlay, (cx, cy - 80), (cx, cy + 80), (255, 255, 255, 160), 3)

    # Draw boxes and labels (thicker so scaling doesn't erase them)
    for x1, y1, x2, y2, color, label in boxes_text:
        # color is BGR; add alpha 255
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (*color, 255), 6)
        cv2.putText(
            overlay, label, (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (*color, 255), 3
        )

    # FPS box
    cv2.rectangle(overlay, (w - 240, h - 60), (w - 10, h - 10), (0, 0, 0, 160), -1)
    cv2.putText(
        overlay, fps_text, (w - 230, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255, 255), 2
    )
    return overlay  # BGRA

def stdin_key_available():
    return select.select([sys.stdin], [], [], 0)[0]

# ===== Main =====
def main():
    CAP_W, CAP_H = 2592, 1944          # full-res capture/display stream
    FB_W, FB_H = fb_resolution()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (CAP_W, CAP_H)}  # 32-bit for DRM
    )
    picam2.configure(config)

    # Fullscreen DRM preview (stretched to FB size)
    picam2.start_preview(Preview.DRM, x=0, y=0, width=FB_W, height=FB_H)
    picam2.start()

    last_detection_time = 0.0
    last_boxes = []  # carry over boxes between detection cycles
    fps = 0.0

    # Read 'q' without blocking
    old_attrs = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            loop_start = time.time()

            frame = picam2.capture_array()[:, :, :3]  # RGB (drop the X channel)

            # Periodic detection
            if (time.time() - last_detection_time) >= REDETECT_INTERVAL:
                dets = run_detection(frame)
                last_detection_time = time.time()
                last_boxes = []
                for d in dets:
                    x1, y1, x2, y2 = map(int, d["bbox"])
                    last_boxes.append((x1, y1, x2, y2, (0, 255, 0), "Detected"))

            # FPS calc
            inst_fps = 1.0 / max(1e-6, (time.time() - loop_start))
            fps = 0.9 * fps + 0.1 * inst_fps

            # Build BGRA overlay and PUSH with explicit format
            overlay = make_overlay(CAP_H, CAP_W, last_boxes, f"FPS: {fps:.2f}")
            picam2.set_overlay(overlay, format="bgra")  # <-- KEY FIX

            # Quit on 'q'
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
