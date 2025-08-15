#!/usr/bin/env python3
import degirum as dg
import cv2
import numpy as np
import time
import sys, termios, tty, select
from picamera2 import Picamera2, Preview
import norfair

# ===== Parameters =====
TILE_WIDTH, TILE_HEIGHT = 480, 480
OVERLAP_X, OVERLAP_Y = 0.2, 0.2
STRIDE_X = int(TILE_WIDTH * (1 - OVERLAP_X))
STRIDE_Y = int(TILE_HEIGHT * (1 - OVERLAP_Y))
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
REDETECT_INTERVAL = 0.5
TM_METHOD = cv2.TM_CCOEFF_NORMED
TM_THRESH_LOCKED = 0.25
DISTANCE_THRESHOLD = 300.0  # in full-res pixels

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
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, iou_thresh)
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

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

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

    tracker = norfair.Tracker(distance_function=euclidean_distance,
                               distance_threshold=DISTANCE_THRESHOLD)
    template = None
    tpl_w = tpl_h = 0

    tracking = False
    last_detection_time = 0.0
    fps = 0.0

    old_attrs = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            loop_start = time.time()
            frame = picam2.capture_array()
            frame_rgb = frame[:, :, :3]

            boxes_for_overlay = []
            need_redetect = (time.time() - last_detection_time) >= REDETECT_INTERVAL

            if not tracking or need_redetect:
                detections = run_detection(frame_rgb)
                last_detection_time = time.time()
                if detections:
                    best = max(detections, key=lambda d: d["score"])
                    x1, y1, x2, y2 = map(int, best["bbox"])
                    w, h = x2 - x1, y2 - y1
                    if w > 0 and h > 0:
                        template = frame_rgb[y1:y2, x1:x2].copy()
                        tpl_w, tpl_h = w, h
                        tracker = norfair.Tracker(
                            distance_function=euclidean_distance,
                            distance_threshold=DISTANCE_THRESHOLD
                        )
                        det = norfair.Detection(
                            points=np.array([[x1 + w/2, y1 + h/2]], dtype=np.float32),
                            scores=np.array([float(best["score"])]),
                            data={"size": (w, h)}
                        )
                        tracker.update([det])
                        tracking = True
                    boxes_for_overlay.append((x1, y1, x2, y2, (0, 255, 0), "Re-Detected"))
                else:
                    tracking = False

            else:
                # Template match to get detection
                res = cv2.matchTemplate(frame_rgb, template, TM_METHOD)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if TM_METHOD in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                    score = 1.0 - min_val
                    tl = min_loc
                else:
                    score = max_val
                    tl = max_loc

                nf_dets = []
                if score >= TM_THRESH_LOCKED:
                    cx_d = tl[0] + tpl_w/2
                    cy_d = tl[1] + tpl_h/2
                    nf_dets.append(
                        norfair.Detection(
                            points=np.array([[cx_d, cy_d]], dtype=np.float32),
                            scores=np.array([float(score)]),
                            data={"size": (tpl_w, tpl_h)}
                        )
                    )
                tracks = tracker.update(nf_dets)

                for t in tracks:
                    est_cx, est_cy = t.estimate[0]
                    w, h = tpl_w, tpl_h
                    x1 = int(est_cx - w / 2)
                    y1 = int(est_cy - h / 2)
                    x2 = int(est_cx + w / 2)
                    y2 = int(est_cy + h / 2)
                    boxes_for_overlay.append((x1, y1, x2, y2, (255, 0, 0), f"Tracking s:{score:.2f}"))

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

