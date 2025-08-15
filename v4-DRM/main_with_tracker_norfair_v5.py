#!/usr/bin/env python3
import sys, termios, tty, select, time
import numpy as np
import cv2
from picamera2 import Picamera2, Preview
import degirum as dg
import norfair

# ====================== Params ======================
TILE_WIDTH, TILE_HEIGHT = 480, 480
OVERLAP_X, OVERLAP_Y = 0.2, 0.2
STRIDE_X = int(TILE_WIDTH * (1 - OVERLAP_X))
STRIDE_Y = int(TILE_HEIGHT * (1 - OVERLAP_Y))
CONF_THRESHOLD = 0.3
NMS_IOU = 0.5
REDETECT_INTERVAL = 0.8

MAIN_W, MAIN_H = 2592, 1944
LORES_W, LORES_H = 640, 480

TM_METHOD = cv2.TM_CCOEFF_NORMED
TM_THRESH = 0.35           # stricter to reduce false positives

# Norfair settings
DISTANCE_THRESHOLD = 80.0  # stricter for stability
INITIALIZATION_DELAY = 0
HIT_COUNTER_MAX = 120

# ====================== Model ======================
model = dg.load_model(
    model_name="best",
    inference_host_address="@local",
    zoo_url="../models/model_cropped_big_hailo8",
    token="",
    overlay_color=(0, 255, 0)
)

# ====================== Helpers ======================
def fb_resolution():
    with open("/sys/class/graphics/fb0/virtual_size") as f:
        w, h = f.read().strip().split(",")
    return int(w), int(h)

def stdin_key_available():
    return select.select([sys.stdin], [], [], 0)[0]

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
    boxes = np.array([d["bbox"] for d in detections], dtype=float)
    scores = np.array([d["score"] for d in detections], dtype=float)
    xywh = boxes.copy()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), CONF_THRESHOLD, iou_thresh)
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
            score = float(det["score"])
            if score >= CONF_THRESHOLD:
                x1, y1, x2, y2 = det["bbox"]
                detections.append({
                    "bbox": [x1 + dx, y1 + dy, x2 + dx, y2 + dy],
                    "score": score,
                })
    print(f"[DETECTION] time={time.time()-t0:.3f}s  n={len(detections)}")
    return apply_nms(detections, NMS_IOU)

def make_overlay(h, w, boxes_text, fps_text):
    ov = np.zeros((h, w, 4), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.line(ov, (cx-120, cy), (cx+120, cy), (255,255,255,180), 3)
    cv2.line(ov, (cx, cy-120), (cx, cy+120), (255,255,255,180), 3)
    for x1, y1, x2, y2, color, label in boxes_text:
        cv2.rectangle(ov, (x1, y1), (x2, y2), (*color, 255), 6)
        cv2.putText(ov, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (*color, 255), 3)
    cv2.rectangle(ov, (w - 240, h - 60), (w - 12, h - 12), (0, 0, 0, 160), -1)
    cv2.putText(ov, fps_text, (w - 230, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255, 255), 2)
    return ov

def nf_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

# ====================== Main ======================
def main():
    FB_W, FB_H = fb_resolution()
    print(f"[INFO] FB {FB_W}x{FB_H} | main {MAIN_W}x{MAIN_H} | lores {LORES_W}x{LORES_H}")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (MAIN_W, MAIN_H)},
        lores={"format": "YUV420", "size": (LORES_W, LORES_H)},
        display="main",
    )
    picam2.configure(config)

    try:
        picam2.set_controls({"FrameDurationLimits": (33333, 33333)})
    except Exception:
        pass

    picam2.start_preview(Preview.DRM, x=0, y=0, width=FB_W, height=FB_H)
    picam2.start()
    print("[INFO] Started. Press 'q' to quit.")

    sx, sy = MAIN_W / LORES_W, MAIN_H / LORES_H

    tracker = norfair.Tracker(
        distance_function=nf_distance,
        distance_threshold=DISTANCE_THRESHOLD,
        initialization_delay=INITIALIZATION_DELAY,
        hit_counter_max=HIT_COUNTER_MAX,
    )

    track_box_sizes = {}
    template = None
    tpl_w = tpl_h = 0

    last_detection_time = 0.0
    fps = 0.0

    old_attrs = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            loop_t = time.time()

            lo = picam2.capture_array("lores")
            if lo.ndim == 3 and lo.shape[2] == 3:
                lo_bgr = lo
            else:
                lo_bgr = cv2.cvtColor(lo, cv2.COLOR_YUV2BGR_I420)

            detections_for_tracker = []

            if (time.time() - last_detection_time) >= REDETECT_INTERVAL:
                main_frame = picam2.capture_array()[:, :, :3]
                dets = run_detection(main_frame)
                last_detection_time = time.time()

                if dets:
                    best = max(dets, key=lambda d: d["score"])
                    x1m, y1m, x2m, y2m = map(int, best["bbox"])
                    x1l, y1l = max(0, int(x1m / sx)), max(0, int(y1m / sy))
                    x2l, y2l = min(LORES_W, int(x2m / sx)), min(LORES_H, int(y2m / sy))
                    if x2l > x1l and y2l > y1l:
                        template = lo_bgr[y1l:y2l, x1l:x2l].copy()
                        tpl_h, tpl_w = template.shape[:2]
                        det_center = np.array([[(x1l + x2l)/2.0, (y1l + y2l)/2.0]], dtype=np.float32)
                        detections_for_tracker.append(
                            norfair.Detection(
                                points=det_center,
                                scores=np.array([float(best["score"])], dtype=np.float32),
                                data={"size": (x2l - x1l, y2l - y1l)},
                            )
                        )

            if template is not None and template.size and tpl_w > 0 and tpl_h > 0:
                res = cv2.matchTemplate(lo_bgr, template, TM_METHOD)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if TM_METHOD in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                    score = 1.0 - min_val; tl = min_loc
                else:
                    score = max_val; tl = max_loc

                if score >= TM_THRESH:
                    cx_l = tl[0] + tpl_w / 2.0
                    cy_l = tl[1] + tpl_h / 2.0
                    detections_for_tracker.append(
                        norfair.Detection(
                            points=np.array([[cx_l, cy_l]], dtype=np.float32),
                            scores=np.array([float(score)], dtype=np.float32),
                            data={"size": (tpl_w, tpl_h)},
                        )
                    )

            # Keep only best detection for tracker
            if detections_for_tracker:
                detections_for_tracker = [max(detections_for_tracker, key=lambda d: d.scores[0])]

            tracks = tracker.update(detections_for_tracker)

            # Keep only best track
            if tracks:
                tracks = [max(tracks, key=lambda t: (t.last_detection.scores[0] if t.last_detection else 0))]

            boxes = []
            for t in tracks:
                est_cx, est_cy = t.estimate[0]
                if getattr(t, "last_detection", None) and getattr(t.last_detection, "data", None):
                    if "size" in t.last_detection.data:
                        track_box_sizes[t.id] = t.last_detection.data["size"]
                w_l, h_l = track_box_sizes.get(t.id, (tpl_w or 60, tpl_h or 60))
                x1m = int((est_cx - w_l/2) * sx)
                y1m = int((est_cy - h_l/2) * sy)
                x2m = int((est_cx + w_l/2) * sx)
                y2m = int((est_cy + h_l/2) * sy)
                boxes.append((x1m, y1m, x2m, y2m, (255, 0, 0), f"ID {t.id}"))

            inst_fps = 1.0 / max(1e-6, (time.time() - loop_t))
            fps = 0.9 * fps + 0.1 * inst_fps
            overlay = make_overlay(MAIN_H, MAIN_W, boxes, f"FPS: {fps:.1f}")
            picam2.set_overlay(overlay)

            if stdin_key_available() and sys.stdin.read(1).lower() == 'q':
                print("[INFO] Quit requested (q)."); break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        try: picam2.set_overlay(None)
        except Exception: pass
        picam2.stop_preview(); picam2.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)
        print("[INFO] Clean exit.")

if __name__ == "__main__":
    main()
