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
REDETECT_INTERVAL = 0.4
TM_METHOD = cv2.TM_CCOEFF_NORMED
TM_THRESH_LOCKED = 0.35
DISTANCE_THRESHOLD = 200.0  # Norfair distance (in MAIN pixels)

# "Same object" test: if IoU >= this, treat detector hit as same object
MATCH_IOU = 0.30  # tune 0.25–0.5

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
                detections.append({
                    "bbox": [x1 + dx, y1 + dy, x2 + dx, y2 + dy],
                    "score": float(det["score"]),
                })
    print("Detection time:", f"{time.time()-t0:.3f}s", "| dets:", len(detections))
    return apply_nms(detections, IOU_THRESHOLD)

def make_overlay(h, w, boxes_text, fps_text):
    overlay = np.zeros((h, w, 4), dtype=np.uint8)  # BGRA
    # crosshair
    cx, cy = w // 2, h // 2
    cv2.line(overlay, (cx-120, cy), (cx+120, cy), (255,255,255,180), 3)
    cv2.line(overlay, (cx, cy-120), (cx, cy+120), (255,255,255,180), 3)

    for x1, y1, x2, y2, color, label in boxes_text:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (*color, 255), 6)
        cv2.putText(overlay, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (*color, 255), 3)
    # FPS badge
    cv2.rectangle(overlay, (w - 240, h - 60), (w - 12, h - 12), (0, 0, 0, 160), -1)
    cv2.putText(overlay, fps_text, (w - 230, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255, 255), 2)
    return overlay

def stdin_key_available():
    return select.select([sys.stdin], [], [], 0)[0]

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def bbox_from_center(cx, cy, w, h):
    x1 = int(cx - w/2); y1 = int(cy - h/2)
    x2 = int(cx + w/2); y2 = int(cy + h/2)
    return x1, y1, x2, y2

def iou(a, b):
    # a, b: (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

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

    # Template + “current track box” state (MAIN coords)
    template = None
    tpl_w = tpl_h = 0
    have_track = False
    track_cx = track_cy = None  # last estimated center (MAIN coords)

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

            # ========== 1) Periodic detector ==========
            if need_redetect:
                dets = run_detection(frame_rgb)
                last_detection_time = time.time()

                if dets:
                    best = max(dets, key=lambda d: d["score"])
                    x1d, y1d, x2d, y2d = map(int, best["bbox"])
                    wd, hd = x2d - x1d, y2d - y1d

                    # Compare detector bbox to current track bbox (if we have one)
                    same_object = False
                    if have_track and tpl_w > 0 and tpl_h > 0 and track_cx is not None and track_cy is not None:
                        prev_box = bbox_from_center(track_cx, track_cy, tpl_w, tpl_h)
                        det_box  = (x1d, y1d, x2d, y2d)
                        current_iou = iou(prev_box, det_box)
                        same_object = (current_iou >= MATCH_IOU)
                        # Debug:
                        # print(f"[MATCH] IoU={current_iou:.2f} (>= {MATCH_IOU}) -> same={same_object}")

                    if wd > 0 and hd > 0:
                        # Always refresh the template to keep appearance up-to-date
                        template = frame_rgb[y1d:y2d, x1d:x2d].copy()
                        tpl_w, tpl_h = wd, hd

                        # Build a Norfair detection from the detector center
                        det_center = np.array([[x1d + wd/2, y1d + hd/2]], dtype=np.float32)
                        det_obj = norfair.Detection(points=det_center,
                                                    scores=np.array([float(best["score"])]),
                                                    data={"size": (wd, hd)})

                        if same_object and have_track:
                            # SAME OBJECT: do NOT reset tracker, do NOT draw green box
                            tracker.update([det_obj])  # reinforce track
                            # leave drawing to the tracking block below
                        else:
                            # NEW OBJECT (or no active track): reset tracker and draw green
                            tracker = norfair.Tracker(distance_function=euclidean_distance,
                                                      distance_threshold=DISTANCE_THRESHOLD)
                            tracker.update([det_obj])
                            boxes_for_overlay.append((x1d, y1d, x2d, y2d, (0, 255, 0), "Re-Detected"))
                            have_track = True
                            # initialize center for next loop
                            track_cx, track_cy = float(x1d + wd/2), float(y1d + hd/2)

                else:
                    # No detections: keep tracking via template match + Norfair
                    pass

            # ========== 2) Between detections: template match to produce detection ==========
            nf_dets = []
            track_score = 0.0
            if template is not None and template.size and tpl_w > 0 and tpl_h > 0:
                res = cv2.matchTemplate(frame_rgb, template, TM_METHOD)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if TM_METHOD in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                    track_score = 1.0 - min_val; tl = min_loc
                else:
                    track_score = max_val; tl = max_loc

                if track_score >= TM_THRESH_LOCKED:
                    cx_d = tl[0] + tpl_w/2
                    cy_d = tl[1] + tpl_h/2
                    nf_dets.append(
                        norfair.Detection(
                            points=np.array([[cx_d, cy_d]], dtype=np.float32),
                            scores=np.array([float(track_score)]),
                            data={"size": (tpl_w, tpl_h)}
                        )
                    )

            # Update Norfair with (possibly empty) detections from template match
            tracks = tracker.update(nf_dets)

            # Draw tracks (blue) and remember center for “same object” IoU next cycle
            for t in tracks:
                est_cx, est_cy = t.estimate[0]
                # Size from latest detection if present
                if getattr(t, "last_detection", None) and getattr(t.last_detection, "data", None):
                    if "size" in t.last_detection.data:
                        tpl_w, tpl_h = t.last_detection.data["size"]
                x1, y1, x2, y2 = bbox_from_center(est_cx, est_cy, tpl_w, tpl_h)
                boxes_for_overlay.append((x1, y1, x2, y2, (255, 0, 0), f"Tracking s:{track_score:.2f}"))
                track_cx, track_cy = float(est_cx), float(est_cy)
                have_track = True

            # ========== 3) FPS & overlay ==========
            inst_fps = 1.0 / max(1e-6, (time.time() - loop_start))
            fps = 0.9 * fps + 0.1 * inst_fps
            overlay = make_overlay(CAP_H, CAP_W, boxes_for_overlay, f"FPS: {fps:.2f}")
            picam2.set_overlay(overlay)

            # Quit
            if stdin_key_available() and sys.stdin.read(1).lower() == 'q':
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
