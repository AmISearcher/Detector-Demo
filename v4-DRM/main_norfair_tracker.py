#!/usr/bin/env python3
# Norfair center-lock tracker (no OpenCV tracker). Picamera2 DRM fullscreen.
# SPACE: lock target in center box (template matching on lores); q: quit.

import sys, time, select, termios, tty
import numpy as np
import cv2
from picamera2 import Picamera2, Preview
import norfair

# ---------- Settings ----------
MAIN_W, MAIN_H = 2592, 1944     # displayed stream
LORES_W, LORES_H = 640, 480     # tracking stream (fast)
BOX_FRAC = 0.25                 # central lock box size (fraction of lores dims)
TM_METHOD = cv2.TM_CCOEFF_NORMED
TM_THRESH_LOCKED = 0.35         # detection accepted if score >= this
REDETECT_EVERY = 1              # run matching every frame (keep simple/fast)
DISTANCE_THRESHOLD = 35.0       # Norfair distance (pixels in lores coords)

# ---------- Helpers ----------
def fb_resolution():
    with open("/sys/class/graphics/fb0/virtual_size") as f:
        w, h = f.read().strip().split(",")
    return int(w), int(h)

def stdin_key_available():
    return select.select([sys.stdin], [], [], 0)[0]

def make_overlay(h, w, boxes, msg):
    # BGRA overlay matching MAIN size
    ov = np.zeros((h, w, 4), dtype=np.uint8)
    # Banner + crosshair
    cv2.rectangle(ov, (10, 10), (w-10, 80), (0, 0, 0, 150), -1)
    cv2.putText(ov, msg, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255,255), 2)
    cx, cy = w//2, h//2
    cv2.line(ov, (cx-120, cy), (cx+120, cy), (255,255,255,180), 3)
    cv2.line(ov, (cx, cy-120), (cx, cy+120), (255,255,255,180), 3)
    for x1, y1, x2, y2, color, label in boxes:
        cv2.rectangle(ov, (x1, y1), (x2, y2), (*color, 255), 8)
        cv2.putText(ov, label, (x1, max(0, y1-12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (*color,255), 3)
    return ov

def euclidean_distance(detection, tracked_object):
    # Norfair-compatible distance (lores pixel space)
    return np.linalg.norm(detection.points - tracked_object.estimate)

# ---------- Main ----------
def main():
    FB_W, FB_H = fb_resolution()
    print(f"[INFO] Framebuffer: {FB_W}x{FB_H}")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (MAIN_W, MAIN_H)},
        lores={"format": "YUV420", "size": (LORES_W, LORES_H)},
        display="main",
    )
    picam2.configure(config)

    # Try to raise FPS a bit (best-effort)
    try:
        picam2.set_controls({"FrameDurationLimits": (33333, 33333)})  # ~30fps
    except Exception:
        pass

    picam2.start_preview(Preview.DRM, x=0, y=0, width=FB_W, height=FB_H)
    picam2.start()
    print("[INFO] Camera started. SPACE to lock target in center; 'q' to quit.")

    # Terminal cbreak for non-blocking keys
    old_attrs = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    # Center lock box in **lores** coordinates
    bw, bh = int(LORES_W * BOX_FRAC), int(LORES_H * BOX_FRAC)
    cx, cy = LORES_W // 2, LORES_H // 2
    c_x1, c_y1 = cx - bw // 2, cy - bh // 2

    # Scale from lores → main for drawing
    sx, sy = MAIN_W / LORES_W, MAIN_H / LORES_H

    # Norfair tracker (works with older versions)
    tracker = norfair.Tracker(distance_function=euclidean_distance,
                              distance_threshold=DISTANCE_THRESHOLD)

    locked = False
    template = None           # lores template (BGR or gray)
    tpl_w, tpl_h = None, None # size of template in lores space
    last_size = (bw, bh)      # default draw size if no det data yet
    fps = 0.0

    try:
        while True:
            t0 = time.time()

            # --- Grab lores frame for matching ---
            lo = picam2.capture_array("lores")  # YUV420 (H*1.5 x W)
            if lo.ndim == 3 and lo.shape[2] == 3:
                lo_bgr = lo
            else:
                lo_bgr = cv2.cvtColor(lo, cv2.COLOR_YUV2BGR_I420)

            boxes_main = []
            msg = "SPACE: lock target in center | q: quit"

            if not locked:
                # Draw center box on MAIN to show where lock will happen
                x1m = int(c_x1 * sx); y1m = int(c_y1 * sy)
                x2m = int((c_x1 + bw) * sx); y2m = int((c_y1 + bh) * sy)
                boxes_main.append((x1m, y1m, x2m, y2m, (0, 255, 255), "CENTER"))
            else:
                # --- TEMPLATE MATCHING on lores ---
                # Make sure template is valid
                if template is not None and template.size != 0:
                    res = cv2.matchTemplate(lo_bgr, template, TM_METHOD)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if TM_METHOD in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                        score = 1.0 - min_val
                        tl = min_loc
                    else:
                        score = max_val
                        tl = max_loc

                    # Create Norfair detection if confidence is decent
                    nf_dets = []
                    if score >= TM_THRESH_LOCKED:
                        x, y = tl
                        w, h = tpl_w, tpl_h
                        cx_d = x + w/2.0
                        cy_d = y + h/2.0
                        nf_dets.append(
                            norfair.Detection(
                                points=np.array([[cx_d, cy_d]], dtype=np.float32),
                                scores=np.array([float(score)], dtype=np.float32),
                                data={"size": (w, h)}
                            )
                        )
                        # Update last_size for drawing if we have a detection
                        last_size = (w, h)
                    else:
                        # No detection this frame → allow Norfair to predict
                        nf_dets = []

                    # Update tracker (returns active objects in most versions)
                    tracks = tracker.update(detections=nf_dets)

                    # Draw whichever tracks are active
                    for t in tracks:
                        est_cx, est_cy = t.estimate[0]
                        # size from latest detection if present
                        if getattr(t, "last_detection", None) and \
                           getattr(t.last_detection, "data", None) and \
                           "size" in t.last_detection.data:
                            last_size = t.last_detection.data["size"]

                        w, h = last_size
                        x1 = int((est_cx - w / 2) * sx)
                        y1 = int((est_cy - h / 2) * sy)
                        x2 = int((est_cx + w / 2) * sx)
                        y2 = int((est_cy + h / 2) * sy)
                        boxes_main.append((x1, y1, x2, y2, (0, 255, 0), f"ID {t.id}  s:{score:.2f}"))
                        msg = f"Tracking with Norfair | score:{score:.2f} | 'SPACE' relock, 'q' quit"
                else:
                    locked = False
                    msg = "Template empty; press SPACE to lock again."

            # FPS
            inst = 1.0 / max(1e-6, (time.time() - t0))
            fps = 0.9 * fps + 0.1 * inst
            msg = f"{msg}   |   FPS(lores): {fps:.1f}"

            # Push overlay on MAIN stream
            overlay = make_overlay(MAIN_H, MAIN_W, boxes_main, msg)
            picam2.set_overlay(overlay)

            # ---- Keys ----
            if stdin_key_available():
                ch = sys.stdin.read(1)
                if ch.lower() == "q":
                    print("[INFO] Quit requested.")
                    break
                if ch == " ":
                    # Grab template from lores center box
                    x1, y1 = c_x1, c_y1
                    x2, y2 = x1 + bw, y1 + bh
                    tpl = lo_bgr[y1:y2, x1:x2]
                    if tpl.size == 0:
                        print("[WARN] Empty template; try again.")
                        locked = False
                    else:
                        template = tpl.copy()
                        tpl_h, tpl_w = template.shape[:2]
                        last_size = (tpl_w, tpl_h)
                        tracker = norfair.Tracker(
                            distance_function=euclidean_distance,
                            distance_threshold=DISTANCE_THRESHOLD
                        )  # reset tracker when relocking
                        locked = True
                        print(f"[INFO] Locked with template {tpl_w}x{tpl_h} at lores center box.")

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        try:
            picam2.set_overlay(None)
        except Exception:
            pass
        picam2.stop_preview()
        picam2.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)
        print("[INFO] Clean exit.")

if __name__ == "__main__":
    main()

