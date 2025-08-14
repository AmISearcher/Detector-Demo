import degirum as dg
import cv2
import numpy as np
import time
from picamera2 import Picamera2, Preview
#from picamera2.previews import DRMPreview

# ====== Parameters ======
TILE_WIDTH, TILE_HEIGHT = 480, 480
OVERLAP_X, OVERLAP_Y = 0.2, 0.2
STRIDE_X = int(TILE_WIDTH * (1 - OVERLAP_X))
STRIDE_Y = int(TILE_HEIGHT * (1 - OVERLAP_Y))
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
REDETECT_INTERVAL = 3.0  # seconds

# ====== Load model ======
model = dg.load_model(
    model_name="best",
    inference_host_address="@local",
    zoo_url="../models/model_cropped_big_hailo8",
    token="",
    overlay_color=(0, 255, 0),
)

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

    idx = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), CONF_THRESHOLD, iou_thresh
    )
    # Normalize return shapes
    if isinstance(idx, np.ndarray):
        idx = idx.flatten().tolist()
    elif isinstance(idx, list) and idx and isinstance(idx[0], (list, tuple)):
        idx = [i[0] for i in idx]
    return [detections[i] for i in idx] if idx else []

def run_detection(frame):
    t0 = time.time()
    tiles, coords = generate_tiles(frame)
    results = model.predict_batch(tiles)

    detections = []
    for result, (dx, dy) in zip(results, coords):
        for det in result.results:
            if det["score"] >= CONF_THRESHOLD:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
                detections.append(det)
    print("Detection time:", time.time() - t0)
    return apply_nms(detections, IOU_THRESHOLD)

def make_overlay(frame_shape, boxes_text, fps_text):
    """
    Create a BGRA overlay the same size as the preview.
    boxes_text: list of tuples (pt1, pt2, color_bgr, label)
    """
    h, w = frame_shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)  # BGRA

    # Draw boxes and labels
    for (x1, y1), (x2, y2), color, label in boxes_text:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (*color, 255), 2)
        cv2.putText(
            overlay, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (*color, 255), 2
        )

    # Draw FPS (bottom-right)
    cv2.putText(
        overlay, fps_text, (w - 180, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 2
    )
    return overlay

def main():
    picam2 = Picamera2()
    # Use a preview configuration; DRMPreview shows it without X/Wayland
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (2592, 1944)}
    )
    picam2.configure(config)
    picam2.start_preview(Preview.DRM)  # <— headless "window"
    picam2.start()

    tracker = None
    tracking = False
    last_detection_time = 0.0
    fps = 0.0

    try:
        while True:
            loop_start = time.time()
            frame = picam2.capture_array()  # RGB888

            boxes_text = []
            now = time.time()
            need_redetect = (now - last_detection_time) >= REDETECT_INTERVAL
            print("tracking:", tracking, "need_redetect:", need_redetect)

            if not tracking or need_redetect:
                detections = run_detection(frame)
                last_detection_time = now
                print("detections:", detections)

                if detections:
                    best_det = max(detections, key=lambda d: d["score"])
                    x1, y1, x2, y2 = map(int, best_det["bbox"])
                    w, h = x2 - x1, y2 - y1
                    print(f"[DEBUG] BBox: ({x1},{y1})-({x2},{y2}) w={w} h={h}")

                    if w > 0 and h > 0:
                        try:
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, (x1, y1, w, h))
                            tracking = True
                        except Exception as e:
                            print(f"[ERROR] tracker.init: {e}")
                            tracking = False
                    else:
                        print("[WARN] Invalid bbox; skip tracker init")
                        tracking = False

                    boxes_text.append(((x1, y1), (x2, y2), (0, 255, 0), "Re-Detected"))
                else:
                    tracking = False

            elif tracking:
                ok, bbox = tracker.update(frame)
                if ok:
                    x, y, w, h = map(int, bbox)
                    boxes_text.append(((x, y), (x + w, y + h), (255, 0, 0), "Tracking"))
                else:
                    print("tracking lost")
                    tracking = False

            # FPS
            inst_fps = 1.0 / max(1e-6, (time.time() - loop_start))
            fps = 0.9 * fps + 0.1 * inst_fps

            # Build and push overlay (BGRA)
            overlay = make_overlay(frame.shape, boxes_text, f"FPS: {fps:.2f}")
            # NOTE: set_overlay expects the same size as preview and BGRA order.
            picam2.set_overlay(overlay)

            # No waitKey in headless mode; small sleep to avoid 100% busy loop
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        # Clear overlay and stop cleanly
        picam2.set_overlay(None)
        picam2.stop_preview()
        picam2.stop()

if __name__ == "__main__":
    main()
