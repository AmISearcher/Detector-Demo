import degirum as dg
import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Parameters
TILE_WIDTH, TILE_HEIGHT = 480, 480
OVERLAP_X, OVERLAP_Y = 0.2, 0.2
STRIDE_X = int(TILE_WIDTH * (1 - OVERLAP_X))
STRIDE_Y = int(TILE_HEIGHT * (1 - OVERLAP_Y))
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
REDETECT_INTERVAL = 3.0  # seconds

# Load model
model = dg.load_model(
    model_name="best",
    inference_host_address="@local",
    zoo_url="../models/model_cropped_big",
    token="",
    overlay_color=(0, 255, 0)
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
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, iou_thresh)

    # Handle different return shapes
    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    elif isinstance(indices, list) and isinstance(indices[0], (list, tuple)):
        indices = [i[0] for i in indices]

    return [detections[i] for i in indices]


def run_detection(frame):
    start_detection = time.time()
    tiles, coords = generate_tiles(frame)
    results = model.predict_batch(tiles)

    detections = []
    for result, (dx, dy) in zip(results, coords):
        for det in result.results:
            if det["score"] >= CONF_THRESHOLD:
                det["bbox"] = [
                    det["bbox"][0] + dx,
                    det["bbox"][1] + dy,
                    det["bbox"][2] + dx,
                    det["bbox"][3] + dy,
                ]
                detections.append(det)
    end_detection = time.time()
    detection_time = end_detection - start_detection
    print("Detection time: ",detection_time)

    return apply_nms(detections, IOU_THRESHOLD)

def main():
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            {"format": "RGB888", "size": (2592, 1944)}
        )
    )
    picam2.start()

    tracker = None
    tracking = False
    last_detection_time = 0
    fps = 0.0

    try:
        while True:
            start_time = time.time()
            frame = picam2.capture_array()

            current_time = time.time()
            need_redetect = (current_time - last_detection_time) >= REDETECT_INTERVAL
            print("traccking: ",tracking)
            print("need redetect: ",need_redetect)
            if not tracking or need_redetect:
                detections = run_detection(frame)
                last_detection_time = current_time
                print("detecting: ", detections)
                if detections:
                    best_det = max(detections, key=lambda d: d["score"])
                    x1, y1, x2, y2 = map(int, best_det["bbox"])
                    w, h = x2 - x1, y2 - y1
                    print(f"[DEBUG] BBox coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}, width={w}, height={h}")

                    # Check if the bounding box is valid
                    if w > 0 and h > 0:
                        tracker = cv2.TrackerCSRT_create()
                        try:
                            tracker.init(frame, (x1, y1, w, h))
                            tracking = True
                            print(f"[DEBUG] tracker.init(...) returned: {tracking} (type: {type(tracking)})")
                        except Exception as e:
                            print(f"[ERROR] Exception during tracker.init(): {e}")
                            tracking = False
                    else:
                        print("[WARNING] Invalid bounding box dimensions ? skipping tracker initialization.")
                        tracking = False

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Re-Detected", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    print("tracking deactivating")
                    tracking = False  # no new object found

            elif tracking:
                success, bbox = tracker.update(frame)
                if success:
                    print("tracking")
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "Tracking", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    print("tracking lost")
                    tracking = False  # tracking lost

            # FPS calculation
            fps = 0.9 * fps + 0.1 * (1.0 / (time.time() - start_time))
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (frame.shape[1] - 120, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Tiled Detection + CSRT Tracking", frame)
            if cv2.waitKey(1) & 0xFF in [ord("q"), ord("x")]:
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
