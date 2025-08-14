from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
# Use a 32‑bit format for DRM and a modest size the display can scale easily
config = picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (1280, 720)}
)
picam2.configure(config)
picam2.start_preview(Preview.DRM)
picam2.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    picam2.stop_preview()
    picam2.stop()
