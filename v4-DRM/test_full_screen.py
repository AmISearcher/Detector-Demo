from picamera2 import Picamera2, Preview
import time, sys, termios, tty, select

def fb_resolution():
    try:
        with open("/sys/class/graphics/fb0/virtual_size") as f:
            w, h = f.read().strip().split(",")
            return int(w), int(h)
    except Exception:
        return 1280, 720  # safe fallback

def kbhit():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr

def getch_nonblocking():
    if not kbhit():
        return None
    return sys.stdin.read(1)

W, H = fb_resolution()  # match console
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (W, H)}
)
picam2.configure(config)
picam2.start_preview(Preview.DRM)
picam2.start()

# Put terminal into raw mode so we can read single keys
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

try:
    print(f"Running at {W}x{H}. Press 'q' to quit.")
    while True:
        ch = getch_nonblocking()
        if ch in ('q', 'Q'):
            break
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    picam2.stop_preview()
    picam2.stop()
