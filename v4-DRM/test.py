from picamera2 import Picamera2, Preview
import time, sys, termios, tty, select

def kbhit():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr

def getch_nonblocking():
    if not kbhit():
        return None
    return sys.stdin.read(1)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (1920, 1080)}  # use your display's mode
)
picam2.configure(config)
picam2.start_preview(Preview.DRM)
picam2.start()

# Put terminal into raw mode so we can read single keys
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

try:
    print("Running. Press 'q' to quit.")
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
