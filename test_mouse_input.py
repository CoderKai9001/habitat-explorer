"""Quick check that raw mouse motion is readable.

Run it, then move your mouse around for a few seconds:
    python test_mouse_input.py
Does not grab the mouse, so your cursor keeps working normally.
"""
import time

from hm3d_explorer_fps import MOUSE_SENSITIVITY, CursorDeltaLook, RawMouseLook

look = RawMouseLook(grab=False)
if not look.available:
    from pynput import mouse
    print("Falling back to cursor tracking.")
    look = CursorDeltaLook(mouse.Controller())

print("\nMove your mouse now — sampling for 8 seconds...\n")
total, frames = 0, 0
t0 = time.time()
while time.time() - t0 < 8:
    dx = look.poll()
    if dx:
        total += dx
        frames += 1
        print(f"  dx={dx:+5d}   yaw so far: {abs(total) * MOUSE_SENSITIVITY:6.2f} rad "
              f"({abs(total) * MOUSE_SENSITIVITY * 57.3:6.1f} deg)")
    time.sleep(0.01)
look.close()

print(f"\nframes with motion: {frames}   net dx: {total}")
if frames:
    print("OK — raw motion is being read; mouse look will work.")
else:
    print("NO MOTION SEEN. If you were moving the mouse, tell me and we'll dig further.")
