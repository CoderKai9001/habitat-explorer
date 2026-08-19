import os
import sys
import argparse
import json
import select
import time
import numpy as np
import cv2
from pynput import keyboard, mouse
from magnum import Vector3

try:
    from Xlib import display as xdisplay
    import Xlib.ext.xfixes  # noqa: F401  (registers xfixes_* methods on Display/Window)
    _XLIB_AVAILABLE = True
except ImportError:
    _XLIB_AVAILABLE = False

try:
    import evdev
    from evdev import ecodes
    _EVDEV_AVAILABLE = True
except ImportError:
    _EVDEV_AVAILABLE = False

# --- Graphics & Platform Overrides ---
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

# -----------------------------
# Tunables
# -----------------------------
# Radians of yaw per raw mouse count. NOTE: raw counts are not screen pixels -
# they come straight off the device at its DPI with no pointer acceleration, so
# this value is not comparable to what it meant in a cursor-based setup.
# To set it deterministically:  MOUSE_SENSITIVITY = 2*pi / (DPI * inches_per_360)
# e.g. a 1600 DPI mouse at a typical 10 inches per full turn -> 0.00039.
MOUSE_SENSITIVITY = 0.001
MOVE_SPEED = 1.5            # meters per second
GRAB_MOUSE = True           # take exclusive control of the mouse while exploring

# -----------------------------
# Global State
# -----------------------------
key_states = {"w": False, "s": False, "a": False, "d": False}
yaw = 0.0
recording = False
exit_requested = False


def on_key_press(key):
    global recording, exit_requested
    try:
        if key == keyboard.Key.esc:
            exit_requested = True
            return
        k = key.char.lower()
        if k in key_states:
            key_states[k] = True
        if k == "r":
            recording = not recording
            print("\n[REC]:", "STARTING" if recording else "STOPPING")
    except AttributeError:
        if key == keyboard.Key.esc:
            exit_requested = True


def on_key_release(key):
    try:
        k = key.char.lower()
        if k in key_states:
            key_states[k] = False
    except Exception:
        pass


class CursorManager:
    """Reports screen size, hides/shows the OS cursor (X11 XFixes).

    Degrades gracefully to a no-op + a guessed screen size if python-xlib
    or the XFixes extension isn't available, so the explorer still runs
    (just with a visible cursor) rather than crashing.
    """

    def __init__(self):
        self.display = None
        self.root = None
        self.hidden = False
        self.screen_size = (1920, 1080)
        if not _XLIB_AVAILABLE:
            print("[WARN] python-xlib not found: cursor will stay visible.")
            return
        try:
            self.display = xdisplay.Display()
            screen = self.display.screen()
            self.root = screen.root
            self.screen_size = (screen.width_in_pixels, screen.height_in_pixels)
            if not self.display.has_extension("XFIXES"):
                print("[WARN] X server has no XFIXES extension: cursor will stay visible.")
                self.display = None
                return
            self.display.xfixes_query_version()
        except Exception as e:
            print(f"[WARN] Could not open X display ({e}): cursor will stay visible.")
            self.display = None

    def hide(self):
        if self.display is None:
            return
        try:
            self.root.xfixes_hide_cursor()
            self.display.flush()
            self.hidden = True
        except Exception as e:
            print(f"[WARN] Could not hide cursor (XFixes unavailable?): {e}")

    def show(self):
        if self.display is None or not self.hidden:
            return
        try:
            self.root.xfixes_show_cursor()
            self.display.flush()
        except Exception:
            pass


def _is_pure_pointer(dev):
    """True for devices that are mice/touchpads and nothing else.

    Deliberately rejects combo devices that also deliver real keystrokes
    (some keyboards expose relative axes for a trackpoint or media keys, and
    virtual devices like ydotool's expose everything). Those must never be
    grabbed: an exclusive grab on a device carrying WASD/Esc would swallow
    the keyboard controls and leave the explorer unusable.
    """
    caps = dev.capabilities()
    rel = caps.get(ecodes.EV_REL, [])
    keys = caps.get(ecodes.EV_KEY, [])
    if ecodes.REL_X not in rel or ecodes.REL_Y not in rel:
        return False
    if ecodes.BTN_LEFT not in keys:
        return False
    return not any(k in keys for k in
                   (ecodes.KEY_A, ecodes.KEY_ESC, ecodes.KEY_W, ecodes.KEY_SPACE))


class RawMouseLook:
    """Yaw-only mouse look driven by raw relative motion from the kernel.

    Reads REL_X straight off the input devices via evdev instead of tracking
    the desktop cursor's position. That sidesteps every failure mode the
    cursor-based approach hit on this machine:

      - Wayland compositors refuse pointer warps from X11/XWayland clients,
        so recentering the cursor silently did nothing. The cursor drifted
        until it pinned against a physical screen edge, where further real
        mouse movement produced no position change at all - the view simply
        stopped turning - while any stale reference point produced sustained
        phantom rotation. There is no cursor in this path, so neither can
        happen.
      - Desktop pointer acceleration no longer applies, so turn rate is a
        constant multiple of physical movement regardless of how fast you
        move. Slow and fast movements of the same distance now rotate the
        view by the same amount, which also makes recorded trajectories
        reproducible.

    Only REL_X is read; REL_Y is never inspected, so vertical mouse movement
    cannot affect yaw and the camera can never pitch up or down.
    """

    def __init__(self, grab=GRAB_MOUSE, device_paths=None):
        self.devices = []
        self.grabbed = []
        self.available = False
        if not _EVDEV_AVAILABLE:
            print("[WARN] python-evdev not installed; falling back to cursor tracking.")
            return

        for path in (device_paths or evdev.list_devices()):
            try:
                dev = evdev.InputDevice(path)
            except Exception:
                continue  # unreadable (needs the 'input' group) or vanished
            if device_paths is None and not _is_pure_pointer(dev):
                dev.close()
                continue
            try:
                os.set_blocking(dev.fd, False)
            except Exception:
                pass
            self.devices.append(dev)

        if not self.devices:
            print("[WARN] No readable mouse devices found (is your user in the "
                  "'input' group?); falling back to cursor tracking.")
            return

        self.available = True
        print("Raw mouse input: " + ", ".join(d.name for d in self.devices))

        if grab:
            for dev in self.devices:
                try:
                    dev.grab()
                    self.grabbed.append(dev)
                except Exception as e:
                    print(f"[WARN] Could not grab {dev.name}: {e}")
            if self.grabbed:
                print("Mouse captured exclusively - the desktop cursor will not "
                      "move and clicks won't reach other windows. Esc releases it.")

    def poll(self):
        """Return accumulated raw horizontal motion since the last call."""
        if not self.available:
            return 0
        by_fd = {d.fd: d for d in self.devices}
        ready, _, _ = select.select(list(by_fd), [], [], 0)
        dx = 0
        for fd in ready:
            try:
                for ev in by_fd[fd].read():
                    if ev.type == ecodes.EV_REL and ev.code == ecodes.REL_X:
                        dx += ev.value
            except (BlockingIOError, OSError):
                pass
        return dx

    def close(self):
        for dev in self.grabbed:
            try:
                dev.ungrab()
            except Exception:
                pass
        for dev in self.devices:
            try:
                dev.close()
            except Exception:
                pass


class CursorDeltaLook:
    """Fallback when raw input is unavailable: plain cursor-position deltas.

    No warping is attempted, since that is exactly what Wayland blocks and
    what produced the phantom-rotation spins. The tradeoff is that look
    stops once the cursor reaches a physical screen edge.
    """

    def __init__(self, mouse_ctrl):
        self.ctrl = mouse_ctrl
        self.last_x = mouse_ctrl.position[0]

    def poll(self):
        x = self.ctrl.position[0]
        dx = x - self.last_x
        self.last_x = x
        return dx

    def close(self):
        pass


# -----------------------------
# Helper Functions
# -----------------------------
def get_camera_intrinsics(sensor_spec, height, width):
    """FIXED: Extracts numeric degrees and calculates Pinhole Matrix"""
    hfov_deg = float(sensor_spec.hfov)
    hfov_rad = np.deg2rad(hfov_deg)

    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K


def save_metadata(session_folder, scene_path, rgb_spec, K, num_frames):
    metadata = {
        "scene_path": scene_path,
        "num_frames": num_frames,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera_intrinsics": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "matrix": K.tolist(),
        }
    }
    with open(os.path.join(session_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {session_folder}/metadata.json")


def yaw_forward_right(current_yaw):
    """Forward/right unit vectors (world XZ plane) for a yaw-only orientation.
    Habitat convention: forward is -Z, right is +X at yaw=0."""
    forward = np.array([-np.sin(current_yaw), 0.0, -np.cos(current_yaw)])
    right = np.array([np.cos(current_yaw), 0.0, -np.sin(current_yaw)])
    return forward, right


# -----------------------------
# Main explorer function
# -----------------------------
def run_explorer(scene_path, grab_mouse=GRAB_MOUSE, mouse_devices=None):
    global exit_requested, recording, yaw

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.gpu_device_id = 0

    res = [120, 160]

    sensor_specs = []

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid, rgb_spec.sensor_type = "rgb", habitat_sim.SensorType.COLOR
    rgb_spec.resolution = res
    rgb_spec.position = Vector3(0.0, 1.25, 0.0)
    sensor_specs.append(rgb_spec)

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid, depth_spec.sensor_type = "depth", habitat_sim.SensorType.DEPTH
    depth_spec.resolution = res
    depth_spec.position = Vector3(0.0, 1.25, 0.0)
    sensor_specs.append(depth_spec)

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    # No discrete action_space: movement/rotation are driven directly every
    # frame from mouse yaw + WASD displacement instead of stepped actions.

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    agent = sim.initialize_agent(0)

    # Initialize yaw from the agent's spawn orientation so we don't snap on frame 1.
    yaw = 0.0
    start_state = agent.get_state()
    start_state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    agent.set_state(start_state, reset_sensors=False)

    K = get_camera_intrinsics(rgb_spec, res[0], res[1])
    print(f"Loaded Scene: {scene_path}")
    print(f"Camera Matrix K:\n{K}")

    # Start Keyboard Listener (WASD = move, R = record, Esc = quit)
    kb_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    kb_listener.start()

    # Mouse look (yaw-only, raw relative motion - see RawMouseLook)
    mouse_look = RawMouseLook(grab=grab_mouse, device_paths=mouse_devices)
    if not mouse_look.available:
        mouse_look = CursorDeltaLook(mouse.Controller())

    # Hiding the cursor only has an effect under a real X11 session; on
    # Wayland it is a no-op, but an exclusive grab already freezes the
    # cursor so it cannot wander over the window.
    cursor_mgr = CursorManager()
    cursor_mgr.hide()

    cv2.namedWindow("HM3D FPS Explorer: RGB | Depth", cv2.WINDOW_AUTOSIZE)
    frame_count, session_folder, pose_file = 0, None, None
    base_scene = os.path.splitext(os.path.basename(scene_path))[0]

    last_time = time.time()

    try:
        while not exit_requested:
            now = time.time()
            dt = now - last_time
            last_time = now

            # --- Orientation: apply current mouse-driven yaw (pitch locked to 0) ---
            mouse_dx = mouse_look.poll()
            if mouse_dx:
                yaw -= mouse_dx * MOUSE_SENSITIVITY
                yaw %= 2 * np.pi

            state = agent.get_state()
            state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))

            # --- Displacement: WASD relative to current facing, diagonal-normalized ---
            forward, right = yaw_forward_right(yaw)
            move_vec = np.zeros(3)
            if key_states["w"]:
                move_vec += forward
            if key_states["s"]:
                move_vec -= forward
            if key_states["d"]:
                move_vec += right
            if key_states["a"]:
                move_vec -= right

            norm = np.linalg.norm(move_vec)
            if norm > 1e-6:
                move_vec = (move_vec / norm) * MOVE_SPEED * dt
                target = state.position + move_vec
                if sim.pathfinder.is_loaded:
                    state.position = sim.pathfinder.try_step(state.position, target)
                else:
                    state.position = target

            agent.set_state(state, reset_sensors=False)
            obs = sim.get_sensor_observations()

            # --- Visualization (RGB + Turbo Depth) ---
            rgb_bgr = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)
            depth_vis = (np.clip(obs["depth"], 0, 10.0) / 10.0 * 255).astype(np.uint8)
            depth_bgr = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

            combined_view = np.hstack((rgb_bgr, depth_bgr))
            cv2.imshow("HM3D FPS Explorer: RGB | Depth", combined_view)

            # --- Data Collection Logic ---
            if recording:
                if session_folder is None:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    session_folder = f"{base_scene}_{timestamp}"
                    os.makedirs(os.path.join(session_folder, "images"), exist_ok=True)
                    os.makedirs(os.path.join(session_folder, "depth_raw"), exist_ok=True)
                    pose_file = open(os.path.join(session_folder, "poses.txt"), "w")
                    pose_file.write("# frame x y z qx qy qz qw\n")
                    frame_count = 0
                    print(f"Recording to {session_folder}...")

                cv2.imwrite(os.path.join(session_folder, "images", f"{frame_count:05d}.jpg"), rgb_bgr)
                np.save(os.path.join(session_folder, "depth_raw", f"{frame_count:05d}.npy"), obs["depth"])
                p, r = state.position, state.rotation
                pose_file.write(f"{frame_count} {p[0]} {p[1]} {p[2]} {r.x} {r.y} {r.z} {r.w}\n")
                frame_count += 1

            elif session_folder is not None:
                save_metadata(session_folder, scene_path, rgb_spec, K, frame_count)
                pose_file.close()
                print(f"Recording session ended. {frame_count} frames saved.")
                session_folder = None

            if cv2.waitKey(1) & 0xFF == 27:
                exit_requested = True

            time.sleep(0.01)

    finally:
        print("\nCleaning up and closing...")
        kb_listener.stop()
        mouse_look.close()   # releases the exclusive grab
        cursor_mgr.show()
        if pose_file:
            pose_file.close()
        sim.close()
        cv2.destroyAllWindows()


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help="Path to the .glb file")
    parser.add_argument("--no-grab", action="store_true",
                        help="Don't take exclusive control of the mouse (the desktop "
                             "cursor keeps moving; useful if a grab gets in your way)")
    parser.add_argument("--mouse-device", type=str, nargs="+", default=None,
                        help="Explicit evdev path(s) to read mouse motion from, e.g. "
                             "/dev/input/event5. Default: auto-detect all pointers.")
    args = parser.parse_args()

    if not os.path.exists(args.scene):
        print(f"Error: File {args.scene} not found.")
        sys.exit(1)

    run_explorer(args.scene, grab_mouse=not args.no_grab,
                 mouse_devices=args.mouse_device)
