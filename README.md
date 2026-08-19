# HM3D Explorer

Interactive explorer for HM3D scenes using [Habitat-Sim](https://github.com/facebookresearch/habitat-sim). Lets you walk through a scene, view live RGB + depth, and record RGB-D + pose trajectories to disk. Two variants are provided:

- **`hm3d_explorer_fps.py`** — FPS-style controls (mouse look, WASD strictly for movement).

## Requirements

- Python environment with:
  - `habitat-sim`
  - `numpy`
  - `opencv-python` (`cv2`)
  - `pynput`
  - `magnum` (comes with `habitat-sim`)
  - `evdev` — raw mouse input for `hm3d_explorer_fps.py` (see [Requirements for raw input](#requirements-for-raw-input))
  - `python-xlib` — optional, hides the cursor on X11 sessions
- An NVIDIA GPU with drivers set up for EGL/GLX offload (the script hardcodes `__NV_PRIME_RENDER_OFFLOAD`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`, and expects `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` to exist). Adjust these env vars at the top of the script if your setup differs.
- An X11 session (`QT_QPA_PLATFORM=xcb`, `XDG_SESSION_TYPE=x11` are set explicitly) since a live OpenCV window is opened.

## Data

Scenes live at:

```
/shared/HM3D_val_data
```

Each scene is a subfolder named `<id>-<hash>` containing a `.basis.glb` mesh, e.g.:

```
/shared/HM3D_val_data/00868-vd3HHTEpmyA/vd3HHTEpmyA.basis.glb
```

## Running

```bash
# FPS-style variant (mouse look)
python hm3d_explorer_fps.py --scene /shared/HM3D_val_data/00868-vd3HHTEpmyA/vd3HHTEpmyA.basis.glb

# ...without taking exclusive control of the mouse
python hm3d_explorer_fps.py --scene <scene.glb> --no-grab
```

`--scene` is required and must point to an existing `.glb` file, or the script exits with an error.

A window opens ("HM3D Explorer: RGB | Depth" / "HM3D FPS Explorer: RGB | Depth") showing the RGB view side-by-side with a Turbo-colormapped depth view.

## Controls

### `hm3d_explorer_fps.py` (FPS-style, mouse look)

| Input | Action |
|-------|--------|
| Mouse (horizontal) | Turn in place (yaw) |
| `W` / `S` | Move forward / backward |
| `A` / `D` | Strafe left / right |
| `R` | Toggle recording on/off |
| `Esc` | Quit |

- Orientation is driven entirely by horizontal mouse movement; **vertical mouse movement is ignored**, so you can never look up or down (pitch is permanently locked to 0). Rotation is applied directly to the agent every frame rather than in discrete steps.
- Yaw is driven by **raw relative motion read from the input device via evdev** (`REL_X` straight off the kernel), not by tracking the desktop cursor's position. Only `REL_X` is ever read, so pitch is impossible by construction.
- The mouse is **grabbed exclusively** while the explorer runs: the desktop cursor freezes, clicks don't reach other windows, and the pointer can't wander outside the window. `Esc` releases it, and the grab is released on exit (including on crash, since closing the fd releases it). Pass `--no-grab` to disable.
- There is **no pointer acceleration**, so turn rate is a constant multiple of physical movement — moving the same distance slowly or quickly rotates the view by the same amount. This is what makes recorded trajectories reproducible.

### Tuning sensitivity

`MOUSE_SENSITIVITY` is radians of yaw **per raw mouse count** — counts come off the device at its DPI with no acceleration, so this value is *not* comparable to a pixel-based setting. Set it deterministically with:

```
MOUSE_SENSITIVITY = 2*pi / (DPI * inches_per_360)
```

e.g. a 1600 DPI mouse at a typical 10 inches per full turn → `0.00039`. Lower = slower turning.

### Requirements for raw input

Your user must be able to read `/dev/input/event*` — normally by being in the `input` group (`groups | grep input`; add with `sudo usermod -aG input $USER`, then re-login). Needs the `evdev` package. If raw input is unavailable the script warns and falls back to plain cursor-delta tracking, which works but stops turning once the cursor hits a screen edge.

Devices are auto-detected, and combo devices that also carry typing keys (some keyboards expose relative axes for a trackpoint, as do virtual devices like ydotool's) are deliberately excluded — grabbing one would swallow WASD/Esc. Override with `--mouse-device /dev/input/eventN` if needed.

To verify raw input works on your hardware, run `python test_mouse_input.py` and move the mouse; it prints deltas and does not grab the mouse.
- WASD no longer turns the agent — it's pure displacement (1.5 m/s, frame-rate independent) relative to the current facing direction, and diagonal movement (e.g. `W`+`D`) is normalized like in a typical FPS.
- Movement respects the scene's navmesh via `sim.pathfinder.try_step` when a navmesh is loaded for the scene, so you'll slide along walls/obstacles instead of walking through them; if no navmesh is present it falls back to unobstructed movement.

## Recording

Press `R` to start recording. A new session folder is created in the **current working directory**, named:

```
<scene_basename>_<YYYYMMDD_HHMMSS>/
├── images/         # RGB frames, 00000.jpg, 00001.jpg, ...
├── depth_raw/       # raw float32 depth, 00000.npy, 00001.npy, ...
├── poses.txt        # "frame x y z qx qy qz qw" per line
└── metadata.json     # written when recording stops
```

Press `R` again to stop. On stop, `metadata.json` is written with the scene path, frame count, timestamp, and camera intrinsics (pinhole `K` matrix derived from horizontal FOV).

You can start/stop recording multiple times in one session; each start creates a new timestamped folder.

## Exiting

Press `Esc` or close via `Ctrl+C`. On exit the script stops the keyboard listener, closes any open pose file, shuts down the simulator, and destroys the OpenCV window.
