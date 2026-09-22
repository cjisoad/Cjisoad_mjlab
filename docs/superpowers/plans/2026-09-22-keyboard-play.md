# Pedipulation Keyboard Playback

Goal: Implement the approved standalone `scripts/keyboard_play.py` and open
the selected model with keyboard control in the native MuJoCo window.

Design: W/S ramps forward/backward velocity, A/D ramps positive/negative yaw.
Movement returns to zero on release. Up/Down changes target +Y/-Y,
Left/Right changes -X/+X, Q/E changes +Z/-Z at 0.04 m/s. Foot offsets persist
after release, R clears the target, Backspace resets the robot and commands,
Space pauses. Velocity limits are [-0.2, 0.6] m/s and [-1, 1] rad/s. Target
limits are +/-[0.10, 0.08, 0.10] m with the original workspace height floor.
Zero offset retains the trained no-target meaning. Targets use the robot's
gravity-aligned anchor frame, not camera coordinates. The limit box is not a
guarantee of reachability in every pose.

MuJoCo's passive callback does not expose release events. Poll X11 key state
on the simulation thread, gated by the focused window's process ID. Override
conflicting mjlab shortcuts. Render only command values, target state and
simulation state in the overlay; document key mappings outside the viewer.

The command subclass overrides random resampling and heading feedback with
manual commands, preserving anchor frame updates and the trained observation
layout. It lives in the playback script and does not modify training code.
Default model/source are the downloaded 13800 checkpoint and archived 540b9f5
source. Explicit checkpoint/source arguments support subsequent experiments.

## Tasks

- [x] Add tests in `tests/test_keyboard_control.py` for mappings, held keys,
  cancellation, release, focus, pause, bounds, reset and timestep invariance.
  Run `python -m unittest discover -s tests -p test_keyboard_control.py` and
  verify failure before implementing `scripts/keyboard_control.py`.
- [x] Implement `scripts/keyboard_x11.py` using system libX11, with explicit
  ctypes signatures and focus gating. Test real press/release with libXtst.
- [x] Implement `scripts/keyboard_play.py`: explicit source selection, manual
  command subclass, playback viewer, numeric overlay, and bounded headless
  smoke that crosses the old random-resampling time and resets the robot.
- [x] Run unit and integration checks, then native-window press/release tests.
  Replace only the prior viewer started for the selected checkpoint. Capture
  the new window and verify simulation step count increases.
- [x] Add `doc/KEYBOARD_PLAY.md` with mapping, limits, launch command and
  source-version behavior. Record verified status and leave viewer running.

## Verified

- Focused controller, manual-command and X11 regression tests pass.
- 550-step CUDA smoke passed against checkpoint 13800 and source 540b9f5.
- Real XTest input verified forward/backward, left/right yaw, simultaneous
  controls, release, target persistence, focus loss, pause, opposing keys,
  target clear and robot reset. The active native window rendered nonblank
  moving frames (mean pixel delta 4.84 between captured frames).
- Read-only review identified the stale-window X11 error race. Regression
  coverage and connection-scoped BadWindow handling were added; unrelated
  X11 errors retain their previous handler.
- Test outputs, native screenshots and live status are under
  `outputs/wandb_a3vowvqj/keyboard_*`.
