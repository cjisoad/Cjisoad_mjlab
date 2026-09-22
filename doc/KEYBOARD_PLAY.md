# Pedipulation Keyboard Playback

Run from any directory using the mjlab Python environment:

```bash
/home/eden/miniconda3/envs/mjlab/bin/python /home/eden/unitree_rl_mjlab/scripts/keyboard_play.py
```

By default this loads `model_13800.pt` from the analyzed W&B run `a3vowvqj`
and its original `540b9f5` source snapshot. A native MuJoCo window opens with
one robot and the policy running on CUDA. Click the window to enable input.

| Key | Command |
|---|---|
| W / S | Increase forward / backward velocity while held |
| A / D | Increase left / right yaw velocity while held |
| Up / Down | Increase / decrease right-front target Y offset |
| Left / Right | Decrease / increase right-front target X offset |
| Q / E | Raise / lower right-front target Z offset |
| R | Clear all three foot offsets; return to the trained default-foot mode |
| Backspace | Reset the robot and clear all commands |
| Space | Pause / resume simulation |

Movement ramps at 1 m/s^2 and 2 rad/s^2. Releasing its keys sets the corresponding
velocity to zero immediately. Opposing keys cancel. Switching to another window
also clears movement. Foot offsets change at 4 cm/s and remain at their current
values when keys are released or the window loses focus. While paused, movement
is zero and offsets do not change. Mouse camera controls remain available.

The status overlay shows commanded velocity, target offsets, measured target
error, input focus and simulation step. A green sphere marks an active target.

Offsets use the training gravity-aligned anchor frame attached to the base:
X is forward, Y is left, Z is upward. This is independent of the viewing camera.
The accepted arrow mapping is Up/Down for Y and Left/Right for X. Limits are:

- Forward velocity: -0.2 to +0.6 m/s; lateral velocity is always zero.
- Yaw velocity: -1 to +1 rad/s.
- Foot offsets: X +/-0.10 m, Y +/-0.08 m, Z +/-0.10 m.
- Z also respects the original workspace's minimum anchor-frame foot height.

Zero XYZ is the trained "no target" sentinel. Intermediate offsets are allowed
for smooth testing; the training sampler used a discrete FK workspace and a
2 cm minimum nonzero offset, so small/intermediate offsets and box corners are
not guaranteed to be training samples or reachable in all poses. No IK overrides
the policy. The controller only supplies the six-dimensional observation command.
The policy still produces all twelve joint actions. Automatic command sampling
and heading feedback are overridden only inside this playback process.

This implementation uses system `libX11` and is supported on Linux X11 desktops
(`DISPLAY` required). It polls only the listed keys and applies them only when
the focused window belongs to the playback process. No global input hook or
additional Python package is installed.

For another checkpoint, supply the matching repository or archived source root:

```bash
python scripts/keyboard_play.py \
  --checkpoint-file /absolute/path/model.pt \
  --source-root /absolute/path/matching/repository
```

For an explicit checkpoint other than the default, omitting `--source-root` uses
the current repository. The selected default fails with a clear error if its
archived source is missing; it never silently switches to the newer reward code.
The source must expose the six-command Pedipulation task (actor 48 / critic 89).

Verification and optional live diagnostics:

```bash
python -m unittest discover -s tests -p 'test_keyboard*.py'
python scripts/keyboard_play.py --smoke
python scripts/keyboard_play.py --status-file outputs/keyboard_status.json
```

The bounded smoke runs 550 steps, checks finite policy outputs and observations,
crosses the old 10-second resampling boundary, and checks command reset.
