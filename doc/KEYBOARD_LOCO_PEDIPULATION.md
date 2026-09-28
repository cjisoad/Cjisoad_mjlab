# Loco Pedipulation Keyboard Playback

Launch the dedicated native MuJoCo player after training a checkpoint with the current 48-value actor interface:

```bash
python scripts/keyboard_loco_pedipulation.py \
  --checkpoint-file logs/rsl_rl/loco_pedipulation/RUN/model_XXXX.pt
```

`model_14900.pt` from `2026-09-24_03-13-35` has a 65-value actor input and is incompatible with this interface.

The playback window runs one Go2 robot. Click it before sending commands.

| Key | Command |
|---|---|
| W / S | Increase forward / backward velocity while held |
| A / D | Increase / decrease yaw velocity while held |
| Up / Down | Increase / decrease right-front target Y offset |
| Left / Right | Decrease / increase right-front target X offset |
| Q / E | Raise / lower the right-front target |
| R | Clear the right-front target and return to quadruped locomotion |
| Backspace | Reset robot and clear all keyboard commands |
| Space | Pause / resume simulation |

The controller is the same `KeyboardController` used by Pedipulation playback:
velocity ramps while its keys are held and becomes zero on release; foot offsets
persist until changed, cleared with R, or reset. Losing focus clears velocity and
prevents target changes. Focused controller keys do not activate conflicting
MuJoCo display shortcuts.

A nonzero foot offset enables the existing Loco Pedipulation right-front phase
machine (`PREPARE`, `REACH`, `HOLD`, `LOWER`, `RECOVER`). The target is expressed
in the robot's gravity-aligned base frame: X forward, Y left, Z upward. The
script adds the keyboard Z delta to the task's 7 cm lifting baseline and calls
the task's existing `set_command()` method. That method maps the requested
Cartesian point to its tested, reachable FK target bank; it does not run a new
IK solver or alter policy actions. A zero offset disables manipulation and
returns to the normal four-foot gait.

The overlay displays simulation state, focused input state, velocity, target
offset, target state and current FR phase. To capture it for diagnostics:

```bash
python scripts/keyboard_loco_pedipulation.py \
  --status-file outputs/loco_keyboard_status.json
```

Use an explicit model or runtime device when needed:

```bash
python scripts/keyboard_loco_pedipulation.py \
  --checkpoint-file /absolute/path/model.pt \
  --device cuda:0
```

The native keyboard path requires Linux X11 with `DISPLAY` set. It reads only
the listed keys when the focused playback window belongs to this process; it
installs no global keyboard hook.

Run the headless checkpoint verification with:

```bash
python scripts/keyboard_loco_pedipulation.py --smoke --device cuda:0
```

The smoke test loads the checkpoint, checks finite inference and observations,
keeps a manual right-front target active across automatic command-resampling
intervals, and verifies reset clears the target and releases manual control.
