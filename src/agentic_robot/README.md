# Agentic Elevator Tools

Start the composed three-floor scene and local control service from the repository root:

```bash
python scripts/run_agentic_elevator.py --device cuda:0 --viewer native
```

Select exactly one third-person display with `--viewer native|web|none`:

- `native` (default) opens the MuJoCo desktop window. Drag to orbit/pan and scroll
  to zoom. Left-click a visible button (or its label) to apply a short physical
  plunger force; left-drag rotates and right-drag pans without pressing buttons.
  Clicks queue scene input and do not submit movement commands to the robot.
  Both mouse presses and robot contact use the same button travel thresholds,
  debounce, lamp toggle and elevator-call logic. The window mirrors lamp emission
  and call indicator colors as well as the live robot/elevator state, including scene motion and
  FSM transitions between actions. The GUI has no authority to step/reset the physical world;
  closing the window stops the service. A working desktop `DISPLAY` is required.
- `web` serves the keyboard/RGBD/third-person page at `http://127.0.0.1:8766/`
  and opens no native window. Open the URL in your browser to use it.
- `none` runs the robot API without a third-person display, for headless use.

All modes retain `/action`, `/observe`, `/ack`, `/resources/...` and `/health`.
Native/none disable `/`, `/scene.png` and `/scene-click`, so an old browser tab
cannot render a second scene. First-person RGBD for the model is independent of
the chosen human display; `--no-images` disables only that RGBD stream.

```bash
python scripts/run_agentic_elevator.py --device cuda:0 --viewer web
python scripts/run_agentic_elevator.py --device cuda:0 --viewer none
```

In web mode, the root URL serves a local keyboard and observation page. On a CPU-only host,
use `--device cpu`. Hold W/S to walk, A/D to turn, the arrow keys to adjust the
FR target in X/Y, and Q/E to raise/lower it. R restores the default FR target;
Space stops sending manual commands. Releasing the movement keys holds the robot
in QUADRUPED/BIPED; scene time, elevator/doors/buttons, and active RISE/DOWN
transitions continue. The page shows RGB, depth visualization, structured robot state, and all
control-surface world positions. Its third-person MuJoCo view follows the same
simulation state; left-click a visible physical button to press it, left-drag
to orbit, right-drag to pan, and the mouse wheel to zoom. Each control-surface
row also has a Press button for direct testing. Both mouse paths apply the same
short physical plunger force as the scene's standalone viewer.

The server listens on `http://127.0.0.1:8766` and stays on loopback. `GET /health` reports readiness. `POST /observe` returns current robot pose, FSM owner, all ten world-space control surface poses, and at most two new RGBD frame records. Each image/depth resource is read with `GET /resources/<resource_id>`. A successful model-context insertion is followed by `POST /ack` with the returned consumer ID and cursor; a repeated observation before acknowledgement returns the same frames.

```json
{"consumer_id":"agent-1","max_frames":2}
```

For a new physical action, call `POST /action` with exactly `cmd_v`, `cmd_p`, and `t`. `t` must be within `(0, 1]` simulation seconds. The current command uses the existing teacher frame and FR offset contract. The result reports the accepted state owner, exact executed duration, resulting observation and any physical failure.

Use `POST /button` with `{"button":"call_up_f1"}` to apply a short physical press to a named scene control. `POST /scene-click` accepts the current camera plus normalized image coordinates and presses a button only when MuJoCo picking selects one.

```json
{"cmd_v":[0.2,0,0],"cmd_p":[0,0,0],"t":0.5}
```

Use `POST /reset` with `{}` to reset the robot and elevator and create a new episode. Old frame cursors become invalid. A transition owner may ignore a requested command; check `command_accepted` and `owner_after` in the action result.

Run one bounded action and emit a compact report without starting the service:

```bash
python scripts/run_agentic_elevator.py --device cuda:0 --smoke
```

For structural checks without a GPU renderer, add `--no-images`. RGBD uses native MuJoCo EGL rendering on Linux, captures every 0.5 simulation seconds, stores raw metric optical-Z depth in NPY, and returns RGB PNG, depth visualization PNG, and a bit-packed valid-pixel mask. Control positions explicitly identify `position_source=sim_ground_truth` and include controls outside the current image.

Commands are bounded, while the scene continues running between calls. When the
FSM owner is QUADRUPED or BIPED, only robot joint positions, base pose, velocities
and solver warm-start state are held; lights, buttons, elevator and doors remain
live. RISE/DOWN use the original FSM policies autonomously until they reach the
next manual owner, then hold the robot again. Background steps and actions share
one runtime lock. Action results retain their bounded executed duration; a later
observation may have a newer simulation time and FSM owner. Cancellation or
physical failure holds the robot, including transition owners, until reset or
the cancellation signal is cleared. The smoke command runs without this background loop.

For a fresh external model session after earlier frames were acknowledged,
`POST /observe` additionally accepts `replay_latest: true`. When no unacknowledged
batch exists, it returns up to `max_frames` retained recent frames with their
original capture times, `frames_replayed: true`, and `ack_required: false`.
This read-only replay does not advance physics, change the consumer cursor, or
create a pending batch. New or still-pending frames keep ordinary delivery and
report `frames_replayed: false`, `ack_required: true`. Clients must not ACK a
replayed batch. Omit the option for the existing observation behavior.

## Experimental asynchronous planner

The copied skill `$agentic-elevator-async-control` uses one persistent Codex
planner and one local serial executor. Enable its interfaces explicitly:

```bash
python scripts/run_agentic_elevator.py --device cuda:0 --viewer native --experimental-pipeline
```

`POST /pipeline/start` establishes an objective, action budget and position bounds.
`POST /pipeline/observe` reads immutable snapshots during an executing action;
RGBD resources are pinned by planning requests. The planner views images,
acknowledges `/pipeline/ack`, and submits a fixed JSON plan to `/pipeline/plan`.
It then requests the next snapshot immediately, allowing model work to overlap
the current plan. Plans have at most five <=1-second commands. Nonzero foot
targets use world coordinates and are converted using the current anchor.

One executor checks source age, episode/FSM, bounds and quantized command
duration, and replaces remaining commands only at action boundaries. With no
valid plan the robot holds in manual owners while the scene and active FSM
transitions continue. Direct action/reset/button routes return 409
while the pipeline owns execution. `/pipeline/stop` signals cancellation at a
control-step boundary; `/pipeline/status` reports progress and wall timestamps.

Use `scripts/agentic_pipeline_client.py --help` for transport commands. The
repository skill source is `skills/agentic-elevator-async-control`; its
`references/async-api.md` defines the protocol. The default startup and original
synchronous skill do not enable these experiment interfaces. This integration
uses a persistent Codex subagent; it does not require a separate model API key.

## Repository skills

The synchronous robot workflow is versioned in
[`skills/agentic-elevator-control`](../../skills/agentic-elevator-control/SKILL.md).
The experimental asynchronous workflow is versioned in
[`skills/agentic-elevator-async-control`](../../skills/agentic-elevator-async-control/SKILL.md).
Each directory includes its instructions, interface metadata and API references;
the asynchronous skill also includes its transport client. Clone or roll back
these directories together with the corresponding runtime code.

Relative symlinks under `.agents/skills/` expose both skills to Codex when working
in this repository. Keep those symlinks when cloning; on systems without symlink
support, copy the corresponding skill directories into `.agents/skills/` instead.
Invoke `$agentic-elevator-control` for ordinary building tasks, or explicitly
select `$agentic-elevator-async-control` for asynchronous experiments. Both use
the current checkout root rather than requiring a particular home-directory path.
An older separately installed copy can also appear in skill selectors; choose
the repository path when working with this version of the runtime.
