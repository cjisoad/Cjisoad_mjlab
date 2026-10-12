# Agentic Elevator Tools

Start the composed three-floor scene and local control service from the repository root:

```bash
python scripts/run_agentic_elevator.py --device cuda:0
```

The server listens on `http://127.0.0.1:8766` and stays on loopback. `GET /health` reports readiness. `POST /observe` returns current robot pose, FSM owner, all ten world-space control surface poses, and at most two new RGBD frame records. Each image/depth resource is read with `GET /resources/<resource_id>`. A successful model-context insertion is followed by `POST /ack` with the returned consumer ID and cursor; a repeated observation before acknowledgement returns the same frames.

```json
{"consumer_id":"agent-1","max_frames":2}
```

For a new physical action, call `POST /action` with exactly `cmd_v`, `cmd_p`, and `t`. `t` must be within `(0, 1]` simulation seconds. The current command uses the existing teacher frame and FR offset contract. The result reports the accepted state owner, exact executed duration, resulting observation and any physical failure.

```json
{"cmd_v":[0.2,0,0],"cmd_p":[0,0,0],"t":0.5}
```

Use `POST /reset` with `{}` to reset the robot and elevator and create a new episode. Old frame cursors become invalid. A transition owner may ignore a requested command; check `command_accepted` and `owner_after` in the action result.

Run one bounded action and emit a compact report without starting the service:

```bash
python scripts/run_agentic_elevator.py --device cuda:0 --smoke
```

For structural checks without a GPU renderer, add `--no-images`. RGBD uses native MuJoCo EGL rendering on Linux, captures every 0.5 simulation seconds, stores raw metric optical-Z depth in NPY, and returns RGB PNG, depth visualization PNG, and a bit-packed valid-pixel mask. Control positions explicitly identify `position_source=sim_ground_truth` and include controls outside the current image.

The loop is round-based: physics advances during each bounded action and pauses between calls while the agent reads and processes observations.
