# Agentic Robot P0–P4 Verification

Date: 2026-10-12

## Implemented

- P0 composed the three-floor elevator and robot in one mjlab world, connected the existing elevator controller, and configured scene contact sensors. The GPU runtime reported all four feet contacting the entry pad after settling, and accepted repeated bounded walking actions.
- P1 returns robot pose and teacher anchor data plus all ten dynamic control surfaces, each marked `position_source=sim_ground_truth`.
- P2 renders robot-attached RGB and metric optical-Z depth through EGL, with intrinsics, camera pose, validity mask, and resource reads.
- P3 samples at 0.5 simulation-second intervals, buffers frame delivery per consumer, supports acknowledgement/retry, and clears resources on episode reset.
- P4 exposes `observe`, `action`, `ack`, `reset`, and resource reads through a loopback-only HTTP service. Actions enforce `0 < t <= 1s` and return execution duration, owner state, stop reason, elevator state and events.

## Verification Evidence

- `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/agentic-mpl MUJOCO_GL=egl pytest -q tests/test_agentic_elevator_scene.py tests/test_agentic_observation_delivery.py tests/test_agentic_observation_rgbd.py tests/test_agentic_observation_state.py tests/test_agentic_runtime.py tests/test_agentic_server.py tests/test_three_floor_elevator.py`: **48 passed**.
- A CPU runtime smoke (`--device cpu --initial-state quadruped --smoke --no-images`) reported ten controls and completed a 0.1-second action without an early stop. The current environment has no CUDA device, so the requested `cuda:0` run could not be repeated here. A prior CUDA/EGL smoke completed with RGBD enabled.
- A 1-second idle command and a 1-second `[0.4, 0, 0]` velocity command completed without an early stop. The latter moved the robot about 0.35 m in its forward direction.
- `python -m compileall -q src/agentic_robot scripts/run_agentic_elevator.py tests/test_agentic_elevator_scene.py tests/test_three_floor_elevator.py` and `git diff --check` passed.
- Runtime regression tests confirm reference position/orientation checks only run while RISE or DOWN owns the robot. Physical-failure checks remain active for all owners.
- The first-floor call button and panel are now centered at `z=0.62m`; its indicator is at `z=0.84m`. The call-panel post remains visible but no longer collides with the lowered button or robot. The MuJoCo contact regression presses `call_up_f1` past the controller threshold and confirms the elevator request path.

## Remaining Acceptance Gap

A direct scripted approach to the first-floor call panel advanced for nine one-second actions. On the next action, the robot contacted the call-panel support and stopped with `physical_failure` before its FR foot reached the button. A second attempt reached `BIPED` after the seven-second rise reference completed. Sending a zero foot target while walking in `BIPED` triggered `DOWN` because `cmd_p.z` crossed the configured descent threshold; that attempt returned to `QUADRUPED` before the high button target and produced no event.

A third attempt retained `BIPED` with `cmd_p.z=0.4` while approaching. The prior call button surface at `[4.25, 1.047, 1.10]m` exceeded the measured FR reach in this configuration, and the robot produced no button event. The button is now lowered to `z=0.62m`; standalone MuJoCo contact testing confirms this button actuates and raises the elevator request.

The full command-to-button-to-elevator event chain through the robot's policy runtime remains unverified. The current environment has no CUDA device, and although the runtime starts and accepts bounded actions on CPU, a robot reach-and-press run has not been completed. For that run, the agent must retain a `BIPED` target above the descent threshold while approaching, then target the lowered surface using `world_to_command()`.
