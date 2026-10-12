# Agentic Elevator P0–P4 Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans. Work in the authorized current workspace without committing unrelated changes.

**Goal:** Run the existing teacher/FSM in the elevator world and expose synchronized RGBD, ground-truth state, and bounded action execution through a local service.

**Architecture:** Extend `load_full_chain()` with an optional configuration callback. Compose the scene with `SceneCfg.spec_fn`, adapt the existing elevator controller to the authoritative simulation arrays at each physics substep, and keep rendering snapshots read-only. A separate observation package implements coordinates, RGBD, sampling, delivery cursors, and resources. A synchronous runtime and HTTP service provide round-based action/observe/reset operations.

**Tech Stack:** Existing Python 3.11 environment, MuJoCo 3.5, mjlab, PyTorch, NumPy, pytest, standard-library HTTP.

## P0 — One world, existing policies

- [x] Add `tests/test_agentic_elevator_scene.py`: compile the composed spec; assert one robot, ten buttons, camera, no infinite terrain, scene contacts, and adapter state isolation.
- [x] Implement `src/agentic_robot/scene.py`, `elevator_adapter.py`; preserve teacher physics and use named environment IDs. The adapter writes only its ctrl and applied-force indices and reuses button queue/debounce logic.
- [x] Add a backward-compatible `configure_env` keyword to `scripts/keyboard_transition_fsm.py:load_full_chain()`; execute it before environment construction.
- [ ] Verify real rigid-body button actuation. Robot floor contact and locomotion are verified; the call-panel support collision currently causes a physical failure before the button is reached. See [verification record](../../agentic_robot_verification_20261012.md).
- [x] Run baseline elevator controller regressions.

## P1 — Structured observation

- [x] Write tests for rotated robot poses, anchor-to-world round trips, all ten surface positions, button compression, and car elevation.
- [x] Implement `src/agentic_robot/observation/state.py` and `controls.py`. Reuse the teacher anchor contract and command zero. Return stable IDs and current contact surface positions with outward normals.
- [x] Verify deterministic transformations against MuJoCo forward kinematics.

## P2 — RGBD

- [x] Write image tests using known geometry and camera pose.
- [x] Implement `observation/rgbd.py`: same frozen state for RGB/depth, metric optical-Z depth, mask, K, optical camera pose, PNG/NPY resources, and explicit failure status.
- [x] Verify image shapes, depth, camera pose, robot-attached view, and lighting changes under EGL.

## P3 — Sampling and delivery

- [x] Write delivery tests for 2/1/0 frames, retries, explicit acknowledgement, per-consumer cursors, dropped frames and episode reset.
- [x] Implement `observation/buffer.py` and `service.py`. Sampling uses simulation time, starts at t=0, and runs during action execution. `observe()` does not advance physics. Acknowledge only after successful model input.
- [x] Restrict resource access to generated resource IDs, with correct MIME types.
- [x] Run observation test suite.

## P4 — Runtime and local tools

- [x] Write runtime validation tests for command fields and time-step rounding; write HTTP tests for action/observe/ack/reset/resource/errors.
- [x] Implement `src/agentic_robot/runtime.py` and `server.py`, `scripts/run_agentic_elevator.py`. Reuse `FullChainPolicy` and `term.submit()`, run bounded steps, stop on physical failure, and report commands/results/evaluation events.
- [x] Disable training rewards and finite training episode timeout; coordinate reset with scene state and new observation episode IDs. Do not silently reset inside an action.
- [x] Document launch and request sequence in `src/agentic_robot/README.md`; add a bounded smoke mode producing an observation report.
- [x] Run EGL RGBD smoke, regressions, CUDA probe and 1-second locomotion check. Record evidence and limitations in [verification record](../../agentic_robot_verification_20261012.md).

## Review and completion

- [x] Review spec coverage and runtime stop checks; owner-gate reference tracking so QUADRUPED/BIPED locomotion is not compared with a transition-only reference. Physical failure remains active for every owner.
- [x] Rerun affected unit, HTTP, EGL, scene and elevator tests; record validation evidence and the remaining button contact gap.
