# Loco Pedipulation Keyboard Playback Design

## Goal

Provide a native MuJoCo playback command for the trained Loco Pedipulation checkpoint at `logs/rsl_rl/loco_pedipulation/2026-09-24_03-13-35/model_14900.pt`, using the same keyboard behavior as the existing Pedipulation controller.

## Entry point

Add `scripts/keyboard_loco_pedipulation.py`. Its default checkpoint is the specified `model_14900.pt`; `--checkpoint-file`, `--device`, `--seed`, `--status-file`, `--smoke`, and `--steps` allow equivalent explicit use and headless verification.

## Input behavior

The script reuses `scripts/keyboard_control.py` and `scripts/keyboard_x11.py` unchanged. W/S control forward velocity, A/D yaw, arrows control right-front target X/Y offset, Q/E control target Z offset, R clears the target, Backspace resets, and Space pauses. Focus gating, held-key behavior, velocity limits, and target persistence match `scripts/keyboard_play.py`.

## Command adaptation

The Loco Pedipulation command term already exposes `set_command()`. On each viewer input poll, the script provides all environments with the controller velocity. A nonzero target offset enables the right-front manipulation phase machine and sends the offset through `set_command()`, which projects it to the task's reachable FK target bank. A zero target offset disables right-front manipulation and returns to normal quadruped locomotion. Existing task code, training behavior, and automatic sampling remain unmodified.

## Viewer and diagnostics

A native `NativeMujocoViewer` subclass suppresses only conflicting controller shortcuts. Its overlay displays run state, focused input state, velocity command, target offset, right-front manipulation state, and phase. An optional status JSON records the same information.

## Verification

Add an isolated unit test for the Loco Pedipulation keyboard-command bridge using the task's existing lightweight fake environment. The headless smoke loads the checkpoint, checks manual commands override automatic resampling through the requested interval, verifies finite policy outputs and observations, and verifies reset clears input and releases manipulation.
