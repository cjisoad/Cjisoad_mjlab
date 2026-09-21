# Pedipulation Implementation Plan

**Goal:** Copy the rear-leg stand task into an independent `Pedipulation` task
with a gravity-aligned, body-heading-following velocity frame.

**Architecture:** Keep the stand task unchanged. In `src/tasks/pedipulation`,
copy its physics, actions, PPO and rewards. A command-owned batched frame uses
the horizontal projection of body -Z as forward, opposite gravity as up, and
up cross forward as left. Reset initializes forward from projected body +X.
Projection thresholds 0.15/0.05 implement acquisition/release hysteresis;
reacquisition is limited to 6 radians/second until aligned. Invalid projection
holds the previous heading. Repeated evaluations within a control step use
the same previous-step frame so smoothing is not applied twice. Partial resets
defer orientation reads until MuJoCo has forwarded the reset state.

Commands remain [forward velocity, left velocity, vertical-axis angular velocity].
Rewards and critic velocity use the horizontal frame; actor retains body angular
velocity and gravity without observing the frame itself, as requested by the
user. Actor/critic dimensions are 45/86 and the original mirror layout is kept.
Earlier 48/89 Pedipulation checkpoints are dimensionally incompatible; original
stand checkpoints have different command/reward and critic-velocity semantics.

**Tech Stack:** Existing mjlab managers, PyTorch quaternion math, unittest,
CPU MuJoCo/Warp smoke validation.

## Tasks

- [x] Add `tests/test_pedipulation.py` and first observe missing-task failure.
  Check orthonormality/right handedness, pure vertical motion, tilt-invariant
  tracking, yaw following, singularity hysteresis, rate-limited reacquisition,
  idempotence, partial reset, fresh post-forward state, observation and mirror
  contracts, and independent task registration.
- [x] Copy `src/tasks/stand/` to `src/tasks/pedipulation/`, mechanically rename
  task-local imports/classes, register `Pedipulation`, and use experiment name
  `pedipulation`. Add `mdp/frame.py`; integrate with commands, rewards and critic
  velocity; retain the 45/86 PPO and mirror contracts without frame observations.
- [x] Run CPU unit tests including all existing stand tests:
  `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'`.
- [x] Add `scripts/smoke_pedipulation.py`, validate tiny CPU rollout, resets,
  one PPO update, checkpoint save/load and policy export. Document train/play,
  frame conventions, thresholds, observation dimensions and limitations in
  `doc/PEDIPULATION.md`. Check diff, original task unchanged, and registration.

Implementation runs locally in the user's current checkout; it does not start
a long training job, change installed dependencies, or publish changes.

## Verification Results

- 62 unit tests passed, including 13 new frame/task tests and the original 49.
- Two CPU worlds completed a 16-step rollout, timeout reset and partial reset.
- One PPO update completed with the 45/86 observation contract and mirror loss.
- Checkpoint reload restored policy outputs after deliberately perturbing weights.
- Explicit ONNX export passed `onnx.checker`; the base runner does not export
  ONNX automatically on checkpoint save.
- Task listing and both train/play CLI parsers recognize `Pedipulation`.
- Existing stand/velocity tasks and train/play scripts have no tracked diff.
