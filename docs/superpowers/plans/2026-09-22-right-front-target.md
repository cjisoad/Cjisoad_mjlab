# Right Front Foot Target Implementation Plan

**Goal:** Extend Pedipulation with three anchor-frame FR target offsets and
command-aware front-foot rewards, preserving direct contact-force checks.

**Architecture:** Reuse the gravity-aligned velocity frame as `anchor_frame`.
For positions, use the base-link origin. Compute the FR zero point from model
FK at DESIRED_ANGLES in a reference upright pose (body pitch -pi/2). Keep this
anchor coordinate fixed so body pitch/roll cannot lower the target below its
height floor. Zero offset corresponds to DESIRED_ANGLES in that reference pose;
the no-target reward uses desired joint angles under any current body pose.
Commands are `[vx, vy, wz, dx, dy, dz]` in SI units. Zero XYZ means no target;
nonzero XYZ means a target. Sample targets independently of zero-velocity draws.
Default target probability is 0.5 and resampling retains the 10 s cadence.

Build a cached bank of offsets using MuJoCo FK in an upright reference posture,
varying only FR joints within conservative neighborhoods of DESIRED_ANGLES and
inside hard joint limits. Retain points above the quadruped torso collision-box
top plus margin in the base-origin anchor frame, and separated from the zero
target sentinel. This guarantees reference-pose kinematic reach without moving
the support legs; it does not guarantee reach during arbitrary body inversion.

`default_pos_reward_FR` uses exp(-squared Cartesian error / std^2) when active,
otherwise the existing exponential joint error. Both are multiplied by the
existing batch-wide height gate. `feet_height_symmetry` is enabled for no target
OR a closed height gate. Other reward weights and penalties remain unchanged.

Actor/critic dimensions become 48/89. Mask PPO mirror loss for target-active
samples, since a right-only target has no valid left-hand counterpart. Keep
mirror regularization for no-target samples, including both gradient branches.
Existing 45/86 and historical unrelated 48/89 checkpoints are not compatible.

**Tech Stack:** mjlab managers, MuJoCo FK, PyTorch, existing RSL-RL PPO adapter.

## Tasks

- [x] Add tests for command shape, zero point, reference FK reach and height,
  target sampling/reset behavior, height/command gate truth tables, observation
  layout/noise and masked symmetry gradients. Run to demonstrate missing behavior.
- [x] Rename the existing frame and implement cached reference workspace and
  six-dimensional command sampling. Keep FK off the GPU rollout hot path.
- [x] Implement gated reward switching, 48/89 observations and PPO contract;
  update existing tests and CPU smoke assertions.
- [x] Run the focused tests, full repository tests and CPU rollout/PPO/save/load/
  ONNX smoke; inspect workspace visualization derived from actual FK.
- [x] Update task documentation with exact coordinate conventions, tunables,
  remaining posture penalties, compatibility limits and verified scope.

## Validation Commands

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_pedipulation*.py'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_*.py'
python scripts/smoke_pedipulation.py --steps 16 --ppo
```
