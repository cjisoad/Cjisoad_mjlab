# Go2 Rear-Leg Stand Migration Plan

> Execute in this session with focused subagent implementation and independent review.

**Goal:** Register a native manager-based Unitree-Go2-RearStand task in unitree_rl_mjlab matching the executable Go2_Handstand source contract.

**Architecture:** Use standalone task config, named robot selectors, action/command terms, reward/observation functions and startup/reset events. Keep mjlab's environment loop and existing tasks unchanged. Use a local PPO adapter only for source symmetry gradients and optimizer semantics unavailable in upstream PPO.

**Tech Stack:** mjlab 1.2, MuJoCo/Warp 3.5, installed RSL-RL, PyTorch; conda mjlab.

## Contract

- Rear support RL/RR; front FL/FR airborne; body-frame gravity target (-1,0,0).
- Canonical policy joint order FL,FR,RL,RR, each hip/thigh/calf, independent of engine indexing.
- 45 actor / 86 critic inputs, one frame, source scaling/uniform noise/clipping. Critic reuses the same noisy actor sample, plus linear velocity, 34 DR entries and 4 contacts. Preserve zero-valued friction observation and duplicated restitution entries explicitly.
- Plane, no effective terrain curriculum, dt .005, decimation 4, 20 s timeout with source strict > comparison; no base-height/orientation termination.
- All nonzero source rewards and original weights, dt scaling, negative totals allowed; alphabetical evaluation and shared height gate preserved.
- Source heading command with 10 s resampling, x [-.2,.6], y0, heading [-pi,pi], 20% initial all-zero draw, independent 10% xy-zero draw and .1 deadband. Heading feedback overwrites yaw even after a zero draw, as in source.
- Explicit PD kp40/kd1, scale .25, torque limits from source; per-policy-step random switch at substeps0..3 from previous action; startup DR distribution and reset ranges preserved.
- Native MuJoCo contacts require solver-specific conversion for restitution and friction; document physical non-equivalence, model differences, and lifecycle ordering rather than claiming trajectories are identical.
- PPO: source architecture/hyperparameters, no observation running normalization, exact differentiable mirror loss (both branches).

## Tasks

- [x] Inspect source, target APIs and running-system resource budget.
- [x] Add CPU contract tests for reward arithmetic, observation layout/noise sharing, action delay, command sampling, reset/termination and task registration; demonstrate missing implementation.
- [x] Implement task-local robot configuration, manager terms and environment factory; preserve executable source behavior.
- [x] Implement and independently test PPO symmetry adapter/configuration.
- [x] Integrate reviewed new files into target repository with required filesystem escalation.
- [x] Run staged CPU unit tests, tiny manager-environment rollout, reset and PPO update smoke. No GPU allocations while existing training consumes 7/8 GiB. Repeat from target after integration.
- [x] Repeat target checks: 49 CPU tests, 2-world rollout/reset/DR checks, PPO update/save/reload, and existing train.py CLI for one CPU iteration. All passed.
- [x] Review parity, record exact validation and runnable commands in migration documentation.

## Resource Limits

Use CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, low process priority, 1-2 simulation worlds, bounded rollout and training iteration count, no renderer. Use /tmp for temporary caches/logs. Do not stop, restart, reconfigure, or attach to the user's training process. A successful smoke test establishes integration, not policy convergence.
