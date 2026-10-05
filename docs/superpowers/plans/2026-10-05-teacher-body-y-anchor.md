# Teacher body-Y anchor implementation plan

**Goal:** Apply the approved `normalize(body_y × gravity_up)` frame to both teachers, audit reward effects, and prepare fresh remote training.

**Architecture:** A teacher-only anchor module supplies both command classes. All existing reward and observation transforms consume the command's basis. Original tasks and the current student retain their existing frame. A new checkpoint contract identifies the changed semantics.

**Tech stack:** Python, PyTorch, mjlab, MuJoCo/Warp, RSL-RL, W&B.

## Steps

- [x] Add geometry and consumer regression tests, and observe expected failures against the current X-Z definition.
- [x] Add `src/tasks/teacher_common/anchor_frame.py`: up is normalized negative gravity; forward is normalized body-Y cross up; near-zero cross products use the existing deterministic horizontal reference fallback.
- [x] Switch both teacher command classes to this basis. Record fallback incidence through teacher metrics. Use checkpoint contract version 2 and anchor ID `teacher_body_y_cross_up_v1`.
- [x] Audit all source reward functions and their coordinate consumers. Verify weights, parameters, ordering, mirrored observations, heading control, foot target decoding, and unchanged original/student behavior.
- [x] Run focused CPU regressions and a CUDA reward comparison on fixed simulator states; document direct frame effects separately from indirect changes in future trajectories.
- [ ] Package the exact source and patch; verify remote hashes. Preserve the existing run/checkpoints and apply the user's selected scheduling order for two fresh teacher runs.
- [ ] Record experiment paths, W&B IDs, startup/checkpoint evidence, and remaining limitations.

## Validation commands

```bash
MUJOCO_GL=disable MPLCONFIGDIR=/tmp/mjlab_matplotlib PYTHONPATH=.:tests /home/eden/miniconda3/envs/mjlab/bin/python -m unittest -v test_teacher_anchor test_teacher_tasks test_leg_manip_anchor
```

CUDA reward audit compares each reward at the same physical state and fixed six-channel command under the old and new bases. It also compares automatic biped heading-generated wz separately. Do not attribute later policy improvement to startup checks.
