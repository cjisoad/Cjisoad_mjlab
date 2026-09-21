# Go2 Rear-Leg Stand

`Unitree-Go2-RearStand` migrates the executable `go2_handstand` task from
`My_unitree_go2_gym`. Despite its original name, its front feet FL/FR are
rewarded for staying airborne and its rear feet RL/RR provide support.
This task includes upright velocity tracking and stepping, not only static balance.

## Run

From this repository, activate the existing environment:

```bash
conda activate mjlab
python scripts/list_envs.py
```

Resource-limited validation hides CUDA, uses two CPU simulation worlds, one
Torch/BLAS thread, no renderer, and temporary logs. The script caps rollout
length at 64 steps; `--ppo` adds one short PPO iteration and checkpoint reload.

```bash
nice -n 15 python scripts/smoke_go2_stand.py --steps 16 --ppo
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_stand*.py'
```

The reward-oracle tests use `GO2_STAND_SOURCE` or the original checkout at
`/home/eden/project/My_unitree_go2_gym`. They extract methods with Python AST;
Isaac Gym is not imported. Without the source checkout these tests explicitly
skip. Production training only needs this repository and its installed packages.

After the currently running training has released enough resources, a small
training run can use:

```bash
python scripts/train.py Unitree-Go2-RearStand \
  --env.scene.num-envs 64 --agent.max-iterations 100 \
  --agent.logger tensorboard --agent.run-name rear_stand_trial
```

The task's source-matching default is **4096 worlds / 15000 iterations**.
Do not omit the environment-count override when GPU capacity is constrained.
Logs/checkpoints are under `logs/rsl_rl/go2_stand/`.

```bash
python scripts/play.py Unitree-Go2-RearStand --num-envs 1 \
  --checkpoint-file logs/rsl_rl/go2_stand/RUN/model_100.pt
```

Replace `RUN/model_100.pt` with an existing checkpoint. The migrated task does
not reuse the old task's hardcoded play command `[0, 0, 1]`; play uses the same
command generator with noise and pushes disabled. Existing source checkpoints
are not required. Resuming an old source optimizer state is not supported or
validated; newly trained checkpoints can be saved, loaded and exported normally.

## Layout

| Module | Responsibility |
| --- | --- |
| `src/tasks/stand/env_cfg.py` | Native manager configuration and term ordering |
| `src/tasks/stand/robot.py` | Task-local Go2 model and explicit PD actuators |
| `src/tasks/stand/source_physics.json` | Source URDF physics snapshot and SHA256 provenance |
| `src/tasks/stand/constants.py` | Canonical policy joint order, poses and reward weights |
| `src/tasks/stand/mdp/actions.py` | Scaled position targets and substep action delay |
| `src/tasks/stand/mdp/commands.py` | Heading commands and source resampling rules |
| `src/tasks/stand/mdp/observations.py` | Actor state and shared asymmetric critic input |
| `src/tasks/stand/mdp/rewards.py` | Individual source reward terms |
| `src/tasks/stand/mdp/events.py` | Startup randomization, resets and pushes |
| `src/tasks/stand/rl/` | PPO configuration and exact mirror-loss adapter |

The environment remains `ManagerBasedRlEnv`. Existing velocity/tracking tasks,
their model assets, and installed mjlab/RSL-RL source files are unchanged.

## Preserved Training Contract

- Plane, no effective terrain curriculum or height scan, 0.005 s physics step,
  decimation 4, 50 Hz policy, 20 s episode, strict `episode_step > 1000` timeout.
- Canonical joint order FL, FR, RL, RR, each hip/thigh/calf. Twelve actions,
  clipped to +/-100, target `q_default + .25 * action + motor_offset`.
  Explicit PD gains 40/1 and source torque limits 23.7/23.7/35.55 Nm per leg.
- Independent delay switch in substeps 0..3 for each world and policy step.
  The terminal action is retained across resets, matching source action history.
- Actor 45D: body angular velocity x.25, gravity, commands x[2,2,.25], joint
  offsets, joint velocity x.05, action. Uniform noise follows the source's
  per-block amplitudes; clipping is +/-100 and no running normalization is used.
- Critic 86D: body linear velocity x2, the SAME noisy actor sample, 34 domain
  parameters, four foot vertical-force contact bits. Source quirks are preserved:
  the friction observation is zero and restitution appears twice.
- All 23 active reward formulas/weights, alphabetical evaluation, .02 dt scaling,
  signed total rewards, global batch-mean height gate >.70. `ang_xz` sees the
  previous gate; subsequent gated rewards see the new `base_height` score.
- Front-foot target .67 m, base target .52 m, rear swing target .06 m with
  1.6 s cycle. Non-foot collision penalty covers every non-foot source body,
  and termination uses only base contact >1 N.
- Commands resample on reset and every 500 episode steps: vx[-.2,.6], vy0,
  heading[-3.14,3.14], 20% all-zero draw, independent 10% xy-zero draw, .1
  linear deadband. Heading feedback is `.5*wrap(error)` clipped to +/-1 and
  overwrites yaw even after an all-zero draw. Lazy command synchronization
  gives tracking rewards the current command before mjlab's command update.
- Reset root identity at z=.42, root velocities U(-.5,.5), zero joint velocity,
  joint angles default x U(.5,1.5). Startup randomization persists across resets.
- Friction[.2,1.2], restitution[0,.3], added base mass[-1,2], link mass[.9,1.1],
  base COM +/- .05 m, P/D multipliers[.9,1.1], motor offsets +/- .035 rad,
  shared joint friction[.01,.1], damping[0,.1], armature[.003,.08].
  Global pushes overwrite vx/vy +/- .4 and angular velocity +/- .6 every 400
  policy steps; vertical velocity is retained.
- PPO actor/critic 512/256/128 ELU, seed1, rollout24, minibatches4, epochs5,
  LR .001 adaptive KL .01, gamma .99, lambda .95, clipping .2, entropy .01,
  value coefficient1, joint gradient norm1, differentiable mirror MSE coefficient1.

The RSL-RL 5 built-in symmetry loss detaches one actor branch. `StandPPO` adds
both symmetry gradients through temporary instance-local optimizer/forward
hooks and retains the upstream PPO update loop. Tests compare the complete
gradient to the source objective, check cleanup after exceptions, and preserve
the source KL epsilon and combined actor/critic norm clipping.

## Engine Boundaries

Mathematical reward parity does not imply identical PhysX/MuJoCo trajectories.
The physical snapshot restores all 29 source links, explicit inertias, joint
ranges and collision ownership, while reusing the target Go2 visual meshes.
Eight collision-only links lack URDF inertials; their tiny masses are inferred
from the source asset density .001 kg/m^3. Cylinders become capsules as requested
by the original importer. Adjacent collision exclusions use MuJoCo rules.

MuJoCo has no identical PhysX restitution/static-dynamic friction controls.
Restitution is mapped to contact damping ratio
`-log(e) / sqrt(pi^2 + log(e)^2)` (critical damping at e=0), with a .02 s
contact time constant. Robot collision priority1 ensures sampled friction and
contact parameters take precedence over plane priority0. Joint friction maps
to MuJoCo frictionloss. Mass factors also scale body inertia; this approximates
PhysX's requested inertia recomputation. These mappings require empirical
calibration for cross-engine trajectory or hardware parity.

The native mjlab lifecycle is retained: reward/termination derived quantities
can lag one physics substep, post-reset observations use fresh simulator state,
and pushes are applied after reward calculation. The source exposes stale
pre-reset velocity/gravity/contact caches on reset steps. That stale-cache
artifact is not reproduced; observation content, noise and layouts are unchanged.
Checkpoint numbering follows the installed RSL-RL version (first update is 0).

## Validation Scope

CPU checks cover source reward numerical parity, mirrored gradients, commands,
action delay/history, observation noise sharing, timeout boundaries, model
structure, physical limits, tiny manager rollouts and a PPO update with save/load.
The smoke script additionally checks physical model randomization against critic
slots, actuator gains, and persistence across episode resets.

These checks establish a runnable migration. They do not establish convergence
or a trained rear-standing policy. GPU throughput, long-run stability, and final
policy performance must be measured when the existing GPU training has finished.

Verified on 2026-09-21 in conda `mjlab`: 49 unit/oracle/model/PPO tests passed;
the two-world smoke completed 16 steps plus timeout/reset and one PPO update,
checkpoint save and reload. The original `scripts/train.py` also completed one
CPU iteration with two worlds and four rollout steps, writing YAML configuration
and `model_0.pt`. The smoke took 10.87 s with peak RSS 1,494,556 KiB (about
1.43 GiB) and no swaps after the initial CPU kernel cache had been built.
CUDA was hidden throughout; Warp's no-CUDA diagnostic in this mode is expected.
