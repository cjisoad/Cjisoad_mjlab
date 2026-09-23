# Loco Pedipulation

`loco_pedipulation` extends the standard Go2 flat locomotion configuration with
optional right-front (FR) Cartesian foot control. A single PPO policy controls
all 12 joints. The task combines quadruped standing/walking, tripod standing,
and low-speed tripod locomotion. It does not include object interaction.

## Commands and Anchor Frame

The `twist` command term coordinates both command types:

- `command`: `[vx, vy, wz]`, in m/s and rad/s, after velocity ramping.
- `enabled`: an explicit per-environment FR enable flag.
- `requested_offset`: `[dx, dy, dz]`, in meters from the nominal standing FR site.
- `reference`: the current interpolated FR reference in the anchor frame.

The anchor origin is the current `base_link` origin. Its Z axis is opposite
gravity, X is the horizontal projection of body **+X**, and Y is `Z cross X`.
This is distinct from the rear-leg `Pedipulation` task's body **-Z** projection.
The frame translates and turns with the robot while remaining horizontal:

```text
p_goal_W = p_base_W + R_WH * (p_zero_H + requested_offset)
```

Velocity rewards also use this anchor frame. Goals are relative to the moving
base, not fixed world points. Disabling FR control is independent of the target
coordinates; zero input offsets do not implicitly disable control.

The nominal FR site is approximately `[0.1934, -0.11509, -0.27302]` m relative
to the base. Targets come from a MuJoCo FK grid of FR hip/thigh/calf angles,
bounded by joint limits and offsets `+/-[0.10, 0.07, 0.14]` m. The vertical
offset is limited to `[0.05, 0.12]` m above the nominal standing site. With the
feet resting on the plane, this corresponds to 5-12 cm sole clearance.

Startup validation solves nine points along each nominal-to-target Cartesian
segment with SciPy least squares and MuJoCo Jacobians. It checks foot/calf
clearance from the torso and left front leg. The bank retains a joint-space FK
witness for each endpoint. Manual target inputs are projected to the nearest
bank target, including zero or out-of-range inputs; the green marker shows the
accepted target. Initial manipulation stages use a smaller subset.

These checks cover the reference quadruped posture and nominal-to-target
segments. Arbitrary body tilt, target-to-target transitions and actual policy
trajectories are not guaranteed collision-free or dynamically balanced. The
task is intended for flat terrain and conservative foot excursions.

## Transitions and Observations

```text
QUAD -> PREPARE -> REACH -> HOLD -> LOWER -> RECOVER -> QUAD
```

Preparation lasts 0.3 s, reaching/lowering 0.7 s, and recovery 0.3 s. The foot
reference uses quintic interpolation; manipulation reward masks blend during
reaching and recovery. The landing reference follows the plane height and
current base height. Returning to quadruped gait requires 0.06 s of FR contact
and completion of the lowering trajectory. Failure to land within another
2 s terminates the episode. Commands can be canceled, retargeted or reenabled
during transitions. Reset reference initialization waits for fresh kinematics.

All modes share the same **65D actor**, **92D critic**, and **12D action**
contracts. Actor observations retain flat locomotion proprioception and phase,
then append 18 values: enable (1), manipulation blend (1), one-hot phase (6),
transition progress (1), accepted goal offset (3), current reference offset (3),
and FR reference error (3). The goal is visible during preparation. FR error
is computable from joint kinematics and attitude. Critic also observes anchor
linear velocity, foot heights, air times, contacts and contact forces.

Original `velocity` and `Pedipulation` checkpoints have incompatible inputs.
Train this task with its own checkpoints. Its own checkpoints preserve the
curriculum stage, evaluation window and environment step counter on resume.

## Rewards and Training

The flat task supplies the robot, PD position actions, proprioception, domain
randomization, general stability costs and PPO settings. FR control adds:

- FR reference tracking (weight 2.5, position tolerance 0.05 m).
- FR ground-contact penalty during HOLD (weight -1).
- Stationary support contact reward and nominal body height reward (0.5 each).
- FR exclusion from manipulation-mode posture, swing clearance and support costs.
- Relaxed posture of the other legs so the body can shift to balance.
- A 1.2 s tripod gait with 0.8 stance fraction and sequential FL/RR/RL swings.

The four-leg gait uses the original 0.6 s diagonal timing. Stationary modes do
not impose a stepping gait. Tripod walking permits support-leg swing and thus
temporary two-foot support. A strong penalty for non-foot ground contact and
60-degree orientation termination remain active. Mirror augmentation is not
used for this right-only task.

Curriculum stages:

| Stage | FR targets | Tripod velocity limits `[vx, vy, wz]` |
| --- | --- | --- |
| 0 | Disabled; train quadruped locomotion | Not applicable |
| 1 | Small targets, stationary tripod practice | `[0, 0, 0]` |
| 2 | Small targets plus slow tripod motion | `[0.2, 0.1, 0.2]` |
| 3 | Full target bank and tripod motion | `[0.35, 0.15, 0.4]` |

After stage 0, FR targets occur with probability 0.4 at command resampling
(every 4-7 s). Quadruped samples therefore remain in training. Stand commands
and FR enable are sampled together with explicit stationary tripod examples;
stage 1 forces stationary FR control. Linear/angular command changes are ramped.

Advancement requires at least 12000 control steps per stage, 64 completed
episodes per evaluation window, quadruped velocity error below 0.25 and at
least 80% survival to timeout. Manipulation stages also require mean FR error
below 0.05 m, HOLD contact fraction below 10%, and at least 80% landing success
over ten or more landing attempts. Stage 2 additionally requires tripod
velocity error below 0.15. Velocity error is XY speed error plus 0.25 times yaw
rate error. Failed windows are discarded so early exploration does not
permanently dilute later success. Thresholds are initial training settings,
not measured policy performance.

Useful logs include `Curriculum/manipulation/stage`,
`Metrics/twist/quad_velocity_error`, `fr_position_error`, `fr_contact_fraction`,
`tripod_velocity_error`, and `landing_success`. Command metrics are conditioned
on the relevant mode and return zero when that mode has no samples.

## Run

From the repository root, with the existing environment activated:

```bash
python scripts/list_envs.py --keyword loco_pedipulation
python scripts/train.py loco_pedipulation --env.scene.num-envs 1024
```

Defaults are 4096 environments, 15000 PPO iterations, TensorBoard logging and
no model upload. Logs and checkpoints go to `logs/rsl_rl/loco_pedipulation/`.
To practice the mixed task immediately, bypass automatic curriculum explicitly:

```bash
python scripts/train.py loco_pedipulation --env.scene.num-envs 1024 \
  --env.commands.twist.initial-stage 3 \
  --env.commands.twist.curriculum-enabled False
```

Play a trained checkpoint with interactive Viser controls:

```bash
python scripts/play.py loco_pedipulation --num-envs 1 --viewer viser \
  --checkpoint-file logs/rsl_rl/loco_pedipulation/RUN/model_14999.pt
```

Replace `RUN` and the checkpoint name with the actual output. In the task's
control folder, enable Manual, set velocity and FR enable, then adjust the
three FR offset sliders. Green spheres show accepted goals; orange spheres show
current interpolated references. Stop requests zero velocity and FR lowering.
Turning Manual off resumes sampled commands. Play defaults to stage 3. The
native viewer can display sampled commands but does not expose these sliders.

For programmatic control:

```python
term = env.command_manager.get_term('twist')
term.set_command([0], velocity=(0.15, 0., 0.),
                 fr_enabled=True, target_offset=(0.02, -0.02, 0.08))
term.set_command([0], fr_enabled=False)
term.release_manual([0])
```

Manual inputs still follow the selected stage's tripod velocity limits.

## Validation

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_*.py'
CUDA_VISIBLE_DEVICES='' python scripts/smoke_loco_pedipulation.py --ppo
python scripts/smoke_loco_pedipulation.py --device cuda:0 --ppo
```

The smoke check uses two environments, shortened transition times and zero
actions to verify command scheduling/contact plumbing. It also checks partial
and timeout resets, one PPO update, a fresh runner's checkpoint restoration,
curriculum persistence and exported ONNX validity. These are implementation
checks; successful tripod control and switching with a trained policy still
require training and evaluation.

Verification on 2026-09-23:

- Full unittest discovery reported `Ran 72 tests`, `OK (skipped=4)`, including
  all 12 new task tests. Existing X11/source-oracle checks skipped when their
  external prerequisites were unavailable.
- CPU and CUDA smoke checks both passed rollout, resets, one PPO update,
  checkpoint/curriculum restoration and ONNX validation.
- Train/play CLI parsers recognize the task. Viser controls were instantiated
  and exercised for enable, velocity changes, target retention and manual release.
- Python compilation and trailing-whitespace checks passed. Ruff was not
  available in the environment.
- No full policy training or convergence assessment was performed.
