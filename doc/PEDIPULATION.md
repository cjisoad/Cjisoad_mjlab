# Pedipulation

`Pedipulation` is an independent copy of `Unitree-Go2-RearStand`. It preserves
the rear-foot support task, physics randomization, action delay and reward weights,
but uses a gravity-aligned anchor frame for velocity tracking and right-front
foot target commands.
The original standing and velocity tasks are unchanged. This task does not yet
add object manipulation, grasping, or a new robot asset.

## Coordinate Convention

Let B denote the body frame, W the world, and H the horizontal command frame.
All basis vectors below are expressed in W:

```
up = -gravity / norm(gravity)
forward = project_to_plane(R_WB * [0, 0, -1], up)
X_H = normalize(forward)
Z_H = up
Y_H = Z_H cross X_H
R_WH = [X_H, Y_H, Z_H]
v_H = transpose(R_WH) * root_link_lin_vel_w
w_H = transpose(R_WH) * root_link_ang_vel_w
```

Commands `[vx, vy, wz]` mean horizontal forward/left linear velocity and angular
velocity about gravity-up. Positive directions do not flip when commands become
negative. The frame follows body heading, not a fixed initial world direction.
Y is orthogonalized using the cross product; it is not the raw tilted body Y.
The tracked point remains the base-link origin; no moving-origin velocity is
subtracted. The frame may be pictured at the base link, but it only expresses
absolute velocity components. Angular velocity about up is not, under arbitrary
tilt, exactly the time derivative of projected heading.

At a horizontal quadruped pose, body -Z has almost no horizontal projection.
Each world initializes X from the horizontal body +X projection after reset.
Projection norm below 0.05 releases tracking and holds the previous heading;
norm at least 0.15 reacquires it. During reacquisition only, heading approaches
the projected target at at most 6 rad/s, then follows it directly. These values
are `frame_release`, `frame_acquire`, `frame_transition_rate` on the task's
`PedipulationCommandCfg`. This temporary transition means X does not match the
projection exactly until aligned. For severe inversions the frame can change
direction; the definition is intended for the standing operating region.

The frame is owned by the command term (`stand`, retained as an internal name)
and exposed as `command.anchor_frame` (`AnchorFrame` in `mdp/anchor_frame.py`).
Partial resets affect only selected worlds. Initialization is deferred until
reset state has been forwarded. Reward and observation reads may see different
MuJoCo kinematic stages, as in native mjlab, but use matching frame/velocity
data at each stage. Repeated reads never accumulate additional smoothing.

## Rewards and Observations

Linear tracking compares command XY against `v_H.xy`; turning compares command
WZ against `w_H.z`. Vertical and tilt angular-speed terms use `v_H.z` and
`w_H.xy`. The original source height gate and other 19 reward terms are retained.
Heading feedback uses the azimuth of X_H rather than the near-vertical body X.

Actor observations have **48 values**: body angular velocity (3), projected
gravity (3), commands (6), joint offsets (12), joint velocities (12), previous
actions (12). Commands are scaled by `[2, 2, .25, 1, 1, 1]` and have no observation
noise. No frame direction or rotation is appended. Critic begins with H-frame
linear velocity, followed by the shared noisy actor sample, 34 randomization
parameters and 4 contact flags, for **89 values**.
Body angular velocity and projected gravity retain their body-frame semantics.
During degenerate projection and reacquisition, the held/smoothed frame has
history that is not explicitly observed by the actor. This is the requested
observation design; its effect on learning must be evaluated during training.

Old 45/86 Pedipulation checkpoints, including `model_14999.pt` from W&B run
`8a4c0ajx`, cannot load directly into the new input layers. Historical 48/89
checkpoints that appended frame direction have a different layout despite equal
dimensions and must not be reused. Train a new policy. Default budgets remain
4096 environments / 15000 iterations, so override them on a small GPU.

## Right Front Foot Targets

The command is `[vx, vy, wz, dx, dy, dz]`. The last three values are FR target
offsets in meters along the same anchor axes used for velocity. Exactly `[0,0,0]`
means no manipulation target. Nonzero offsets are sampled independently of
zero-velocity commands with probability 0.5, at reset and every 10 s.

For positions, the anchor origin is the base-link origin. Let `p0_B` be the FR
site position obtained from MuJoCo forward kinematics at `DESIRED_ANGLES`, with
the base at identity orientation. Define its anchor zero in the reference
upright pose (pitch `-pi/2`) and retain that coordinate under current pitch/roll:

```
p0_H = R_upright * p0_B
p_target_H = p0_H + [dx, dy, dz]
p_target_W = p_base_W + R_WH * p_target_H
```

The zero point is the desired FR endpoint in the reference upright posture,
not a fixed world point or the actual current foot position. Targets move with
the robot's base and heading, but do not follow pitch/roll. This prevents ordinary
body lean from pulling them below the specified anchor height. With no target,
the joint-space reward still directly encourages DESIRED_ANGLES at every pose.

The default target bank uses MuJoCo FK, holding the base, FL and both rear legs
fixed. Only FR hip/thigh/calf vary around `[0, .8, -1.5]` by at most
`[.3, .4, .4]` radians, inside hard joint limits. A 9x9x9 joint grid is filtered
to offset bounds `+/-[.10, .08, .10]` m and a minimum nonzero distance of .02 m.
Retained targets are sampled discretely; the bounding box itself is not sampled
as if every Cartesian point were reachable.

In the reference upright pose (body pitch `-pi/2`), the zero position relative
to the base in H is approximately `[.31131, -.142, .17782]` m. Target Z must be
at least .10 m, above the quadruped torso collision-box top at .057 m in the
base-origin frame. A minimum .02 m torso clearance is enforced even if the
configured height floor is lower. The height bound holds in the anchor frame
regardless of pitch/roll. Reachability is a kinematic guarantee in the reference
upright pose, not under arbitrary body tilt; it is not a dynamic balance or
collision-free trajectory guarantee.

The Cartesian tracking reward is:

```
gate = mean(exp(-5 * abs(base_height - .52))) > .70
active = any([dx, dy, dz] != 0)
r_FR = gate * (exp(-||p_FR_W - p_target_W||^2 / .05^2) if active
               else exp(-sum(abs(q_FR - desired_angles_FR))))
front_height_penalty = abs(z_FL - z_FR) * ((not active) or (not gate))
```

`default_pos_reward_FR` retains weight .5. `feet_height_symmetry` retains weight
-.2. The existing source-compatible height gate is batch-wide, not a per-robot
height threshold, and is identical to the velocity reward gate. Other posture
penalties, including `default_pos_front` and `symmetric_joints`, remain in place
and can still discourage large asymmetric reaches. PPO mirror regularization
is applied only to samples without FR targets because this right-only task has
no left-side target command to represent its reflection. It remains fully
differentiable and returns zero for an all-target minibatch.

Parameters on `PedipulationCommandCfg`: `foot_target_probability`,
`foot_target_joint_delta`, `foot_target_max_offset`, `foot_target_min_height`,
`foot_target_min_offset`, and `foot_target_std`. Setting target probability to
0 trains the no-target case, but retains the 48/89 input layout. Setting it to
1 trains only active targets. Play mode draws active targets as green spheres.

## Run

From `/home/eden/unitree_rl_mjlab`, with the `mjlab` environment activated:

```bash
python scripts/list_envs.py --keyword Pedipulation
python scripts/train.py Pedipulation --env.scene.num-envs 1024 --agent.max-iterations 3000
python scripts/play.py Pedipulation --num-envs 1 --checkpoint-file logs/rsl_rl/pedipulation/RUN/model_2999.pt
```

Replace `RUN` with the actual run directory. Training logs are isolated under
`logs/rsl_rl/pedipulation/`. The copied default logger is W&B and
`upload_model=False`; add `--agent.upload-model True` to upload checkpoints.
Use `--agent.logger tensorboard` for local-only logging.

## Validation

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_*.py'
python scripts/smoke_pedipulation.py --steps 16 --ppo
```

The smoke script runs two CPU worlds, checks observation dimensions, partial and
timeout resets, performs one short PPO update, and saves/reloads a checkpoint
and checks ONNX export in a temporary directory. It does not upload to W&B.
These checks validate the implementation, not long-run policy convergence.
