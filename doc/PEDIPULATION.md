# Pedipulation

`Pedipulation` is an independent copy of `Unitree-Go2-RearStand`. It preserves
the rear-foot support task, physics randomization, action delay, reward weights,
and PPO settings, but uses a gravity-aligned frame for velocity tracking.
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

The frame is owned by the command term (`stand`, retained as an internal name).
Partial resets affect only selected worlds. Initialization is deferred until
reset state has been forwarded. Reward and observation reads may see different
MuJoCo kinematic stages, as in native mjlab, but use matching frame/velocity
data at each stage. Repeated reads never accumulate additional smoothing.

## Rewards and Observations

Linear tracking compares command XY against `v_H.xy`; turning compares command
WZ against `w_H.z`. Vertical and tilt angular-speed terms use `v_H.z` and
`w_H.xy`. The original source height gate and other 19 reward terms are retained.
Heading feedback uses the azimuth of X_H rather than the near-vertical body X.

Actor observations retain the original 45 values; no frame direction or rotation
is appended. Critic begins with H-frame linear velocity, followed by the shared
noisy actor sample, randomization parameters and contacts. Dimensions are
**45 actor / 86 critic**. Mirror mapping retains the original 45-value layout.
Body angular velocity and projected gravity retain their body-frame semantics.
During degenerate projection and reacquisition, the held/smoothed frame has
history that is not explicitly observed by the actor. This is the requested
observation design; its effect on learning must be evaluated during training.

Earlier 48/89 Pedipulation checkpoints are dimensionally incompatible. RearStand
uses the same 45/86 dimensions but different command/reward and critic-velocity
semantics; it is not an equivalent resume. Train a new policy. Default budgets remain
4096 environments / 15000 iterations, so override them on a small GPU.

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
