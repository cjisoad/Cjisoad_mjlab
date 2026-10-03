# Leg Manip

`leg_manip` fuses `loco_pedipulation` (quadruped/tripod locomotion with
right-front foot control) and `Pedipulation` (rear-leg stand with FR targets)
into one task. A single PPO policy controls all 12 joints across three support
modes: quadruped, tripod (FR manipulating), and biped (rear-leg stand, with
a nominal stance reference or an operation target). The two source tasks remain registered and unchanged.

## Robot and Policy ABI

- Robot: the Pedipulation stand spec snapshot (`src/tasks/pedipulation/robot.py`),
  one asset for all modes. Quadruped init: `DEFAULT_ANGLES` at 0.42 m.
- Actor input: 51 values. The first 48 retain the layout shared by both source tasks
  (`ang_vel*.25` 3, `projected_gravity` 3, `command*(2,2,.25,1,1,1)` 6,
  `joint_pos_rel` 12, `joint_vel*.05` 12, `prev_action` 12), same noise and
  ±100 clamp. Indices 48:51 append absolute base-origin linear velocity in the
  unified anchor frame, scaled by 2. Actor training adds per-axis uniform noise
  ±0.02 m/s before scaling; play is noise-free. The velocity is always available
  in quadruped, tripod, biped and transitions. Mode, group weight and phase
  remain internal/critic-only state.
- Command: the unchanged 6-value ABI `[vx, vy, wz, dx, dy, dz]`; all-zero
  `[dx, dy, dz]` is the no-target sentinel.
- Action: `PedipulationPositionAction` (scale .25, substep delay, motor DR
  interface), a superset of the loco action term.
- Critic: 131 values = shared original noisy actor channels (48) + exact anchor linear
  velocity (3) + DR parameters (34) + per-foot contacts (4) + gait phase (2) +
  FR/mode privileged state (20) + foot height (4) + air time (4) + contact
  force logs (12). Velocity at indices 48:51 has the same physical quantity,
  axes, scale (2), component order and simulation time as the actor velocity;
  only actor velocity has measurement noise. There is no duplicate noisy
  velocity channel in the critic.

## Unified Anchor Frame

One instantaneous gravity-aligned frame serves both stances:

```
up      = -gravity / norm(gravity)
forward = normalize( project_up( R_WB * [1, 0, -1] ) )   # body forward diagonal
left    = up x forward
```

- Level quadruped: the body -Z projection vanishes, so the frame is exactly
  the loco_pedipulation +X frame.
- Full rear-leg stand: the body +X projection vanishes, so it is exactly the
  Pedipulation -Z frame.
- The diagonal keeps a nonzero horizontal projection throughout the rear-up
  pitch sweep. It degenerates near a 45° nose-down pose, where a fixed reference
  keeps values finite but does not ensure heading continuity. That pose can
  occur before the LOCO 60° termination threshold; a continuous heading for
  abnormal forward pitching is a separate change.

Velocity commands, FR targets, velocity observations and all velocity rewards
use this frame with the origin at `base_link`. Linear velocity is the absolute
world velocity rotated into anchor axes; the moving origin is not subtracted.

## Mode Switching and Reward Groups

The command term latches an internal mode from the commanded target height
`dz` (offset from the quadruped-stance FR zero ≈ `[0.1778, -0.1726, -0.3002]` m):

- `dz > tripod_high (0.35)` → request BIPED group
- `dz < biped_low (0.25)`   → request LOCO group (zero target always does)
- inside the 10 cm overlap band `[0.25, 0.35]` → hold the current mode

Constants live in `src/tasks/leg_manip/constants.py`
(`TRIPOD_HIGH = BIPED_LOW + 0.10`, `DZ_MIN = .05`, `BIPED_HIGH = .72`).

The group weight `w_biped` ramps toward the requested group at
`1 / group_transition_time` (default 0.8 s), so reward sets blend smoothly.
Every LOCO term is multiplied by `1 - w`, every BIPED term by `w`, shared
terms are unweighted. Mode switches never trigger FR phase transitions.

### Reward table (41 terms)

Shared: `fr_position_tracking` 2.5 (tracks the smoothstep reference × blend;
automatic stage 0/1 wrappers multiply it by 0.15/0.30; stage 2 and manual
commands keep the full weight),
`fr_ground_contact` −1, `is_terminated` −200, `joint_acc_l2` −2.5e-7,
`action_rate_l2` −0.05, `joint_pos_limits` −10, `torques` −2e-4,
`collision` −2, `alive` +1.

LOCO (`×(1−w)`, loco_pedipulation terms): `track_linear_velocity` 1.0 (tripod
tightening), `track_angular_velocity` 1.0, `pose` (mode_posture) 1.0,
`stand_still` −1.0, `foot_gait` 0.5, `foot_clearance` −1.0, `foot_slip` −0.25,
`soft_landing` −0.001, `support_contact` 0.5, `base_height` 0.5 (nominal
0.322 m), `body_orientation_l2` −1.0, `body_ang_vel` −0.05,
`angular_momentum` −0.025.

BIPED (`×w`, Pedipulation terms; `biped_base_height` runs first and writes the
per-environment height gate): `biped_base_height` 1.5 (0.52 m), `biped_tracking_lin_vel`
2.5, `biped_tracking_ang_vel` 2.5, `biped_lin_vel_z` 0.3, `biped_ang_vel_xy`
0.2, `handstand_orientation` −1.0, `handstand_feet_on_air` 0.4,
`handstand_feet_height_exp` 5.0, `biped_feet_clearance` 0.4, `biped_contact`
0.3, `biped_ang_xz` −0.5, `symmetric_joints` −0.1, `orientation_symmetry` −0.5,
`feet_height_symmetry` −0.2, `default_pos_front` −0.1, `default_pos_rear` −0.1,
`default_hip_pos` −0.1, `default_pos_reward_FL` 0.5, `default_pos_reward_FR`
0.5. The Pedipulation `FR_pos_track` is replaced by the shared
`fr_position_tracking` (global weight 2.5 / std .05). Nominal biped FR commands
keep both front default-joint rewards and symmetry active. Actual operation
commands suppress FR-specific default-posture terms. Front default rewards,
joint symmetry, pose penalties, height, gravity alignment, front-air and static
rear support give feedback outside the height gate. Walking velocity bonuses
retain the per-environment `height_score > .70` gate. At zero command speed,
rear contact requires both rear feet and swing-clearance reward is zero;
walking uses the existing one-rear-contact gait bonus.

## FR Workspace

`mdp/foot_workspace.py` builds one FK bank on the stand spec with two segments
(offsets from the shared quadruped zero):

- LOW (tripod): quadruped default stance, Cartesian path IK + collision
  clearance (loco builder logic adapted to snapshot geom names). 84 targets,
  dz ∈ [0.065, 0.336].
- HIGH (biped): reared reference stance rotated into the unified frame,
  endpoint collision clearance replaces the source torso-top floor so the band
  extends below the head end. 56 targets, dz ∈ [0.276, 0.635].
- Overlap band [0.25, 0.35] is populated from both segments (32 low + 10 high).
- Stage 2 operation samples come from the bank; established modes retain the
  overlap segment, while a mode change samples beyond the hysteresis threshold.
  Stages 0/1 use only the fixed nominal references described below. Manual
  (keyboard/GUI) targets stay continuous. Biped samples use velocity ranges
  (vx ∈ [−0.2, 0.6], vy = 0, wz ∈ [−0.6, 0.6]).

## FR Phase Machine

The phase ABI remains
`QUAD→PREPARE→REACH→HOLD→LOWER→RECOVER→QUAD`. A rise starts with PREPARE;
retargeting an established high target passes only through REACH and HOLD.
The reference uses a fifth-order smoothstep. Automatic nominal rise/descent
uses 1.8 seconds; manual and actual operation targets keep 0.7 seconds.
PREPARE and RECOVER each take 0.3 seconds. Landing reference height follows
the live base height. `landing_failed` retains its existing deadline of
`reach_time + landing_timeout` (0.7 + 2.0 = 2.7 seconds in LOWER).

A phase timer does not count as a successful mode switch. Each rise/descent
attempt must start from a physically ready source stance, reach the correct
endpoint, then hold the endpoint for one continuous second before a
5-second deadline. Healthy walking origins are evaluated against their
previous velocity command; the new switch command is stationary. An invalid
source is an `uncertified` failure, so a failed rise cannot be followed by a
falsely successful descent. Reversal, reset, termination, episode timeout and
window rollover interrupt pending attempts.

## Terminations

`time_out`, `base_contact`, `illegal_contact` (non-foot vs terrain >10 N),
`landing_failed`, and mode-aware `fell_over`: LOCO mode keeps the 60° tilt
limit; BIPED mode only limits the lateral lean (`|gravity_b.y| > sin 60°`),
allowing the full 90° pitch of the stand.

On a BIPED→LOCO mode edge (including canceling a target), a per-environment
descent allowance lasts at most `biped_exit_timeout` (default 3.5 s), independently
of the 0.8 s reward ramp. During descent, lateral lean is still limited and
total tilt above 110° terminates. The allowance ends early when the reward ramp
has reached LOCO, total tilt is at most `biped_exit_recovery_angle` (default 50°),
and at least one front foot has ground contact. Then the normal 60° LOCO limit
resumes. Re-entering BIPED or resetting clears this temporary state; collision,
base contact, landing failure and episode timeout protections remain active.

## Events and Sensors

Events: Pedipulation `randomize_physics` (startup; mass/COM/friction/
restitution/armature/damping/kp/kd/motor offsets + 34-D DR observation),
loco reset events (quadruped start pose; joints unscaled), loco interval
`push_robot` (x/y ±0.15, yaw ±0.2; removed in play). The velocity task's
`foot_friction`/`base_com`/`encoder_bias` are dropped as covered by
`randomize_physics`.

Sensors: `feet_ground_contact` (geom vs terrain, air time),
`nonfoot_ground_touch`, `front_contact`/`rear_contact`/`nonfoot_contact`/
`base_contact` (body level).

## Curriculum

The course has three named stages; `STAGE_BIPED` is a compatibility alias for
stage 2 used by play code, not a migration of old curriculum stages.

| Stage | Goal | Automatic FR reference | Velocity |
| --- | --- | --- | --- |
| 0 `STAGE_STANCE` | Quad stand, rear-leg biped stand, and both switches together | Quad sentinel zero; biped fixed nominal FK offset | All zero |
| 1 `STAGE_WALK` | Walking in both stances, retaining static and switch examples | Same fixed references | 25%, 50%, 75%, then 100% of configured ranges |
| 2 `STAGE_OPERATION` | Reachable random FR operation in both modes, retaining nominal stance/walk examples | 40% operation targets, 60% nominal references | Full configured ranges |

The nominal biped reference is computed from the actual stand spec:
`REARED_ROTATION @ FK(DESIRED_ANGLES) - FK(DEFAULT_ANGLES)` = approximately
`[0.133488, 0.030602, 0.478042]` m. A nonzero nominal reference requests the
biped group but does not mark the FR as manipulating. No random operation
coordinates are sampled in stages 0/1.

Every physical reset starts in the usual quadruped pose and receives a
stationary quad command for an initial 6–9 second dwell. This lets the base
settle from its 0.42 m reset height to its supported height near 0.322 m before
source certification. Later command samples select quad/biped with equal
probability and dwell 6–9 seconds; a mode change is always commanded at zero
speed. In moving stages, stable-mode samples retain a 20% standing probability.
The normal 20-second episodes provide rise and descent opportunities after the
initial dwell; an interrupted endpoint hold remains a failed attempt. The
optional partial biped reset is omitted, so evaluation and training both use
normal quad physical resets.

Static readiness checks the whole body: mode-appropriate gravity alignment
within 15 degrees, quad base height within 0.05 m of nominal or biped height
between 0.46 and 0.60 m, mean joint pose error below 0.20 rad (quad) or 0.25 rad
(biped), required support contacts, base speed below 0.12 m/s and angular speed
below 0.25 rad/s. Quad stand requires four feet; biped stand requires both rear
feet and no front support. Moving readiness allows a 0.40 rad pose error,
requires at least two quad contacts or one rear biped contact, and checks
linear/yaw tracking error below 0.15, vertical speed below 0.10 m/s and angular
speed in the anchor x/y axes below 0.30 rad/s. The observed low-base
(0.267 m), tilted (37 degrees from rear-up), FL-supported posture fails biped
readiness even when FR error is only 1 cm.

Promotion independently checks quad stance, biped stance, rise and descent;
stage 1 additionally checks quad and biped moving groups. Each decision needs
a full window of at least 12,000 control steps and 64 completed episodes.
Each required group needs at least 100 settled samples, 80% ready samples,
mean linear-plus-quarter-yaw tracking error below 0.15, at least 16 completed
window-local episodes, and at least 80% survival. Terminating during a
transition still contributes a failure to the active command group. Overall
episode survival must also be at least 80%. Each switch direction needs at
least 10 attempts and 80% held-endpoint success. Two consecutive passing full
windows are required. Stage 1 repeats those checks at every velocity level
before opening stage 2.

Every decision, including failure, clears the window and restarts its clock.
Outcome samples stream per control step, and episode group membership is
window-local. Last-decision diagnostics retain samples, readiness, errors,
survival and switch conditions after clearing the window, exposing the reason
for a blocked promotion in the curriculum logs.

`CourseLogger` emits separate quad/tripod/biped × stationary/moving metrics,
including joint pose error, mode-relative gravity angle, base height, front and
rear contact counts, height gate, FR error, and velocity tracking. Landing
totals retain their existing four events. Rise/descent totals distinguish
attempts, succeeded, timeouts, interrupted and uncertified origins.

`LegManipOnPolicyRunner` saves `infos['leg_manip_course']`, version 1 and name
`stance_walk_operation`, with stage, window start, all capability/survival
windows, episode counts, pass streak, velocity level, retained decision
diagnostics and cumulative landing/switch counts. On resume, pending lifetime
attempts become interrupted because simulation state resets. Old 0–4
curriculum checkpoints, including observation-migrated historical checkpoints,
are rejected explicitly before loading model/optimizer state. Play configs
start at stage 2 and accept only a checkpoint with this course version.

## Training

```
python scripts/train.py leg_manip --env.scene.num-envs=4096
```

Start the stance course from random initialization with fresh optimizer and
stage-0 state, in a separate run:

```bash
python scripts/train.py leg_manip --gpu-ids '[0]' --env.scene.num-envs 4096 \
  --agent.resume False --agent.max-iterations 15000 \
  --agent.run-name stance_course_scratch --agent.logger wandb \
  --agent.wandb-project pedipulation --agent.save-interval 100 --agent.upload-model True
```

The course requires a fresh start. The velocity migration tool is retained
for historical observation-layout validation; its output does not provide a
compatible stance-course state.

The historical 2026-10-02 velocity-input experiment was recorded as W&B run
[`rc8qgs8f`](https://wandb.ai/980955722-harbin-institute-of-technology/pedipulation/runs/rc8qgs8f).
Its files are under `/pedipulation/legmanip1/anchor_velocity_scratch_20261002_185344`.
The actual launch uses `wandb_project=pedipulation`, `save_interval=100` and
`upload_model=True`; every periodic checkpoint and the final checkpoint are
retained on the mount and uploaded. Startup metadata and paths are recorded in
`outputs/leg_manip_velocity_20261002/validation_report.json`.

Runner: `LegManipOnPolicyRunner` (versioned task-local course persistence +
six-group metrics logging). Algorithm: `LegManipPPO` — the Pedipulation symmetry PPO
(`sym_coef=1.0`, target-free samples only) with 51-D actor / 131-D critic
validation. Velocity reflects as `(vx, -vy, vz)`. `experiment_name=leg_manip`, wandb logger,
15000 iterations, `clip_actions=10`.

New checkpoints also retain observation layout, velocity scale and noise
amplitude in `infos['leg_manip_observations']`.

The fresh CPU PPO smoke constructs the actual environment and runner, performs
two updates, verifies finite policy/value outputs, saves and reloads a versioned
checkpoint, checks policy predictions and persisted course state, and exercises
legacy-version rejection:

```bash
DISPLAY='' MUJOCO_GL=disable MPLCONFIGDIR=/tmp/leg_manip_mpl PYTHONPATH=. \
  python scripts/smoke_leg_manip_course.py --result-file outputs/course_smoke.json
```

For the same brief check on the training device, add
`--device cuda:0 --num-envs 4096`. The historical `scripts/smoke_leg_manip.py`
was built for migrated old checkpoints; use the course smoke for this runner.

## Keyboard Play

```
python scripts/keyboard_leg_manip.py --checkpoint-file <model.pt> --device cuda:0
python scripts/keyboard_leg_manip.py --checkpoint-file <model.pt> --smoke --device cpu
```

Keys: W/S forward, A/D yaw, arrows move the FR target horizontally, Q/E
raise/lower it (dz up to the bank max ≈ 0.64 m), R clears the target. The
overlay shows the FR phase, the latched mode (LOCO/BIPED) and the group
weight. The headless `--smoke` check drives dz through the band and asserts
the biped latch; it expects a checkpoint whose policy survives 550 steps, so
it only passes with a policy trained on this task.

## Validation Notes

- The velocity-input validation ran 189 tests with 3 skips and no failures
  (`DISPLAY='' python -m unittest discover -s tests`), covering
  the unified frame, band constants, two-segment workspace, hysteresis/ramp,
  phase machine, reward gating, terminations, cfg assembly, a real CPU env
  build/step, RL registration and keyboard wiring.
- Real PPO update, save/reload and 51-D ONNX export passed on CPU, as well as
  GPU checks with 1024 and 4096 environments. Numerical results and logs are
  in `outputs/leg_manip_velocity_20261002/`; these brief checks do not establish
  successful stance-course promotion or learned biped behavior.
- Historical legacy compatibility probe (before adding velocity inputs): the loco_pedipulation `model_14999.pt` actor
  (same 48→512→256→128→12 architecture) loads into the leg_manip runner with
  `strict=True`; a 550-step CPU rollout with terminations relaxed confirmed
  command flow, band latch, group ramp and phase sequence. Biped behavior
  itself requires training on this task.


## Command-scaled horizontal velocity tracking

The leg_manip quadruped and biped horizontal velocity kernels use
`sigma = max(0.05, 0.75 * norm(command_xy))` and
`exp(-sum((command_xy - actual_anchor_xy)^2) / sigma^2)`.
The parameters are `min_std=0.05` m/s and `speed_ratio=0.75` in both reward configurations.
The command magnitude includes both horizontal axes and is independent of direction;
angular commands do not change this horizontal tolerance. There is no additional deadband.
At zero command, an actual horizontal speed of 0.08 m/s scores about 0.0773.
At command 0.12 m/s, actual speeds 0, 0.06 and 0.12 m/s score about
0.1690, 0.6412 and 1.0 before group weights and other gates.
Higher commands can increase sigma above the old 0.5 m/s value; this is relative
error scaling, not a global upper bound on tolerance.

The LOCO vertical damping, tripod kernel/interpolation, biped height gate,
group weights and angular tracking are unchanged. Standalone loco_pedipulation
and Pedipulation source rewards are unchanged. Policy observations, network shapes,
exploration and curriculum are unchanged, so existing checkpoints remain loadable.
This affects future learning rewards; loading an old checkpoint alone does not
teach it improved tracking or reduce deterministic playback drift.

## Quadruped/biped gradient diagnostics

`LegManipPPO` observes shared actor MLP gradients before the first PPO update
on a completed rollout. The default interval is 20 updates (including the first
update after startup/resume), with at most 512 randomly selected samples per
group. Set `agent.algorithm.gradient_diagnostics_interval=0` to disable it;
`gradient_diagnostics_max_samples` controls the sample cap.

Labels are captured before each action: quadruped means QUAD, mode=false,
group_weight<=0.01; biped means HOLD, mode=true, group_weight>=0.99.
Reward transitions are excluded. Stance quality is not a filter: unsuccessful
attempts in an assigned mode still contribute. The existing command convention
splits stand/moving at norm(command[vx,vy,wz])>0.05. These are command contexts,
not proof that the robot actually achieved either stance or speed.

Scalars under `GradConflict/` compare quad vs biped, both stand groups, both
moving groups, and stand vs moving within each mode. For each pair:

- `policy/cosine` compares the actual PPO clipped surrogate gradients using the
  rollout's globally normalized advantages, without separate group normalization.
- `actor_total/cosine` adds the existing weighted exact symmetry loss to each
  group's policy objective.
- `first_norm` and `second_norm` give gradient sizes; `samples/` records available
  context samples and `used_samples/` records the sampled subset size.
- Cosine <0 means this measurement has opposing group update directions; >0
  means aligned directions; near 0 means near orthogonal directions.
  `negative_fraction` counts negative cosines among valid diagnostic measurements
  since this process started, and `measurements` reports that denominator.
- `valid=0` means a group is missing, has fewer than two samples, or has effectively
  zero gradient. No artificial zero cosine or negative fraction is logged.

The measured parameters are the shared actor mean MLP, excluding the independent
critic and Gaussian standard deviation. State-independent entropy contributes
no gradient to this MLP. Advantages can contain future rewards from mode changes,
so these are sampled local gradient comparisons, not an isolated causal test or
proof of why tracking regresses. Interpret cosine together with gradient sizes,
sample counts, repeated measurements and task performance.

Diagnostics use autograd.grad and a private random generator, restore the actor's
distribution object, and do not alter gradients, optimizer state, advantages or
training random streams. They do not project or modify the training update.
The current diagnostic supports the task's single-GPU, globally normalized
advantage configuration. A resume starts a new diagnostic measurement count.
