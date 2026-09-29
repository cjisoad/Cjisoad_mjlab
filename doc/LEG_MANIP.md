# Leg Manip

`leg_manip` fuses `loco_pedipulation` (quadruped/tripod locomotion with
right-front foot control) and `Pedipulation` (rear-leg stand with FR targets)
into one task. A single PPO policy controls all 12 joints across three support
modes: quadruped, tripod (FR manipulating), and biped (rear-leg stand, FR
manipulating). The two source tasks remain registered and unchanged.

## Robot and Policy ABI

- Robot: the Pedipulation stand spec snapshot (`src/tasks/pedipulation/robot.py`),
  one asset for all modes. Quadruped init: `DEFAULT_ANGLES` at 0.42 m.
- Actor input: the unchanged 48-value layout shared by both source tasks
  (`ang_vel*.25` 3, `projected_gravity` 3, `command*(2,2,.25,1,1,1)` 6,
  `joint_pos_rel` 12, `joint_vel*.05` 12, `prev_action` 12), same noise and
  ±100 clamp. **No new actor inputs were added**; mode, group weight and phase
  are internal/critic-only state.
- Command: the unchanged 6-value ABI `[vx, vy, wz, dx, dy, dz]`; all-zero
  `[dx, dy, dz]` is the no-target sentinel.
- Action: `PedipulationPositionAction` (scale .25, substep delay, motor DR
  interface), a superset of the loco action term.
- Critic: 131 values = shared noisy actor sample (48) + anchor linear
  velocity (3) + DR parameters (34) + per-foot contacts (4) + gait phase (2) +
  FR/mode privileged state (20) + foot height (4) + air time (4) + contact
  force logs (12).

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
  pitch sweep; it only degenerates near a 45° nose-down pose (far outside the
  terminated region), where a fixed reference keeps values finite.

Velocity commands, FR targets, velocity observations and all velocity rewards
use this frame with the origin at `base_link`.

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

Shared: `fr_position_tracking` 2.5 (tracks the smoothstep reference × blend),
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
batch height gate): `biped_base_height` 1.5 (0.52 m), `biped_tracking_lin_vel`
2.5, `biped_tracking_ang_vel` 2.5, `biped_lin_vel_z` 0.3, `biped_ang_vel_xy`
0.2, `handstand_orientation` −1.0, `handstand_feet_on_air` 0.4,
`handstand_feet_height_exp` 5.0, `biped_feet_clearance` 0.4, `biped_contact`
0.3, `biped_ang_xz` −0.5, `symmetric_joints` −0.1, `orientation_symmetry` −0.5,
`feet_height_symmetry` −0.2, `default_pos_front` −0.1, `default_pos_rear` −0.1,
`default_hip_pos` −0.1, `default_pos_reward_FL` 0.5, `default_pos_reward_FR`
0.5. The Pedipulation `FR_pos_track` is replaced by the shared
`fr_position_tracking` (same 2.5 / std .05).

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
- Curriculum easy subset: LOW targets with dz ≤ 0.12 (9 targets).
- Automatic samples come from the bank; manual (keyboard/GUI) targets stay
  continuous. Above-band samples use biped velocity ranges
  (vx ∈ [−0.2, 0.6], vy = 0, wz ∈ [−0.6, 0.6]).

## FR Phase Machine

Unchanged from loco_pedipulation:
`QUAD→PREPARE→REACH→HOLD→LOWER→RECOVER→QUAD`. While biped, only REACH and
HOLD occur (retargeting included); PREPARE/LOWER/RECOVER only appear in
quadruped↔tripod transitions. Landing reference height follows the live base
height, so a LOWER started from a rear-up pose still converges as the loco
group pulls the robot down; `landing_failed` (2 s timeout) still terminates.

## Terminations

`time_out`, `base_contact`, `illegal_contact` (non-foot vs terrain >10 N),
`landing_failed`, and mode-aware `fell_over`: LOCO mode keeps the 60° tilt
limit; BIPED mode only limits the lateral lean (`|gravity_b.y| > sin 60°`),
allowing the full 90° pitch of the stand.

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

Stages 0–3 match loco_pedipulation (quadruped → standing FR targets → slow
tripod → full tripod) with the same promotion statistics; stage 3 → 4 reuses
the stage 2 → 3 criteria. Stage 4 (`STAGE_BIPED`) unlocks above-band bank
samples. Curriculum state is checkpointed by the runner. Play configs start at
stage 4.

## Training

```
python scripts/train.py leg_manip --env.scene.num-envs=4096
```

Runner: `LegManipOnPolicyRunner` (loco curriculum persistence + manipulation
metrics logging). Algorithm: `LegManipPPO` — the Pedipulation symmetry PPO
(`sym_coef=1.0`, target-free samples only) with the critic-width check relaxed
to the merged 131-D layout. `experiment_name=leg_manip`, wandb logger,
15000 iterations, `clip_actions=10`.

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

- 172 tests pass (`python -m unittest <all tests.test_* modules>`), covering
  the unified frame, band constants, two-segment workspace, hysteresis/ramp,
  phase machine, reward gating, terminations, cfg assembly, a real CPU env
  build/step, RL registration and keyboard wiring.
- Legacy compatibility probe: the loco_pedipulation `model_14999.pt` actor
  (same 48→512→256→128→12 architecture) loads into the leg_manip runner with
  `strict=True`; a 550-step CPU rollout with terminations relaxed confirmed
  command flow, band latch, group ramp and phase sequence. Biped behavior
  itself requires training on this task.
