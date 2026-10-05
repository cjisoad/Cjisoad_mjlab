# Teacher body-Y anchor and reward audit

## Update on 2026-10-06

The approved follow-up removes `ang_xz` only from `pedipulation_t`, in both
training and play. The teacher now has 25 rewards; the original pedipulation
task retains 26 and the locomotion teacher retains 19. Existing gravity-based
`handstand_orientation` (-1.) and `orientation_symmetry` (-.5) remain.
All other reward functions, weights, parameters and relative order are retained.
The audit script now compares the remaining 44 teacher rewards and explicitly
records the removed source reward. The 45-reward/GPU evidence below describes
the earlier `f05df8d` anchor change, before this removal.

The follow-up does not modify either running quadruped training snapshot.

## Approved change

Both `pedipulation_t` and `loco_pedipulation_t` use
`forward = normalize(body_Y_world × gravity_up)`, `left = up × forward`.
The module is teacher-specific. Original task and current student anchors
are unchanged. The three previously restored loco physics settings are retained.
Checkpoint contract is version 2 with `teacher_body_y_cross_up_v1`; old X-Z
teacher models cannot be resumed under the new frame.

## Reward effects

Weights, parameters, ordering, timestep scaling and original reward functions
remain unchanged: 26 standing rewards and 19 locomotion rewards.

| Consumer | Direct effect at a fixed physical state and fixed command |
|---|---|
| Biped `tracking_lin_vel`; loco `track_linear_velocity` | Changes for nonzero horizontal commands, because actual vx/vy are expressed in the new frame. Zero horizontal commands use a norm and remain invariant. |
| Biped `FR_pos_track`; loco `fr_position_tracking` | Changes because the same Cartesian offset/reference is decoded through the new horizontal basis. |
| Biped `tracking_ang_vel`; loco `track_angular_velocity` | Invariant for fixed wz: gravity-up is unchanged; the horizontal angular norm is rotation invariant. |
| Biped `lin_vel_z`, `ang_vel_xy` | Invariant: same up component and horizontal norm. |
| Height, world foot height/clearance, contact/landing/slip | Invariant: unchanged world heights, horizontal speed norms, contacts and force magnitudes. |
| Orientation/gravity, Euler roll, joint pose/symmetry/limits, torque/acceleration/action change | Invariant: these read body/world/joint data, not the anchor's horizontal axes. |
| Standing automatic heading controller | Changes generated wz through the new heading; consequently angular reward can change when commands are regenerated. Manual wz remains preserved. |
| Actor added velocity3, critic exact velocity, loco FR privileged state and curriculum metrics | Coordinate values follow the same new command basis; no new channels or normalization. |
| Biped symmetry regularization | Existing reflection signs remain valid; the new basis respects sagittal reflection. Nominal stance-command exception is retained. |

"Invariant" here refers to the same physical state, command and reward history.
Future actions and physical trajectories will change, so later training values
of all reward terms can differ indirectly.

## Foot targets and exceptional poses

Both frames agree at the level quadruped and fully reared nominal FK reference
poses used to build the target bank. Keep the existing quadruped zero,
nominal_biped_offset, LOW/HIGH banks and ten-centimeter height overlap.
The shared goal decoding remains `base + basis @ (zero + offset)`.
Away from those reference poses, the goal direction intentionally changes.

The new singularity is body Y parallel to up (side-fall orientation). A
deterministic projected world-X/world-Y fallback keeps values finite, without
claiming heading continuity. `Episode_Metrics/anchor_degenerate_fraction`
records fallback incidence and does not contribute reward.

## Verification

- 74 focused CPU regression checks pass, including new roll/nose-down geometry,
  tilted gravity, common goal decoding, fixed-command tracking, symmetry and
  old-checkpoint rejection. Original pedipulation/loco/student anchor tests pass.
- Actual CPU MuJoCo/Warp state comparison covers every reward in both tasks,
  using 8 environments across level, rolled, intermediate and near-upright poses.
  All 45 terms are finite. No reward outside horizontal tracking/FR tracking
  changes above absolute tolerance 2e-5.
- Near-target FR comparisons also exercise nonsaturated reward values; their
  change confirms that goal decoding follows the new basis.
- The same 45-reward comparison passes on the new RTX 4090 with 64 environments
  per task and no unexpected reward changes above tolerance.
- Both teachers pass CUDA integration: two full PPO updates each at 64
  environments, finite observations/rewards/losses, actual actor parameter
  updates, checkpoint save/load roundtrip, preserved loco curriculum state,
  and rejection of incompatible frame/action contracts.
- The source biped `base_height` writes one batch-wide height gate, while
  `ang_xz` executes before that write. The audit restores the same previous
  height score between comparisons. This existing ordering/history is retained.
- A separate CUDA check of the unchanged Euler conversion, with no anchor
  involved, finds a singularity at exactly pitch -pi/2: the same quaternion
  gives absolute Euler roll 2.896614 on its first call and 2.853531 on subsequent
  calls (difference 0.043083 rad). At pitch -pi/2 + .01, repeated roll is stable
  at .250000 rad. The biped `ang_xz` penalty therefore remains numerically
  sensitive at exact upright. It is not caused or corrected by this anchor
  change. The reward comparison uses near-upright poses to isolate coordinate
  effects; exact-upright frame geometry is covered separately.

Reproduce with:

```bash
MUJOCO_GL=disable MPLCONFIGDIR=/tmp/mjlab_matplotlib PYTHONPATH=.:tests \
  python -m unittest -v test_teacher_anchor test_teacher_tasks \
  test_leg_manip_anchor test_pedipulation test_loco_pedipulation

MUJOCO_GL=disable PYTHONPATH=. python scripts/audit_teacher_anchor.py \
  --device cpu --num-envs 8 --output /tmp/teacher_anchor_audit.json
```

This establishes implementation consistency, not improved learned policy quality.
The old-anchor run on SSH port 42293 remains running for comparison; the new
quadruped teacher is assigned to the newly supplied server on port 42070.
