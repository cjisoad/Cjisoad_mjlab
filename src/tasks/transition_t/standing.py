"""Upright position hold shared by held-out trials and teacher handoff."""
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat

STANDING_HOLD_SECONDS = 1.
STANDING_BAD_SAMPLE_LIMIT = 1
POSITION_RADIUS = .08


def standing_hold_policy(position_radius=POSITION_RADIUS):
  return dict(hold_s=STANDING_HOLD_SECONDS, bad_sample_limit=STANDING_BAD_SAMPLE_LIMIT,
    bad_sample_credit='reset', success_requires_good_sample=True,
    physical_failure='immediate', objective='upright_position_hold_v2',
    xy_radius_m=position_radius, orientation_limit_rad=.25, height_limit_m=.05,
    target='fixed_aligned_reference_endpoint', contact_and_speed='diagnostic_only')


def advance_standing_hold(duration, bad_samples, good, *, eligible, active, dt):
  """Accumulate continuous good time, resetting on any out-of-bound sample.

  Ineligible active worlds clear immediately (e.g. reference incomplete or
  physical failure); inactive worlds retain their captured state. Call once per
  control step. A recovered good sample both earns credit and clears the streak.
  """
  next_bad = torch.where(good | ~eligible, 0, bad_samples+1).clamp(max=STANDING_BAD_SAMPLE_LIMIT)
  next_duration = torch.where(eligible,
    torch.where(good, duration+dt,
      torch.where(next_bad >= STANDING_BAD_SAMPLE_LIMIT, 0., duration)), 0.)
  return torch.where(active, next_duration, duration), torch.where(active, next_bad, bad_samples)


def standing_conditions(metrics, position_radius=POSITION_RADIUS):
  """Allow stepping; require upright posture near the fixed endpoint."""
  return dict(orientation=metrics['orientation_error'] < .25,
    height=metrics['height_error'] < .05, position=metrics['xy_error'] <= position_radius)


def endpoint_metrics(env):
  motion = env.command_manager.get_term('motion')
  robot = env.scene['robot']
  names = tuple(f'{leg}_foot' for leg in ('FL', 'FR', 'RL', 'RR'))
  feet_ids = robot.find_bodies(names, preserve_order=True)[0]
  ref_ids = [motion.cfg.body_names.index(name) for name in names]
  feet = robot.data.body_link_pos_w[:, feet_ids]
  contact = env.scene['feet_ground_contact'].data
  forces = contact.force.reshape(env.num_envs, 4, -1, 3).sum(2).norm(dim=-1)
  found = contact.found.reshape(env.num_envs, 4, -1).any(-1)
  endpoint = (quat_apply(motion.reference_rotation,
    motion.motion.body_pos_w[-1, motion.motion_anchor_body_index].expand(env.num_envs, -1))
    + motion.reference_translation + env.scene.env_origins)
  # Pre-reset evaluation capture runs before command updates. Rebuild the
  # aligned target from this physical sample without changing command state.
  alignment = yaw_quat(quat_mul(motion.robot_anchor_quat_w, quat_inv(motion.anchor_quat_w)))
  origin = motion.robot_anchor_pos_w.clone()
  origin[:, 2] = motion.anchor_pos_w[:, 2]
  aligned_feet = origin[:, None] + quat_apply(alignment[:, None].expand(-1, 4, -1),
    motion.body_pos_w[:, ref_ids]-motion.anchor_pos_w[:, None])
  aligned_error = (feet-aligned_feet).norm(dim=-1)
  return dict(orientation_error=quat_error_magnitude(motion.anchor_quat_w, motion.robot_anchor_quat_w),
    height_error=(motion.anchor_pos_w[:, 2]-motion.robot_anchor_pos_w[:, 2]).abs(),
    xy_error=(endpoint[:, :2]-motion.robot_anchor_pos_w[:, :2]).norm(dim=-1),
    feet_rms=(feet-motion.body_pos_w[:, ref_ids]).square().sum(-1).mean(-1).sqrt(),
    linear_speed=motion.robot_anchor_lin_vel_w.norm(dim=-1),
    horizontal_speed=motion.robot_anchor_lin_vel_w[:, :2].norm(dim=-1),
    angular_speed=motion.robot_anchor_ang_vel_w.norm(dim=-1),
    rear_supported=(found[:, 2:] & (forces[:, 2:] > 5.)).all(-1),
    rear_any_contact=found[:, 2:].any(-1),
    front_clear=(~found[:, :2]).all(-1) & (feet[:, :2, 2] > .032).all(-1),
    joint_speed=robot.data.joint_vel.norm(dim=-1),
    **{f'{leg}_aligned_error': aligned_error[:, i] for i, leg in enumerate(('FL','FR','RL','RR'))})
