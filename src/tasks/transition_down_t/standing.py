"""Tripod endpoint metrics with the original transition hold policy."""
import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

STANDING_HOLD_SECONDS = 1.
STANDING_BAD_SAMPLE_LIMIT = 3


def standing_hold_policy():
  return dict(hold_s=STANDING_HOLD_SECONDS, bad_sample_limit=STANDING_BAD_SAMPLE_LIMIT,
    bad_sample_credit='pause', success_requires_good_sample=True,
    physical_failure='immediate', strict_endpoint='uninterrupted')


def advance_standing_hold(duration, bad_samples, good, *, eligible, active, dt):
  """Pause credit for two bad samples, resetting on the third.

  Ineligible active worlds clear immediately (e.g. reference incomplete or
  physical failure); inactive worlds retain their captured state. Call once per
  control step. A recovered good sample both earns credit and clears the streak.
  """
  next_bad = torch.where(good | ~eligible, 0, bad_samples+1).clamp(max=STANDING_BAD_SAMPLE_LIMIT)
  next_duration = torch.where(eligible,
    torch.where(good, duration+dt,
      torch.where(next_bad >= STANDING_BAD_SAMPLE_LIMIT, 0., duration)), 0.)
  return torch.where(active, next_duration, duration), torch.where(active, next_bad, bad_samples)


def tripod_conditions(metrics):
  """Contact/posture handoff; global foot accuracy is reported separately."""
  return dict(orientation=metrics['orientation_error'] < .25,
    height=metrics['height_error'] < .05, linear_speed=metrics['linear_speed'] < .3,
    angular_speed=metrics['angular_speed'] < .8, tripod_support=metrics['tripod_supported'],
    fr_clear=metrics['fr_clear'], joint_speed=metrics['joint_speed'] < 20.)


def strict_tripod_endpoint_candidate(completed, metrics):
  # Same criteria as evaluate_transition_t.py; joint speed is a handoff-only gate.
  return (completed & (metrics['orientation_error'] < .25) & (metrics['height_error'] < .05)
    & (metrics['feet_rms'] < .06) & (metrics['linear_speed'] < .3)
    & (metrics['angular_speed'] < .8) & metrics['tripod_supported'] & metrics['fr_clear'])


def tripod_endpoint_metrics(env):
  motion = env.command_manager.get_term('motion')
  robot = env.scene['robot']
  names = tuple(f'{leg}_foot' for leg in ('FL', 'FR', 'RL', 'RR'))
  feet_ids = robot.find_bodies(names, preserve_order=True)[0]
  ref_ids = [motion.cfg.body_names.index(name) for name in names]
  feet = robot.data.body_link_pos_w[:, feet_ids]
  contact = env.scene['feet_ground_contact'].data
  forces = contact.force.reshape(env.num_envs, 4, -1, 3).sum(2).norm(dim=-1)
  found = contact.found.reshape(env.num_envs, 4, -1).any(-1)
  return dict(orientation_error=quat_error_magnitude(motion.anchor_quat_w, motion.robot_anchor_quat_w),
    height_error=(motion.anchor_pos_w[:, 2]-motion.robot_anchor_pos_w[:, 2]).abs(),
    feet_rms=(feet-motion.body_pos_w[:, ref_ids]).square().sum(-1).mean(-1).sqrt(),
    linear_speed=motion.robot_anchor_lin_vel_w.norm(dim=-1),
    angular_speed=motion.robot_anchor_ang_vel_w.norm(dim=-1),
    tripod_supported=(found[:, [0, 2, 3]] & (forces[:, [0, 2, 3]] > 5.)).all(-1),
    fr_clear=~found[:, 1] & (feet[:, 1, 2] > .032),
    joint_speed=robot.data.joint_vel.norm(dim=-1))
