"""Small contact cost and terminal balance-step-friendly velocity tracking."""
import torch


def rise_height_reward(env, command_name, std=.06, activation_height=.02, ramp_height=.04):
  """Continuous symmetric height tracking after the reference starts rising.

  Activation depends on reference height, never on the policy crossing a
  success threshold. Reference-relative activation also survives RSI resets.
  """
  cmd = env.command_manager.get_term(command_name)
  reference_z = cmd.motion.body_pos_w[:, cmd.motion_anchor_body_index, 2]
  rise = reference_z[cmd.time_steps] - reference_z[0]
  phase = ((rise-activation_height)/ramp_height).clamp(0., 1.)
  phase = phase.square()*(3.-2.*phase)
  error = cmd.anchor_pos_w[:, 2]-cmd.robot_anchor_pos_w[:, 2]
  return phase*torch.exp(-(error/std).square())


def knee_ground_cost(env, force_scale=50.):
  # Per substep, the worst non-foot leg contact, then average substeps.
  force = env.scene['knees_ground_contact'].data.force_history.norm(dim=-1)
  return (force.amax(dim=1)/force_scale).clamp(0., 1.).mean(dim=-1)


def terminal_body_velocity_reward(env, command_name, std, body_names=None, angular=False):
  cmd = env.command_manager.get_term(command_name)
  names = cmd.cfg.body_names
  ids = [names.index(n) for n in body_names] if body_names is not None else list(range(len(names)))
  field = 'ang_vel_w' if angular else 'lin_vel_w'
  error = (getattr(cmd, 'body_'+field)[:, ids]-getattr(cmd, 'robot_body_'+field)[:, ids]).square().sum(-1)
  terminal = cmd.time_steps >= cmd.motion.time_step_total-1
  # Keep the original averaging denominator. Root coefficient never changes;
  # terminal leg coefficients are 0.2 so balancing steps can earn credit.
  weights = error.new_ones(error.shape)
  for i, body_id in enumerate(ids):
    if names[body_id] != 'base_link':
      weights[:, i] = torch.where(terminal, .2, 1.)
  return torch.exp(-(error*weights).mean(-1)/std**2)
