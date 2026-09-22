"""48D actor and 89D critic with velocity and FR position commands."""

import torch


def policy_state(env, add_noise=True):
  robot = env.scene['robot'].data
  action = env.action_manager.get_term('joint_pos')
  command = env.command_manager.get_command('stand')
  obs = torch.cat((robot.root_link_ang_vel_b * .25, robot.projected_gravity_b,
    command * command.new_tensor((2., 2., .25, 1., 1., 1.)),
    action.joint_pos - action.default_angles, action.joint_vel * .05,
    action.raw_action), dim=-1)
  if add_noise:
    amplitude = obs.new_tensor((.05,) * 6 + (0.,) * 6 + (.01,) * 12 + (.075,) * 12 + (0.,) * 12)
    obs = obs + (torch.rand_like(obs) * 2 - 1) * amplitude
  action.actor_sample = obs.clamp(-100., 100.)
  return action.actor_sample


def shared_policy_state(env):
  # The critic must see the exact actor noise draw, not independently corrupted data.
  return env.action_manager.get_term('joint_pos').actor_sample


def base_velocity(env):
  velocity = env.command_manager.get_term('stand').anchor_frame.to_frame(env.scene['robot'].data.root_link_lin_vel_w)
  return (velocity * 2.).clamp(-100., 100.)


def domain_parameters(env):
  return env.action_manager.get_term('joint_pos').dr_observation.clamp(-100., 100.)


def foot_contact(env):
  front = env.scene['front_contact'].data.force.norm(dim=-1) > 1.0
  rear = env.scene['rear_contact'].data.force.norm(dim=-1) > 1.0
  return torch.cat((front, rear), dim=-1).float()
