"""Startup physical randomization, episodic reset and global velocity pushes."""

import math

import torch

from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel


def restitution_damping_ratio(restitution):
  """Damped-contact approximation, not an exact PhysX restitution conversion."""
  log_e = restitution.clamp_min(1.e-6).log()
  ratio = -log_e / (torch.pi ** 2 + log_e.square()).sqrt()
  return torch.where(restitution <= 0, torch.ones_like(ratio), ratio)


@requires_model_fields('body_mass', 'body_inertia', 'body_ipos', 'geom_friction',
  'geom_solref', 'dof_armature', 'dof_frictionloss', 'dof_damping',
  recompute=RecomputeLevel.set_const)
def randomize_physics(env, env_ids):
  robot = env.scene['robot']
  state = env.action_manager.get_term('joint_pos')
  model = env.sim.model
  n = env.num_envs

  def sample(low, high, width=1):
    return torch.empty(n, width, device=env.device).uniform_(low, high)

  friction = sample(.2, 1.2)
  restitution = sample(0., .3)
  added_mass = sample(-1., 2.)
  com_offset = sample(-.05, .05, 3)
  state.kp_multipliers.copy_(sample(.9, 1.1, 12))
  state.kd_multipliers.copy_(sample(.9, 1.1, 12))
  state.motor_offsets.copy_(sample(-.035, .035, 12))
  joint_friction = sample(.01, .1)
  joint_damping = sample(0., .1)
  armature = sample(.003, .08)

  body_ids = robot.indexing.body_ids
  base_local = robot.find_bodies('base_link')[0][0]
  base_id = body_ids[base_local]
  nominal_mass = env.sim.get_default_field('body_mass')[body_ids]
  factors = sample(.9, 1.1, len(body_ids))
  factors[:, base_local] = (nominal_mass[base_local] + added_mass[:, 0]) / nominal_mass[base_local]
  model.body_mass[:, body_ids] = nominal_mass[None] * factors
  # Density scaling is MuJoCo's consistent counterpart of inertia recomputation.
  model.body_inertia[:, body_ids] = env.sim.get_default_field('body_inertia')[body_ids][None] * factors[..., None]
  model.body_ipos[:, base_id] = env.sim.get_default_field('body_ipos')[base_id] + com_offset

  geom_ids = robot.indexing.geom_ids
  model.geom_friction[:, geom_ids, 0] = friction
  model.geom_solref[:, geom_ids, 0] = .02
  model.geom_solref[:, geom_ids, 1] = restitution_damping_ratio(restitution)
  dofs = robot.indexing.joint_v_adr
  model.dof_armature[:, dofs] = armature
  model.dof_frictionloss[:, dofs] = joint_friction
  model.dof_damping[:, dofs] = joint_damping

  for actuator in robot.actuators:
    canonical = [list(state._entity.joint_names).index(name) for name in actuator.target_names]
    ordered = [state.joint_ids.tolist().index(index) for index in canonical]
    actuator.set_gains(slice(None), 40. * state.kp_multipliers[:, ordered], state.kd_multipliers[:, ordered])

  # Keep the source ABI: its friction slot is zero, and restitution is repeated.
  state.dr_observation.copy_(torch.cat((torch.zeros_like(friction), added_mass,
    com_offset, state.kp_multipliers, state.kd_multipliers, armature,
    joint_friction, joint_damping, restitution, restitution), dim=-1))


def reset_robot(env, env_ids):
  robot = env.scene['robot']
  n = len(env_ids)
  root = robot.data.default_root_state[env_ids].clone()
  root[:, :3] += env.scene.env_origins[env_ids]
  root[:, 7:13] = torch.empty(n, 6, device=env.device).uniform_(-.5, .5)
  q = robot.data.default_joint_pos[env_ids] * torch.empty(n, 12, device=env.device).uniform_(.5, 1.5)
  robot.write_root_state_to_sim(root, env_ids=env_ids)
  robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)


def push_robot(env, env_ids):
  if env.common_step_counter % math.ceil(8. / env.step_dt):
    return
  robot = env.scene['robot']
  velocity = robot.data.root_link_vel_w.clone()
  velocity[:, :2] = torch.empty(env.num_envs, 2, device=env.device).uniform_(-.4, .4)
  velocity[:, 3:6] = torch.empty(env.num_envs, 3, device=env.device).uniform_(-.6, .6)
  robot.write_root_link_velocity_to_sim(velocity)
