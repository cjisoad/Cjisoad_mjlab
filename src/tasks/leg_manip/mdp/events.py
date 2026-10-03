"""Task-local perturbations with fresh derived velocities before observations."""
import torch
from src.tasks.velocity.mdp import push_by_setting_velocity
from .commands import QUAD, HOLD


def push_robot(env, env_ids, velocity_range):
  term = env.command_manager.get_term('twist')
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  if term.stage == 0:
    if not term.cfg.stage0_light_push:
      return
    stable = (((term.phase == QUAD) & ~term.mode) | ((term.phase == HOLD) & term.mode)) & (term.elapsed >= 1.)
    ids = ids[stable[ids]]
    ranges = {'x': (-.05, .05), 'y': (-.05, .05), 'yaw': (-.05, .05)}
  else:
    # At the first walking level use light pushes, increasing with proficiency.
    factor = .3333333333333333 + (2. / 3.) * min(getattr(term, 'velocity_level', 3), 3) / 3.
    ranges = {key: (low * factor, high * factor) for key, (low, high) in velocity_range.items()}
  if len(ids):
    push_by_setting_velocity(env, ids, ranges)
    env.sim.forward()
