"""Deterministic full-start evaluation; zero policy is an entry-point smoke only."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/transition_t_mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/transition_t_cache')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch


class EndpointHold:
  """Continuous endpoint criteria, with no credit for intermediate RSI starts."""
  def __init__(self, dt):
    self.dt = dt
    self.duration = 0.

  def update(self, completed, orientation_error, height_error, feet_rms,
             linear_speed, angular_speed, rear_supported, front_clear):
    good = (completed and orientation_error < .25 and height_error < .05
            and feet_rms < .06 and linear_speed < .3 and angular_speed < .8
            and rear_supported and front_clear)
    self.duration = self.duration + self.dt if good else 0.
    return self.duration >= 1. - 1e-9


def json_safe(value):
  """Keep numerical failure episodes serializable without hiding their causes."""
  if isinstance(value, dict):
    return {key: json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_safe(item) for item in value]
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--agent', choices=('zero', 'trained'), default='trained')
  parser.add_argument('--checkpoint-file', type=Path)
  parser.add_argument('--motion-file', type=Path)
  parser.add_argument('--episodes-per-group', type=int, default=20)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--output', type=Path, default=Path('outputs/transition_t_evaluation.json'))
  args = parser.parse_args()
  if args.episodes_per_group < 1:
    parser.error('--episodes-per-group must be positive')
  if args.agent == 'trained' and args.checkpoint_file is None:
    parser.error('--agent trained requires --checkpoint-file')
  if args.device == 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = ''
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import TerminationTermCfg
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.lab_api.math import quat_error_magnitude
  from src.tasks.transition_t.env_cfg import transition_env_cfg
  from src.tasks.transition_t.rl import TransitionOnPolicyRunner, transition_ppo_runner_cfg
  torch.set_num_threads(1)
  groups = {}
  for group_index, group in enumerate(('nominal', 'base', 'feet', 'combined')):
    cfg = transition_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed + group_index
    cmd = cfg.commands['motion']
    training = transition_env_cfg().commands['motion']
    cmd.pose_range = training.pose_range if group in ('base', 'combined') else {}
    cmd.velocity_range = training.velocity_range if group in ('base', 'combined') else {}
    cmd.foot_range = training.foot_range if group in ('feet', 'combined') else (0., 0., 0.)
    cmd.undisturbed_probability = 0. if group != 'nominal' else 1.
    cmd.sampling_mode = 'start'
    if args.motion_file: cmd.motion_file = str(args.motion_file)
    latest = {}

    def capture_before_reset(env):
      term = env.command_manager.get_term('motion')
      robot = env.scene['robot']
      feet_ids = robot.find_bodies(tuple(f'{leg}_foot' for leg in ('FL', 'FR', 'RL', 'RR')), preserve_order=True)[0]
      ref_ids = [term.cfg.body_names.index(f'{leg}_foot') for leg in ('FL', 'FR', 'RL', 'RR')]
      feet = robot.data.body_link_pos_w[:, feet_ids]
      ref_feet = term.body_pos_w[:, ref_ids]
      contact = env.scene['feet_ground_contact'].data
      forces = contact.force.reshape(1, 4, -1, 3).sum(2).norm(dim=-1)
      found = contact.found.reshape(1, 4, -1).any(-1)
      latest.update(
        completed=bool(term.time_steps[0] >= term.motion.time_step_total - 1),
        orientation_error=float(quat_error_magnitude(term.anchor_quat_w, term.robot_anchor_quat_w)[0]),
        height_error=float((term.anchor_pos_w[0, 2] - term.robot_anchor_pos_w[0, 2]).abs()),
        feet_rms=float((feet-ref_feet).square().sum(-1).mean().sqrt()),
        linear_speed=float(term.robot_anchor_lin_vel_w[0].norm()),
        angular_speed=float(term.robot_anchor_ang_vel_w[0].norm()),
        rear_supported=bool((found[0, 2:] & (forces[0, 2:] > 5.)).all()),
        front_clear=bool((~found[0, :2]).all() and (feet[0, :2, 2] > .032).all()),
      )
      return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Capture terminal state before the manager automatically initializes a new episode.
    cfg.terminations['evaluation_capture'] = TerminationTermCfg(func=capture_before_reset)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
      policy = None
      if args.agent == 'trained':
        agent = transition_ppo_runner_cfg()
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = TransitionOnPolicyRunner(wrapped, asdict(agent), device=args.device)
        runner.load(str(args.checkpoint_file), load_cfg={'actor': True}, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
      episodes = []
      term = env.command_manager.get_term('motion')
      for episode in range(args.episodes_per_group):
        obs, _ = env.reset(seed=args.seed + group_index * 100000 + episode)
        robot = env.scene['robot']
        ref_ids = [term.cfg.body_names.index(f'{leg}_foot') for leg in ('FL', 'FR', 'RL', 'RR')]
        foot_ids = robot.find_bodies(tuple(f'{leg}_foot' for leg in ('FL', 'FR', 'RL', 'RR')), preserve_order=True)[0]
        initial = dict(
          root_position_delta=(term.robot_anchor_pos_w-term.anchor_pos_w)[0].tolist(),
          root_orientation_error=float(quat_error_magnitude(term.robot_anchor_quat_w, term.anchor_quat_w)[0]),
          linear_velocity_delta=(term.robot_anchor_lin_vel_w-term.anchor_lin_vel_w)[0].tolist(),
          angular_velocity_delta=(term.robot_anchor_ang_vel_w-term.anchor_ang_vel_w)[0].tolist(),
          foot_position_delta=(robot.data.body_link_pos_w[:, foot_ids]-term.body_pos_w[:, ref_ids])[0].tolist(),
          ik_fallback=bool(term.metrics['ik_fallback'][0]),
          perturbation_scale=float(term.metrics['initial_scale'][0]),
        )
        hold = EndpointHold(env.step_dt)
        completed = held = False
        errors = []
        causes = []
        for step in range(math.ceil(cfg.episode_length_s/env.step_dt) + 2):
          with torch.inference_mode():
            actions = policy(obs) if policy else torch.zeros(1, 12, device=args.device)
          obs, _, terminated, truncated, _ = env.step(actions.clamp(-10., 10.))
          completed |= latest['completed'] and not bool(terminated[0])
          held = hold.update(**latest) and not bool(terminated[0])
          errors.append([latest[key] for key in ('orientation_error', 'height_error', 'feet_rms')])
          if bool(terminated[0] or truncated[0]):
            causes = [name for name in env.termination_manager.active_terms
                      if bool(env.termination_manager.get_term(name)[0])]
            break
        episodes.append(dict(episode=episode, full_motion_completed=completed,
          endpoint_held=held, elapsed_s=(step+1)*env.step_dt, termination_causes=causes,
          mean_errors=np.mean(errors, axis=0).tolist(), max_errors=np.max(errors, axis=0).tolist(),
          nonfinite_state_detected=not bool(np.isfinite(errors).all()), initial=initial))
      groups[group] = dict(
        episodes=len(episodes), full_motion_completion_rate=float(np.mean([e['full_motion_completed'] for e in episodes])),
        endpoint_hold_rate=float(np.mean([e['endpoint_held'] for e in episodes])),
        ik_fallback_rate=float(np.mean([e['initial']['ik_fallback'] for e in episodes])),
        termination_causes=dict(Counter(c for e in episodes for c in e['termination_causes'])),
        records=episodes)
      motion_sha256 = term.reference.sha256
    finally:
      env.close()
  report = dict(agent=args.agent, runtime_smoke_only=args.agent == 'zero', seed=args.seed,
                motion_sha256=motion_sha256, error_columns=['orientation_rad', 'height_m', 'feet_rms_m'],
                groups=groups)
  report = json_safe(report)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
  print(json.dumps({g: {k: v for k, v in r.items() if k != 'records'} for g, r in groups.items()}, indent=2))
  print(f'Evaluation written to {args.output}')


if __name__ == '__main__': main()
